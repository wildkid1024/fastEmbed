#!/usr/bin/env python3
"""校验 C++ 实现的 EmbeddingGemma 2 嵌入与 transformers 官方实现是否一致。

用法:
    # 文本
    python verify_embeddinggemma2.py [--model_path PATH] [--cli PATH] [--prompt NAME]
    # 图像（需要 pillow）
    python verify_embeddinggemma2.py --image path/to.jpg [--image path/to2.png ...]

判定标准: 文本 cos > 0.9999 且 max_diff < 1e-4；图像 cos > 0.999
（图像存在缩放重采样差异，阈值略放宽）。
"""
import argparse
import json
import os
import subprocess

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

DEFAULT_MODEL = "/public/Models/google/embeddinggemma-2"
DEFAULT_CLI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "cli")

TEXTS = [
    "Hello world",
    "什么是向量检索？",
    "The northern lights are caused by charged particles from the sun.",
]

# 视觉塔预处理参数（与 processor_config.json 一致）
PATCH_SIZE = 16
POOLING = 3
MAX_SOFT_TOKENS = 280
MAX_PATCHES = MAX_SOFT_TOKENS * POOLING * POOLING
BOS, EOS, BOI, EOI, IMG = 2, 1, 255999, 258882, 258880


def load_prompts(model_path):
    path = os.path.join(model_path, "config_sentence_transformers.json")
    if os.path.exists(path):
        return json.load(open(path)).get("prompts", {})
    return {}


def python_embedding(model, tokenizer, texts, prompt_prefix=""):
    """官方实现：文本塔 + 均值池化 + L2 归一化"""
    inputs = [prompt_prefix + t for t in texts]
    enc = tokenizer(inputs, padding=True, return_tensors="pt")
    with torch.no_grad():
        out = model.language_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.numpy()


def cpp_embeddings(cli_path, model_path, texts, prompt_name=""):
    """通过 cli 获取 C++ 实现的嵌入"""
    cmd = [cli_path, model_path] + ([prompt_name] if prompt_name else [])
    stdin = "\n".join(texts) + "\n"
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI 运行失败: {proc.stderr[-2000:]}")

    results = []
    current = None
    for line in proc.stdout.splitlines():
        if line.startswith("embedding shape:"):
            current = []
            results.append(current)
        elif current is not None and line.strip():
            if line.startswith("请输入文本"):
                current = None
                continue
            try:
                current.extend(float(v) for v in line.split())
            except ValueError:
                current = None
    return results


def cpp_image_embedding(cli_path, model_path, image_path, timeout=3600):
    """通过 cli --image 获取 C++ 实现的图像嵌入"""
    cmd = [cli_path, model_path, "--image", image_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI 图像编码失败: {proc.stderr[-2000:]}")

    values = []
    started = False
    for line in proc.stdout.splitlines():
        if line.startswith("embedding shape:"):
            started = True
            continue
        if not started:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("图像:"):
            continue
        try:
            values.extend(float(v) for v in stripped.split())
        except ValueError:
            continue
    if not values:
        raise RuntimeError("CLI 未返回图像嵌入")
    return torch.tensor(values, dtype=torch.float32)


def gemma_target_size(height, width):
    """与 Gemma4ImageProcessor 的等比缩放规则一致"""
    total_px = height * width
    target_px = MAX_PATCHES * PATCH_SIZE * PATCH_SIZE
    factor = (target_px / total_px) ** 0.5
    side_mult = POOLING * PATCH_SIZE
    th = int(np.floor(factor * height / side_mult)) * side_mult
    tw = int(np.floor(factor * width / side_mult)) * side_mult
    max_side = (MAX_PATCHES // (POOLING * POOLING)) * side_mult
    if th == 0 and tw == 0:
        raise ValueError("图像过小")
    if th == 0:
        th, tw = side_mult, min(int(np.floor(width / height)) * side_mult, max_side)
    elif tw == 0:
        tw, th = side_mult, min(int(np.floor(height / width)) * side_mult, max_side)
    return th, tw


def _resample_coeffs(src_size, dst_size):
    scale = src_size / dst_size
    filter_scale = max(1.0, scale)
    support = 2.0 * filter_scale  # bicubic 的 support
    inv = 1.0 / filter_scale
    coeffs, starts = [], []
    for d in range(dst_size):
        center = (d + 0.5) * scale
        lo = max(0, min(int(center - support + 0.5), src_size))
        hi = max(0, min(int(center + support + 0.5), src_size))
        x = np.arange(lo, hi)
        a = np.abs((x - center + 0.5) * inv)
        k = np.where(a < 1, (1.5 * a - 2.5) * a * a + 1,
                     np.where(a < 2, ((-0.5 * a + 2.5) * a - 4) * a + 2, 0.0))
        coeffs.append(k / k.sum())
        starts.append(lo)
    return coeffs, starts


def reference_resize(image, target_h, target_w):
    """与 torchvision（antialias bicubic，两趟都保持 float）一致的缩放"""
    src_h, src_w = image.shape[0], image.shape[1]
    cx, sx = _resample_coeffs(src_w, target_w)
    tmp = np.zeros((src_h, target_w, 3), dtype=np.float64)
    for x, (k, lo) in enumerate(zip(cx, sx)):
        tmp[:, x, :] = np.tensordot(image[:, lo:lo + len(k), :], k, axes=([1], [0]))
    cy, sy = _resample_coeffs(src_h, target_h)
    out = np.zeros((target_h, target_w, 3), dtype=np.float64)
    for y, (k, lo) in enumerate(zip(cy, sy)):
        out[y, :, :] = np.tensordot(tmp[lo:lo + len(k), :, :], k, axes=([0], [0]))
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


def reference_image_tensors(image_path):
    """参考预处理：等比缩放 -> /255 -> patchify，返回 pixel_values 与位置 id"""
    from PIL import Image  # 延迟导入，避免仅做文本校验时依赖 pillow

    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[0], image.shape[1]
    target_h, target_w = gemma_target_size(height, width)
    if (target_h, target_w) != (height, width):
        image = reference_resize(image, target_h, target_w)

    grid_h, grid_w = target_h // PATCH_SIZE, target_w // PATCH_SIZE
    patches = image.reshape(grid_h, PATCH_SIZE, grid_w, PATCH_SIZE, 3).transpose(0, 2, 4, 1, 3)
    patches = patches.reshape(grid_h * grid_w, -1).astype(np.float32) / 255.0
    xs, ys = np.meshgrid(np.arange(grid_w), np.arange(grid_h), indexing="xy")
    positions = np.stack([xs, ys], -1).reshape(grid_h * grid_w, 2).astype(np.int64)

    n = grid_h * grid_w
    pixel_values = np.zeros((MAX_PATCHES, patches.shape[1]), dtype=np.float32)
    pixel_values[:n] = patches
    position_ids = np.full((MAX_PATCHES, 2), -1, dtype=np.int64)
    position_ids[:n] = positions
    return pixel_values, position_ids, n


def python_image_embedding(model, image_path):
    """官方实现：视觉塔 + soft token 拼接 + 文本塔 + 均值池化 + L2"""
    pixel_values, position_ids, n = reference_image_tensors(image_path)
    ids = [BOS, BOI] + [IMG] * (n // (POOLING * POOLING)) + [EOI, EOS]
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([ids]),
            pixel_values=torch.from_numpy(pixel_values).unsqueeze(0),
            image_position_ids=torch.from_numpy(position_ids).unsqueeze(0),
        )
        hidden = out.last_hidden_state[0]
        pooled = hidden.mean(dim=0)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--cli", default=DEFAULT_CLI)
    parser.add_argument("--prompt", default="", help="prompt_name，例如 SearchQuery / Document")
    parser.add_argument("--image", action="append", default=[], help="待校验的图片路径（可多次指定）")
    parser.add_argument("--text_only", action="store_true", help="只校验文本，跳过图像")
    args = parser.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_path)
    if args.text_only and not args.image:
        cfg.vision_config = None
    cfg.audio_config = None
    model = AutoModel.from_pretrained(args.model_path, config=cfg, dtype=torch.float32)
    model.eval()

    ok = True
    if not (args.image and args.text_only):
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        prefix = load_prompts(args.model_path).get(args.prompt, "") if args.prompt else ""
        if args.prompt and not prefix:
            raise SystemExit(f"未知的 prompt_name: {args.prompt}")

        ref = python_embedding(model, tokenizer, TEXTS, prefix)
        got = cpp_embeddings(args.cli, args.model_path, TEXTS, args.prompt)

        for i, text in enumerate(TEXTS):
            if i >= len(got):
                print(f"[FAIL] {text!r}: C++ 未返回结果")
                ok = False
                continue
            a = torch.tensor(ref[i])
            b = torch.tensor(got[i])
            if a.shape != b.shape:
                print(f"[FAIL] {text!r}: 维度不一致 {tuple(a.shape)} vs {tuple(b.shape)}")
                ok = False
                continue
            cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
            max_diff = float((a - b).abs().max())
            status = "OK  " if (cos > 0.9999 and max_diff < 1e-4) else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"[{status}] {text[:40]!r} cos={cos:.8f} max_diff={max_diff:.3e}")

    for image_path in args.image:
        try:
            a = python_image_embedding(model, image_path)
            b = cpp_image_embedding(args.cli, args.model_path, image_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {image_path}: {exc}")
            ok = False
            continue
        if a.shape != b.shape:
            print(f"[FAIL] {image_path}: 维度不一致 {tuple(a.shape)} vs {tuple(b.shape)}")
            ok = False
            continue
        cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        max_diff = float((a - b).abs().max())
        status = "OK  " if cos > 0.999 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] image {os.path.basename(image_path)} cos={cos:.8f} max_diff={max_diff:.3e}")

    print("\n全部通过 ✅" if ok else "\n存在不一致 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
