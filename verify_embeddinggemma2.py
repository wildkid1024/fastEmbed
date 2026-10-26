#!/usr/bin/env python3
"""校验 C++ 实现的 EmbeddingGemma 2 文本嵌入与 transformers 官方实现是否一致。

用法:
    python verify_embeddinggemma2.py [--model_path PATH] [--cli PATH] [--prompt NAME]

判定标准: 每个句子的余弦相似度 > 0.9999 且最大绝对误差 < 1e-4。
"""
import argparse
import json
import os
import subprocess

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

DEFAULT_MODEL = "/public/Models/google/embeddinggemma-2"
DEFAULT_CLI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "cli")

TEXTS = [
    "Hello world",
    "什么是向量检索？",
    "The northern lights are caused by charged particles from the sun.",
]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--cli", default=DEFAULT_CLI)
    parser.add_argument("--prompt", default="", help="prompt_name，例如 SearchQuery / Document")
    args = parser.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_path)
    cfg.vision_config = None
    cfg.audio_config = None
    model = AutoModel.from_pretrained(args.model_path, config=cfg, dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    prefix = load_prompts(args.model_path).get(args.prompt, "") if args.prompt else ""
    if args.prompt and not prefix:
        raise SystemExit(f"未知的 prompt_name: {args.prompt}")

    ref = python_embedding(model, tokenizer, TEXTS, prefix)
    got = cpp_embeddings(args.cli, args.model_path, TEXTS, args.prompt)

    ok = True
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

    print("\n全部通过 ✅" if ok else "\n存在不一致 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
