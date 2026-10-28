#!/usr/bin/env python3
"""PyTorch 基线：与 scripts/bench_http.py 同输入、同方法的进程内延迟测量。

用途：给 fastEmbed 的 C++/CUDA 实现提供同机同模型的对照（PyTorch 走 cuBLAS + Tensor Core）。
注意：本脚本测的是**进程内**前向（不含 HTTP 序列化），对 PyTorch 略有利，对比时应说明。

用法:
    python3 scripts/bench_torch.py \
        --model-path /Models/models-hf/sentence_models/bge-large-zh-v1.5 \
        --model-type bert --levels 5,15,30,60,128 --reps 10
"""
import argparse
import statistics
import sys
import time

import torch
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer

BASE = "Artificial intelligence is transforming the way we build software."


def build_text(tok: Tokenizer, target: int) -> str:
    text = BASE
    while len(tok.encode(text).ids) < target:
        text += " word"
    return text


def pool_output(last_hidden, attention_mask, model_type: str):
    """按模型家族取句向量（仅影响数值，不影响延迟测量）。"""
    if model_type == "bert":
        vec = last_hidden[:, 0]                     # CLS
    elif model_type == "last_token":
        # Qwen3-Embedding：取最后一个非 pad token
        idx = attention_mask.sum(dim=1) - 1
        vec = last_hidden[torch.arange(last_hidden.size(0)), idx]
    else:                                           # mean
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        vec = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(vec, p=2, dim=-1)


def linear_fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return my - slope * mx, slope


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-type", default="bert",
                    choices=["bert", "last_token", "mean"], help="池化方式，仅影响数值")
    ap.add_argument("--tokenizer", default=None, help="默认用模型目录下的 tokenizer.json")
    ap.add_argument("--levels", default="5,15,30,60,128")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    # 直接用 tokenizers 库编码，绕开 HF tokenizer 的版本差异（部分模型目录的 tokenizer_config
    # 在个别 transformers 版本上会报错），也保证与 bench_http.py 的 token 口径完全一致。
    tok = Tokenizer.from_file(args.tokenizer or f"{args.model_path}/tokenizer.json")

    model = AutoModel.from_pretrained(args.model_path, dtype=torch.float32)
    model = model.to(device).eval()
    torch.set_grad_enabled(False)

    print(f"# torch {torch.__version__} · {torch.cuda.get_device_name(0)} · fp32 · batch=1 · 进程内前向")
    print(f"# model={args.model_path.split('/')[-1]}  pool={args.model_type}  reps={args.reps}\n")
    print("| 目标 tok | 实际 tok | min(ms) | median(ms) | mean(ms) | p95(ms) | max(ms) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    points = []
    for target in [int(x) for x in args.levels.split(",")]:
        text = build_text(tok, target)
        ids = torch.tensor([tok.encode(text).ids], device=device)
        attn = torch.ones_like(ids)
        n_tok = int(ids.shape[1])

        def run_once() -> float:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(input_ids=ids, attention_mask=attn)
                _ = pool_output(out.last_hidden_state, attn, args.model_type)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000

        for _ in range(args.warmup):
            run_once()
        samples = sorted(run_once() for _ in range(args.reps))
        p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
        med = statistics.median(samples)
        print(f"| {target} | {n_tok} | {samples[0]:.1f} | {med:.1f} | "
              f"{statistics.mean(samples):.1f} | {p95:.1f} | {samples[-1]:.1f} |")
        points.append((n_tok, med))

    if len(points) >= 2:
        a, b = linear_fit([p[0] for p in points], [p[1] for p in points])
        print(f"\n线性拟合: 截距 **{a:.1f} ms**，边际 **{b:.2f} ms/token**")


if __name__ == "__main__":
    sys.exit(main())
