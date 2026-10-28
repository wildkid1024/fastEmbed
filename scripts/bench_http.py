#!/usr/bin/env python3
"""HTTP 端到端延迟压测（串行、单实例）。

按目标 token 数构造输入，逐条 POST /v1/embeddings，统计 min / median / mean / p95。
用真实 tokenizer 统计 token 数，避免"~5 token"这类估算误差。

用法:
    python3 scripts/bench_http.py --port 8080 --model Qwen3-Embedding-0.6B \
        --tokenizer /Models/Qwen/Qwen3-Embedding-0___6B/tokenizer.json \
        --levels 5,15,30,60,128 --reps 10

注意: 必须在容器/宿主内执行（要能访问 127.0.0.1:<port> 和 tokenizer.json）。
"""
import argparse
import json
import statistics
import time
import urllib.request

from tokenizers import Tokenizer

BASE = "Artificial intelligence is transforming the way we build software."


def build_text(tok: Tokenizer, target: int) -> str:
    """在 BASE 之后追加填充词，直到 token 数 >= target（尽量贴近）。"""
    text = BASE
    while len(tok.encode(text).ids) < target:
        text += " word"
    return text


def post_embedding(url: str, model: str, text: str, timeout: float = 120.0) -> float:
    """发一条请求，返回耗时（秒）。"""
    payload = json.dumps({"input": text, "model": model}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    elapsed = time.perf_counter() - start
    parsed = json.loads(body)
    if "data" not in parsed or not parsed["data"]:
        raise RuntimeError(f"响应异常: {body[:200]!r}")
    return elapsed


def linear_fit(xs, ys):
    """最小二乘拟合 y = a + b*x，返回 (a, b)。"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return my - slope * mx, slope


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True, help="请求里 model 字段（服务端 --serve_model_name）")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json 路径")
    ap.add_argument("--levels", default="5,15,30,60,128", help="目标 token 数，逗号分隔")
    ap.add_argument("--reps", type=int, default=10, help="每个长度的有效采样次数")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    url = f"http://{args.host}:{args.port}/v1/embeddings"

    print(f"# 目标 {url}  ·  model={args.model}  ·  reps={args.reps}  (warmup={args.warmup})\n")
    print("| 目标 tok | 实际 tok | 字符数 | min(ms) | median(ms) | mean(ms) | p95(ms) | max(ms) |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")

    points = []
    for target in [int(x) for x in args.levels.split(",")]:
        text = build_text(tok, target)
        n_tok = len(tok.encode(text).ids)

        for _ in range(args.warmup):
            post_embedding(url, args.model, text)

        samples = [post_embedding(url, args.model, text) * 1000 for _ in range(args.reps)]
        samples_sorted = sorted(samples)
        p95 = samples_sorted[min(len(samples_sorted) - 1, int(len(samples_sorted) * 0.95))]
        med = statistics.median(samples)
        print(
            f"| {target} | {n_tok} | {len(text)} | {min(samples):.1f} | {med:.1f} | "
            f"{statistics.mean(samples):.1f} | {p95:.1f} | {max(samples):.1f} |"
        )
        points.append((n_tok, med))

    if len(points) >= 2:
        intercept, slope = linear_fit([p[0] for p in points], [p[1] for p in points])
        print(
            f"\n线性拟合（median vs tokens）: 截距 **{intercept:.1f} ms**，"
            f"边际 **{slope:.2f} ms/token**"
        )
        print("截距≈与长度无关的固定开销（权重上传、进程/显存初始化等），边际≈纯计算成本。")


if __name__ == "__main__":
    main()
