#!/usr/bin/env python3
"""嵌入服务回归测试（仅依赖标准库）。

用法:
    python3 scripts/test_embedding.py                 # 默认测 8081 (EmbeddingGemma-2)
    python3 scripts/test_embedding.py --port 8080     # 测 bge-large
    python3 scripts/test_embedding.py --port 8081 --image /path/to/pic.jpg   # 附带视觉塔测试

检查项:
    1. 接口连通与向量维度
    2. 确定性: 相同输入两次调用 cos 应为 1.0
    3. 检索排序: query 与正确文档的相似度应排第一
    4. 跨语言: 中英同义句 cos 应显著高于无关句
    5. 提示词: prompt_name 应改变编码结果
"""
import argparse
import base64
import json
import math
import sys
import time
import urllib.request

DOCS = [
    "巴黎是法国的首都，以埃菲尔铁塔和卢浮宫闻名于世。",
    "苹果公司今天发布了新款智能手机，搭载自研芯片。",
    "深度学习模型的训练需要大量 GPU 算力支持。",
    "今天股市行情波动较大，科技板块普遍下跌。",
]
QUERY = "法国的首都是哪里？"
CROSS_LINGUAL = [
    ("法国的首都是哪里？", "What is the capital of France?"),
    ("如何训练一个神经网络？", "How do I train a neural network?"),
]
NEGATIVE = ("法国的首都是哪里？", "I like to eat pizza with my friends")


def post(port, path, payload, timeout=180):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(port, text, prompt_name=""):
    payload = {"text": text}
    if prompt_name:
        payload["prompt_name"] = prompt_name
    return post(port, "/embed", payload)["embedding"]


def cos(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--prompt-query", default="", help="检索时的 query 提示词，如 query")
    ap.add_argument("--prompt-doc", default="", help="检索时的 document 提示词，如 document")
    ap.add_argument("--image", default="", help="图像路径，附加视觉塔测试")
    args = ap.parse_args()
    port = args.port
    failed = []

    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failed.append(name)

    print(f"目标: http://127.0.0.1:{port}")

    # 1. 维度
    t0 = time.time()
    q = embed(port, QUERY, args.prompt_query)
    vecs = [embed(port, d, args.prompt_doc) for d in DOCS]
    print(f"\n1) 向量维度 = {len(q)}，首次 5 次调用共 {(time.time()-t0)*1000:.0f} ms")
    norm = math.sqrt(sum(x * x for x in q))
    check("归一化", abs(norm - 1) < 1e-3, f"L2 范数 = {norm:.6f}")

    # 2. 确定性
    again = embed(port, QUERY, args.prompt_query)
    check("确定性", abs(cos(q, again) - 1) < 1e-6, f"cos = {cos(q, again):.8f}")

    # 3. 检索排序
    print("\n2) 检索排序:")
    scored = sorted(((cos(q, v), d) for v, d in zip(vecs, DOCS)), reverse=True)
    for i, (s, d) in enumerate(scored, 1):
        print(f"     {i}. cos={s:.4f}  {d[:28]}")
    check("Top-1 命中", scored[0][1] == DOCS[0],
          f"差距 {scored[0][0] - scored[1][0]:.4f}")

    # 4. 跨语言
    print("\n3) 跨语言对齐:")
    pairs = [cos(embed(port, a), embed(port, b)) for a, b in CROSS_LINGUAL]
    neg = cos(embed(port, NEGATIVE[0]), embed(port, NEGATIVE[1]))
    for (a, b), s in zip(CROSS_LINGUAL, pairs):
        print(f"     cos={s:.4f}  「{a}」/「{b}」")
    print(f"     cos={neg:.4f}  无关对照")
    check("跨语言区分", min(pairs) > neg + 0.1,
          f"最低对齐 {min(pairs):.4f} vs 无关 {neg:.4f}")

    # 5. 提示词
    if args.prompt_query:
        plain = embed(port, QUERY, "")
        d = cos(plain, q)
        check("提示词生效", d < 0.999, f"加/不加提示词 cos = {d:.4f}")

    # 6. 视觉塔
    if args.image:
        print("\n4) 视觉塔:")
        with open(args.image, "rb") as f:
            data = f.read()
        payload = {"image_base64": base64.b64encode(data).decode()}
        e1 = post(port, "/image_embedding", payload)["embedding"]
        e2 = post(port, "/image_embedding", payload)["embedding"]
        print(f"     维度 = {len(e1)}")
        check("图像确定性", abs(cos(e1, e2) - 1) < 1e-6, f"cos = {cos(e1, e2):.8f}")
        print(f"     cos(图像, 「{QUERY}」) = {cos(e1, q):.4f}  (跨模态，供参考)")

    # 延迟
    lat = []
    for _ in range(10):
        t = time.time()
        embed(port, "这是一次延迟测试")
        lat.append((time.time() - t) * 1000)
    lat.sort()
    print(f"\n延迟: 平均 {sum(lat)/len(lat):.1f} ms, P50 {lat[len(lat)//2]:.1f} ms, "
          f"最小 {lat[0]:.1f} ms, 最大 {lat[-1]:.1f} ms")

    print(f"\n结果: {'全部通过' if not failed else '失败项 -> ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
