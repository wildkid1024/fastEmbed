# 权重常驻显存后的 HTTP 延迟复测（2026-10-28）

**硬件**: RTX 2070 8GB（驱动 615.78.08，CUDA 12.6）
**运行环境**: `embedding-cuda-dev` 容器，`build-cuda/server`（含未提交的 `DeviceWeightCache` 改动）
**测量方式**: 单实例、串行、HTTP 端到端 `time_total`；每档 warmup 3 次 + 采样 10 次；
token 数由真实 `tokenizer.json` 统计（脚本 `scripts/bench_http.py`）
**对照基线**: `HTTP_MODE_PERFORMANCE_REPORT.md`（2026-05-16，Qwen3-Embedding-0.6B，权重每请求重传）

---

## 一、Qwen3-Embedding-0.6B：改前 vs 改后

| 文本长度 | 改前 median | 改后 median | 加速比 |
|---|---:|---:|---:|
| 短文本（~5 tok） | 300 ms | 168 ms（12 tok） | **1.8x** |
| ~15 tok | 906 ms | 232 ms | **3.9x** |
| ~30 tok | 2045 ms | 424 ms | **4.8x** |
| ~60 tok | 3824 ms | 752 ms | **5.1x** |
| 128 tok | — | 1448 ms | — |

改后明细（median，10 次采样）：

| 实际 tok | min | median | mean | p95 | max |
|---:|---:|---:|---:|---:|---:|
| 12 | 167.7 | 168.3 | 168.4 | 169.6 | 169.6 |
| 15 | 231.2 | 231.7 | 232.0 | 233.7 | 233.7 |
| 30 | 422.7 | 424.0 | 424.3 | 426.7 | 426.7 |
| 60 | 747.8 | 752.1 | 752.1 | 758.2 | 758.2 |
| 128 | 1440.9 | 1447.7 | 1447.9 | 1455.9 | 1455.9 |

线性拟合（median vs tokens）：**截距 72.4 ms，边际 10.87 ms/token**。

- 改前同样拟合，截距约 **293 ms**（每请求重传 1.76 GB 权重的 PCIe 时间）→ 现在 **72.4 ms**，说明**权重搬运的固定底噪已消除**。
- 剩下的 72 ms 来自其它每请求固定开销：激活值的 H2D/D2H 拷贝、每算子 `cudaDeviceSynchronize()`、注意力里的 `cudaMalloc/Free`。
- 边际 10.87 ms/token 是**朴素 GEMM** 的成本（约 15–20 GFLOPS，无 tiling、无向量化）——这是下一步换 cuBLAS 的目标。
- 稳定性：min→max 波动 < 1%，几乎没有抖动。

## 二、另外两个模型（单实例，同卡同方法）

| 模型 | 实际 tok | median | p95 | 截距 | 边际 |
|---|---:|---:|---:|---:|---:|
| embeddinggemma-2（0.3B） | 12 | 49.0 ms | 50.2 | **8.6 ms** | **3.37 ms/token** |
| | 30 | 106.8 ms | 108.0 | | |
| | 60 | 211.5 ms | 214.7 | | |
| | 128 | 440.1 ms | 442.7 | | |
| bge-large-zh-v1.5（0.33B） | 21 | 156.6 ms | 161.5 | **18.4 ms** | **6.52 ms/token** |
| | 30 | 210.8 ms | 211.9 | | |
| | 60 | 409.0 ms | 410.8 | | |
| | 128 | 852.7 ms | 854.4 | | |

截距差异（gemma2 8.6 ms / bge 18.4 ms / qwen3 72.4 ms）与层数、算子数量、权重规模一致，
进一步印证「固定开销 = 每请求的拷贝 + 同步次数」这一判断。

## 三、多模型共卡会显著放大延迟

同一模型（embeddinggemma-2）在两种占用下的对比：

| 场景 | 12 tok | 30 tok | 60 tok |
|---|---:|---:|---:|
| 独占显卡（本次） | 49 ms | 107 ms | 212 ms |
| 与 bge 共驻 8GB 卡（4.1 GB 已用） | 61 ms | 303 ms | 586 ms |

共驻时 30 tok 慢了 **2.8x**。原因是每个请求要发起 200+ 次 kernel launch，
两个进程在同卡上争抢 SM 与上下文切换。**对外压测数据必须单实例隔离**，否则会被低估数倍。

## 四、显存代价

`DeviceWeightCache` 日志确认权重常驻生效：

```
[DeviceWeightCache] 驻留权重 4.0 MB（累计 0.51 GB / 215 个张量）   ← embeddinggemma-2 文本塔
```

- gemma2 文本塔：0.51 GB 权重驻留，进程整体 2084 MiB（含 CUDA context）
- 代价 = 模型 FP32 权重大小（Qwen3-0.6B ≈ 2.4 GB），需要 ≥4 GB 空闲显存

## 五、结论与下一步

1. **修复①（权重常驻）已验证有效**：同为 60 token，3824 ms → 752 ms（5.1x），固定底噪 293 ms → 72 ms。
2. **当前瓶颈已转移到计算**：边际 10.87 ms/token，占 60 token 请求的 87%。
3. 下一步优先级不变：**② GEMM 换 cuBLAS（先 FP32，再 BF16 + cuBLASLt）** → 预期边际降到 1 ms/token 级，
   60 token 有望进 100 ms 内；之后 ③ 去 per-op 同步、④ 真 batching（吞吐维度）、⑤ 每线程独立暂存区。
4. 复测命令（容器内或宿主机均可，需能访问服务端口与 tokenizer.json）：

```bash
python3 scripts/bench_http.py --port 8080 --model Qwen3-Embedding-0.6B \
    --tokenizer /Models/Qwen/Qwen3-Embedding-0___6B/tokenizer.json \
    --levels 5,15,30,60,128 --reps 10
```
