# 线性层换 cuBLAS：实测记录（2026-10-28）

**硬件**：RTX 2070 8GB · 驱动 615.78.08 · CUDA 12.6
**精度**：FP32（`CUBLAS_DEFAULT_MATH`，未启用 TF32 / Tensor Core）
**方法**：HTTP 端到端 `POST /v1/embeddings`，单实例串行，每档预热 3 次 + 采样 10 次取中位数；
token 数由模型自带 tokenizer 统计。脚本 `scripts/bench_http.py`。

---

## 一、改了什么

新增 `include/ops/cuda/cublas_gemm.h` + `src/ops/cuda/cuda_cublas_gemm.cu`，提供三个纯算子：

| 算子 | 语义 |
|---|---|
| `mm_nt` | `C[m,k] = A[m,n] · W[k,n]^T`（Linear 权重布局，直接用，无需转置拷贝） |
| `mm_nn` | `C[m,k] = A[m,n] · B[n,k]` |
| `mm_nt_batch` | `C[b,m,k] = A[b,m,n] · W[k,n]^T`，B 的 batch stride = 0（权重广播一份） |

设计上只做「一次 GEMM」：入参出参都是**设备指针**，异步下发到默认流，每线程独占一个
`cublasHandle`（cuBLAS 句柄非线程安全）。不接管内存分配，也不改变调用方的 H2D / D2H 时机。

调用点改动很小，都是把原来的 kernel launch 换成一行调用：

- `matrix_multiply` → `mm_nn`
- `matrix_multiply_transpose`（q/k/v/o_proj、gate/up/down_proj 全走这里）→ `mm_nt`
- `mlp_fused` / `mlp_fused_gelu` 内部的 3 次 GEMM → `mm_nt`
- `matrix_multiply_transpose_batch` → `mm_nt_batch`

原有手写 kernel 保留在文件里作为参考实现，不再被调用。`CMakeLists.txt` 里两条构建分支
（`USE_CUBLAS` 开/关）都加入该源文件并链接 `CUDA::cublas`——`USE_CUBLAS` 现在只决定
**注意力**用哪套实现，与线性层无关。

## 二、正确性验证

### 2.1 GEMM 与 CPU 双精度参考对拍

| 场景 | 形状 | max_abs_diff |
|---|---|---:|
| mm_nt | 1 × 1 × 1 | 5.4e-09 |
| mm_nt | m=1, n=512, k=1024 | 3.5e-06 |
| mm_nt | m=7, n=13, k=5 | 2.3e-07 |
| mm_nt | m=128, n=768, k=3072 | 1.9e-05 |
| mm_nn | m=3, n=4, k=5 | 6.7e-08 |
| mm_nn | m=60, n=768, k=768 | 8.3e-06 |
| mm_nt_batch | batch=3, m=9, n=16, k=6 | 4.4e-07 |

全部在 fp32 舍入范围内（相对误差 ~1e-7），说明行主序 ↔ 列主序的转置关系和 leading dimension 都没写错。

### 2.2 端到端与 PyTorch 对齐

bge-large-zh-v1.5，输入「人工智能正在改变世界」，与 `transformers` 4.57
（torch 2.8.0+cu128，FP32，CLS pooling + L2 归一化）对比：

```
dim=1024  cos=0.999998927  max_abs_diff=2.042e-04
```

差异来自 fp32 累加顺序不同（cuBLAS 分块累加 vs 原 kernel 顺序累加），经过 24 层后放大到 2e-4 量级；
余弦相似度不影响任何检索/聚类结果。

### 2.3 batch 与单条一致

同一批 3 条文本，分别单条请求与批量请求（`input` 传数组）：

```
text0: cos=1.000000000  max_abs_diff=0.000e+00
text1: cos=1.000000000  max_abs_diff=0.000e+00
text2: cos=1.000000000  max_abs_diff=0.000e+00
```

即 `mm_nt_batch` 的权重广播（stride 0）路径与逐条 `mm_nt` 完全等价。

## 三、性能结果

### 3.1 端到端延迟（median, ms）

| 模型 | 实现 | 12–21 tok | 30 tok | 60 tok | 128 tok |
|---|---|---:|---:|---:|---:|
| embeddinggemma-2（0.3B） | 权重常驻（改动前） | 49 | 107 | 212 | 440 |
| | **+ cuBLAS** | **24** | **36** | **61** | **124** |
| | 加速 | 2.0x | 3.0x | 3.5x | 3.5x |
| bge-large-zh-v1.5（0.33B） | 权重常驻（改动前） | 157 | 211 | 409 | 853 |
| | **+ cuBLAS** | **70** | **91** | **184** | **364** |
| | 加速 | 2.2x | 2.3x | 2.2x | 2.3x |
| Qwen3-Embedding-0.6B（0.6B） | 权重常驻（改动前） | 168 | 424 | 752 | 1448 |
| | **+ cuBLAS** | **78** | **190** | **270** | **516** |
| | 加速 | 2.2x | 2.2x | 2.8x | 2.8x |

### 3.2 固定开销与边际成本

| 模型 | 固定开销 前 → 后 | 每 token 边际 前 → 后 |
|---|---|---|
| embeddinggemma-2 | 8.6 → **11.0 ms** | 3.37 → **0.87 ms** |
| bge-large-zh-v1.5 | 18.4 → **12.0 ms** | 6.52 → **2.77 ms** |
| Qwen3-Embedding-0.6B | 72.4 → **55.3 ms** | 10.87 → **3.62 ms** |

边际成本（纯计算）降了 3–4 倍，这正是 GEMM 从「~15–20 GFLOPS 的朴素实现」换成 cuBLAS 的直接收益。
gemma2 的固定开销略升 2.4 ms（8.6 → 11.0），量级上可忽略，属于首次创建 cublas handle 与测量抖动范围。

### 3.3 共卡干扰消失

改动前，两个实例共驻 8GB 卡会让 embeddinggemma-2 在 30 token 上从 107 ms 涨到 303 ms（2.8x）。
cuBLAS 之后重测：

| 场景 | 30 tok | 60 tok |
|---|---:|---:|
| 独占整卡 | 35.6 ms | 60.6 ms |
| 与另一实例共驻（另一进程空闲） | 35.8 ms | 60.3 ms |

差异在噪声内。原因是原来那 200+ 次串行 kernel launch 与同步把调度器压满了，现在单请求的总工作量
小了一个量级，两个进程不再互相排队。

## 四、剩余瓶颈：每层的主机↔设备往返

边际成本换算成算力利用率：gemma2 ≈230 GFLOPS、bge ≈290 GFLOPS、qwen ≈210 GFLOPS，
相对 2070 的 FP32 峰值 7.9 TFLOPS 只有 3%。也就是说 **GEMM 本身已经不是短板**，时间花在别处：

- 每个算子接口都是 `std::vector` 进、`std::vector` 出，因此每层要做约 10 次
  `cudaMemcpy(H2D) → kernel/gemm → cudaDeviceSynchronize → cudaMemcpy(D2H)`；
- 单请求累计两百余次同步、上百 MB 的 pageable 往返流量（128 token 时每层激活 τ≈0.5 MB，
  一次往返的有效带宽只有 2–3 GB/s）。

对照 PyTorch（同卡、同模型、fp32、batch=1）：bge-large 128 token 为 17.2 ms，当前实现 364 ms。
差距仍在一个数量级，但性质已经变了——**不再是「算得慢」，而是「来回搬」**。

下一步的收益点按顺序：

1. **前向过程芯片内闭环**：激活常驻显存，层内算子串在同一 stream 上，只在最后做一次 D2H；
   预期把固定开销压到 5 ms 以内、边际成本再降一半以上。
2. **注意力融合**：`cuda_attention_ops.cu` 里每个头 `cudaMalloc` + 三段 kernel（QK^T / softmax / PV），
   28 层 × 16 头 = 448 次分配/请求；可换成 `cudaMallocAsync` 池或 batched GEMM + flash 风格融合。
3. **BF16 + Tensor Core**：在 batching 场景下才有意义，单条请求的瓶颈还不在算力。

## 五、复现

```bash
# 单实例
PORT=8080 MODEL_PATH=/Models/models-hf/sentence_models/bge-large-zh-v1.5 \
  MODEL_NAME=bge-large-zh-v1.5 ./scripts/dev.sh run

# 压测（token 数由 tokenizer.json 精确统计）
python3 scripts/bench_http.py --port 8080 --model bge-large-zh-v1.5 \
  --tokenizer /path/to/bge-large-zh-v1.5/tokenizer.json --levels 5,15,30,60,128 --reps 10
```

## 六、第二轮：注意力内部的 Q/K/V 投影换 cuBLAS

第一轮只替换了 `cuda_matrix_ops` 里的线性层，**注意力内部的 Q/K/V 投影仍是朴素 kernel**
（`cuda_attention_ops.cu` 三处 `matrix_multiply_transpose_kernel<<<>>>`）。这一轮把它们换成
`cublas_gemm::mm_nt`，权重直接复用 `DeviceWeightCache` 的常驻指针，共 3 行 + 1 处 include。

### 6.1 正确性

同一句输入（「人工智能正在改变世界」，bge-large-zh-v1.5）：`cos = 1.000000000`，
`max_abs_diff = 8.1e-08`。差异来自朴素 kernel 与 cuBLAS SGEMM 的 fp32 累加顺序，属于噪声级。

### 6.2 延迟

| 模型 | 30 tok | 60 tok | 128 tok | 固定开销 | 每 token 边际 |
|---|---:|---:|---:|---:|---:|
| bge-large-zh-v1.5（唯一走 multi_head_attention 的模型） | 91 → **54** | 184 → **103** | 364 → **220** | 12.0 → **5.1 ms** | 2.77 → **1.67 ms** |
| embeddinggemma-2（走 encoder_attention） | 35.1 | 60.5 | 123.6 | 10.3 ms | 0.88 ms |
| Qwen3-Embedding-0.6B（走 grouped_query_attention） | 182 | 281 | 494 | 58.1 ms | 3.47 ms |

只有 bge 变化，另两个在噪声内——它们的注意力本来就自己算投影，不走 `multi_head_attention`。
bge 128 tok 累计轨迹：853 ms（初始）→ 364 ms（线性层 cuBLAS）→ **220 ms（QKV cuBLAS）**。

### 6.3 为什么 220 ms 而不是预估的 30–40 ms

原先的估计假设 bge 剩余耗时主要来自 QKV 的 19.3 GFLOP 朴素 GEMM（按 60 GFLOPS 折算约 350 ms）。
实测只省下 144 ms，反推朴素 kernel 实际跑到约 134 GFLOPS，比估计快。

更关键的证据在**曲线的形状**：30 token 时注意力的二次项（QK^T / PV）几乎为零，
但每 token 边际成本与 128 token 时一致（1.67 vs 1.67）。如果是 attention score/PV 的朴素 matmul
在拖，边际成本应当随 token 增长而上升，而不是常数。**所以剩余成本不是另一处朴素 GEMM，
而是与 op 数量 × 激活体积成正比的「每算子 H2D → kernel → sync → D2H」**，
即第四节所说的架构问题，继续换 GEMM 换不来收益。

## 七、下一步的唯一收益点

要跨过 1.67 ms/token，必须让激活值常驻显存、整条前向在单 stream 上排队、只在末尾 D2H 一次。
这是一次结构性重构（`sentence_transformers_impl.cpp` 的前向要从 `std::vector` 进 / `std::vector` 出
改成设备指针进 / 设备指针出），无法靠新增算子完成。收益量级：按 op 数 480、每次往返 3.5 µs/token 估算，
可把边际压到 0.1 ms/token 量级，128 token 有望从 220 ms 进到 30 ms 以内。
