#pragma once

// cuBLAS 版 GEMM 封装（行主序语义，与 torch.nn.Linear 的权重布局对齐）
//
// 设计原则：只做「一个算子」，不接管内存管理与流同步——
//   - 输入输出都是**设备指针**，调用方决定何时 H2D / D2H
//   - 全部异步下发到当前设备默认流，与已有的核函数天然保持顺序
//   - 每个线程各自持有一个 cublas handle（cuBLAS 句柄非线程安全）
//
// 与原手写 kernel（~15-20 GFLOPS，B 侧非合并访问）相比，
// cuBLAS 走 tiled GEMM + 寄存器分块，同一张卡上是 1e3 GFLOPS 量级。

#include <cstddef>

namespace cublas_gemm {

// C[m, k] = A[m, n] * B[k, n]^T
// B 为 Linear 权重：[out_features = k, in_features = n]（PyTorch 布局）
// 用于 q/k/v/o_proj、gate/up/down_proj 等所有「x @ W^T」形式的线性层
void mm_nt(const float* d_a, const float* d_b, float* d_c,
           int m, int n, int k);

// C[m, k] = A[m, n] * B[n, k]（B 为普通行主序矩阵）
void mm_nn(const float* d_a, const float* d_b, float* d_c,
           int m, int n, int k);

// 批量版 mm_nt：A 为 [batch, m, n]，B 在所有 batch 间共享（stride 0），
// C 为 [batch, m, k]
void mm_nt_batch(const float* d_a, const float* d_b, float* d_c,
                 int batch, int m, int n, int k);

}  // namespace cublas_gemm
