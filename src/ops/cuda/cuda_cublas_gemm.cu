#include "ops/cuda/cublas_gemm.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// ============================================================================
// cuBLAS 版 GEMM
//
// 行主序 ↔ 列主序的转换方式（cuBLAS 只认列主序）：
//   行主序 C[m,k] = A[m,n] * B[k,n]^T
//   等价于列主序 C^T[k,m] = B_c^T * A_c，其中
//     B_c = B 按列主序解读（即 [n,k]），A_c = A 按列主序解读（即 [n,m]）
//   于是 cublasSgemm(OP_T, OP_N, m=k, n=m, k=n, lda=n, ldb=n, ldc=k)
//   权重以原布局直接传入，不需要任何转置拷贝。
// ============================================================================

namespace {

void check_cublas(cublasStatus_t st, const char* what) {
    if (st == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    const char* name = "UNKNOWN";
    switch (st) {
        case CUBLAS_STATUS_NOT_INITIALIZED: name = "NOT_INITIALIZED"; break;
        case CUBLAS_STATUS_ALLOC_FAILED:    name = "ALLOC_FAILED";    break;
        case CUBLAS_STATUS_INVALID_VALUE:   name = "INVALID_VALUE";   break;
        case CUBLAS_STATUS_ARCH_MISMATCH:   name = "ARCH_MISMATCH";   break;
        case CUBLAS_STATUS_EXECUTION_FAILED: name = "EXECUTION_FAILED"; break;
        default: break;
    }
    std::fprintf(stderr, "[cublas_gemm] %s failed: %s (%d)\n", what, name, static_cast<int>(st));
    std::abort();
}

// 每线程一个 handle：cublas handle 内部持有 stream 与 workspace，不能跨线程共享
struct ThreadHandle {
    cublasHandle_t handle = nullptr;

    ThreadHandle() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "[cublas_gemm] cublasCreate failed (no CUDA context?)\n");
            handle = nullptr;
            return;
        }
        // 保持 IEEE fp32（Turing/TU102 没有 TF32 通路，显式固定默认数学模式，
        // 这样结果可与 PyTorch fp32 逐位对比）
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }

    ~ThreadHandle() {
        if (handle != nullptr) {
            // 进程退出阶段 CUDA context 可能已销毁，此处错误无需上报
            cublasDestroy(handle);
            handle = nullptr;
        }
    }
};

inline cublasHandle_t get_handle() {
    thread_local ThreadHandle thread_handle;
    if (thread_handle.handle == nullptr) {
        std::fprintf(stderr, "[cublas_gemm] cuBLAS handle unavailable\n");
        std::abort();
    }
    return thread_handle.handle;
}

}  // namespace

namespace cublas_gemm {

void mm_nt(const float* d_a, const float* d_b, float* d_c,
           int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // 列主序视点：C^T[k, m] = B^T * A
    check_cublas(cublasSgemm(get_handle(),
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             k, m, n,
                             &alpha,
                             d_b, n,   // B[k,n] 行主序 → 列主序 [n,k]，ld = n
                             d_a, n,   // A[m,n] 行主序 → 列主序 [n,m]，ld = n
                             &beta,
                             d_c, k),  // C[m,k] 行主序 → 列主序 [k,m]，ld = k
                "Sgemm(NT)");
}

void mm_nn(const float* d_a, const float* d_b, float* d_c,
           int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // 列主序视点：C^T[k, m] = B^T[k, n] * A^T[n, m]
    check_cublas(cublasSgemm(get_handle(),
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             k, m, n,
                             &alpha,
                             d_b, k,   // B[n,k] 行主序 → 列主序 [k,n]，ld = k
                             d_a, n,
                             &beta,
                             d_c, k),
                "Sgemm(NN)");
}

void mm_nt_batch(const float* d_a, const float* d_b, float* d_c,
                 int batch, int m, int n, int k) {
    if (batch <= 0 || m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasHandle_t handle = get_handle();

    // B 的批次 stride 取 0：权重常驻一份，由所有 batch 共享
    // （此前的实现为每个 batch 元素各拷一份，显存与 PCIe 流量都按 batch 放大）
    cublasStatus_t st = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        k, m, n,
        &alpha,
        d_b, n, 0,
        d_a, n, static_cast<long long>(m) * n,
        &beta,
        d_c, k, static_cast<long long>(m) * k,
        batch);

    if (st != CUBLAS_STATUS_SUCCESS) {
        // 个别老版本驱动不接受 strideB = 0，退化为逐 batch 调用
        for (int b = 0; b < batch; ++b) {
            check_cublas(cublasSgemm(handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     k, m, n,
                                     &alpha,
                                     d_b, n,
                                     d_a + static_cast<size_t>(b) * m * n, n,
                                     &beta,
                                     d_c + static_cast<size_t>(b) * m * k, k),
                         "Sgemm(batch fallback)");
        }
    }
}

}  // namespace cublas_gemm
