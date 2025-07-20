#ifndef CUDA_MATRIX_OPS_H
#define CUDA_MATRIX_OPS_H

#include "matrix_ops_interface.h"  // 包含头文件
#include <vector>

class CUDAMatrixOps : public MatrixOpsInterface {
public:
    std::vector<float> matrix_multiply(
        const std::vector<float>& mat1,
        const std::vector<float>& mat2,
        size_t m,
        size_t n,
        size_t k
    ) override;
    void softmax(std::vector<float>& input) override;
};

// Check if the code is being compiled by the CUDA compiler
#ifdef __CUDACC__
    // CUDA kernel function declaration
    __global__ void softmax_kernel(float* input, int num_rows, int num_cols);
#endif

#endif // CUDA_MATRIX_OPS_H