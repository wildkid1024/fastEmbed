#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_matrix_ops.h"

__global__ void matrixMultiplyKernel(const float* mat1, const float* mat2, float* result, size_t m, size_t n, size_t k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < static_cast<int>(m) && col < static_cast<int>(k)) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += mat1[row * n + i] * mat2[i * k + col];
        }
        result[row * k + col] = sum;
    }
}

__global__ void softmax_kernel(float* input, int num_rows, int num_cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        // Softmax computation logic
        float sum = 0.0f;
        for (int i = 0; i < num_cols; ++i) {
            sum += exp(input[row * num_cols + i]);
        }
        input[row * num_cols + col] = exp(input[row * num_cols + col]) / sum;
    }
}
