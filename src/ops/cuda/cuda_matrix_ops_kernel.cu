#include <cuda_runtime.h>

__global__ void matrixMultiplyKernel(const float* mat1, const float* mat2, float* result, size_t m, size_t n, size_t k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (size_t p = 0; p < n; ++p) {
            sum += mat1[row * n + p] * mat2[p * k + col];
        }
        result[row * k + col] = sum;
    }
}
