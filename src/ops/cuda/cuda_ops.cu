#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// 矩阵乘法核函数
__global__ void matrix_multiply_kernel(const float* mat1, const float* mat2, float* result, size_t m, size_t n, size_t k) {
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

// Softmax 核函数
__global__ void softmax_kernel(float* input, size_t n) {
    __shared__ float max_val;
    __shared__ float sum;

    // 计算最大值
    if (threadIdx.x == 0) {
        max_val = input[0];
        for (size_t i = 1; i < n; ++i) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }
    }
    __syncthreads();

    // 计算指数和
    if (threadIdx.x == 0) {
        sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += expf(input[i] - max_val);
        }
    }
    __syncthreads();

    // 计算 Softmax
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        input[i] = expf(input[i] - max_val) / sum;
    }
}

// GELU 激活函数核函数
__global__ void gelu_kernel(float* input, size_t n) {
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        float x = input[i];
        input[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }
}

// GELU 激活函数
std::vector<float> gelu(const std::vector<float>& input) {
    std::vector<float> output = input;
    float *d_output;
    cudaMalloc((void**)&d_output, output.size() * sizeof(float));
    cudaMemcpy(d_output, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice);

    gelu_kernel<<<1, 256>>>(d_output, output.size());

    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    return output;
}
