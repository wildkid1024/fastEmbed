#include "ops/cuda/cuda_matrix_ops.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ops/cuda/cuda_matrix_ops.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>  // Add CUDA math constants header

// CUDA错误检查宏
#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// 矩阵乘法核函数 (C=A*B)
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C,
                                       int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// 转置矩阵乘法核函数 (C=A*B^T)
__global__ void matrix_multiply_transpose_kernel(const float* A, const float* B, float* C,
                                                int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[col * n + i];
        }
        C[row * k + col] = sum;
    }
}

// ReLU激活函数核
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(0.0f, input[idx]);
    }
}

// Swish激活函数核
__global__ void swish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / (1.0f + expf(-input[idx]));
    }
}

// GELU激活函数核
__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 内联GELU计算，移除嵌套的__device__函数定义
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / CUDART_PI_F) * (x + 0.044715f * powf(x, 3.0f))));
    }
}

// 层归一化核函数
// 修改核函数参数列表
__global__ void layer_norm_kernel(
    const float* input, 
    const float* gamma, 
    const float* beta, 
    float* output, 
    size_t seq_len, 
    size_t embedding_dim, 
    float epsilon) {
    // 每个线程处理一个特征维度
    int seq_idx = blockIdx.x;  // block索引直接对应序列索引
    int feature_idx = threadIdx.x;

    if (feature_idx < embedding_dim) {
        // 计算全局索引（假设无batch维度）
        int global_idx = seq_idx * embedding_dim + feature_idx;

        // 共享内存存储当前序列所有特征值
        extern __shared__ float s_features[];
        s_features[feature_idx] = input[global_idx];
        __syncthreads();

        // 计算均值 (与CPU实现相同的累加方式)
        float mean = 0.0f;
        for (int i = 0; i < embedding_dim; ++i) {
            mean += s_features[i];
        }
        mean /= embedding_dim;

        // 计算方差 (与CPU实现相同的二次累加)
        float variance = 0.0f;
        for (int i = 0; i < embedding_dim; ++i) {
            float diff = s_features[i] - mean;
            variance += diff * diff;
        }
        variance /= embedding_dim;
        float std_dev = sqrtf(variance + epsilon);

        // 应用LayerNorm (与CPU公式完全一致)
        output[global_idx] = (s_features[feature_idx] - mean) / std_dev * gamma[feature_idx] + beta[feature_idx];
    }
}


// Softmax核函数
__global__ void softmax_kernel(const float* input, float* output, int size, int dim_size, int axis) {
    if (axis == 1) {
        int row = blockIdx.x;
        int col = threadIdx.x;
        int idx = row * dim_size + col;

        if (row >= size / dim_size || col >= dim_size) return;

        // 找到每行的最大值
        __shared__ float row_max[256];
        float max_val = input[idx];
        for (int i = col; i < dim_size; i += blockDim.x) {
            max_val = max(max_val, input[row * dim_size + i]);
        }
        row_max[col] = max_val;

        // 归约找到最大值
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            __syncthreads();
            if (col < s) {
                row_max[col] = max(row_max[col], row_max[col + s]);
            }
        }

        __syncthreads();
        max_val = row_max[0];

        // 计算指数并求和
        float exp_val = expf(input[idx] - max_val);
        __shared__ float row_sum[256];
        row_sum[col] = exp_val;

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            __syncthreads();
            if (col < s) {
                row_sum[col] += row_sum[col + s];
            }
        }

        __syncthreads();
        output[idx] = exp_val / row_sum[0];
    }
}

void CUDAMatrixOps::ensure_memory(size_t required_size) {
    if (required_size > current_alloc_size) {
        // 释放旧内存
        if (d_a != nullptr) {
            CHECK_CUDA_ERROR(cudaFree(d_a));
            CHECK_CUDA_ERROR(cudaFree(d_b));
            CHECK_CUDA_ERROR(cudaFree(d_result));
        }

        // 分配新内存
        CHECK_CUDA_ERROR(cudaMalloc(&d_a, required_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, required_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_result, required_size * sizeof(float)));
        current_alloc_size = required_size;
    }
}

CUDAMatrixOps::~CUDAMatrixOps() {
    if (d_a != nullptr) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
    }
}

std::vector<float> CUDAMatrixOps::matrix_multiply(const std::vector<float>& a, const std::vector<float>& b,
                                                  size_t m, size_t n, size_t k) {
    size_t a_size = m * n;
    size_t b_size = n * k;
    size_t result_size = m * k;
    size_t required_size = std::max({a_size, b_size, result_size});

    ensure_memory(required_size);

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, a.data(), a_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, b.data(), b_size * sizeof(float), cudaMemcpyHostToDevice));

    // 配置核函数网格和块
    dim3 blockDim(16, 16);
    dim3 gridDim((k + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    // 启动核函数
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_result, m, n, k);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 复制结果回主机
    std::vector<float> result(result_size);
    CHECK_CUDA_ERROR(cudaMemcpy(result.data(), d_result, result_size * sizeof(float), cudaMemcpyDeviceToHost));

    return result;
}

std::vector<float> CUDAMatrixOps::matrix_multiply_transpose(const std::vector<float>& a, const std::vector<float>& b,
                                                           size_t m, size_t n, size_t k) {
    size_t a_size = m * n;
    size_t b_size = n * k;
    size_t result_size = m * k;
    size_t required_size = std::max({a_size, b_size, result_size});

    ensure_memory(required_size);

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, a.data(), a_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, b.data(), b_size * sizeof(float), cudaMemcpyHostToDevice));

    // 配置核函数网格和块
    dim3 blockDim(16, 16);
    dim3 gridDim((k + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    // 启动核函数
    matrix_multiply_transpose_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_result, m, n, k);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 复制结果回主机
    std::vector<float> result(result_size);
    CHECK_CUDA_ERROR(cudaMemcpy(result.data(), d_result, result_size * sizeof(float), cudaMemcpyDeviceToHost));

    return result;
}

std::vector<float> CUDAMatrixOps::relu(const std::vector<float>& input) {
    size_t size = input.size();
    ensure_memory(size);

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, input.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);

    relu_kernel<<<gridDim, blockDim>>>(d_a, d_result, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> output(size);
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost));

    return output;
}

std::vector<float> CUDAMatrixOps::swish(const std::vector<float>& input) {
    size_t size = input.size();
    ensure_memory(size);

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, input.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);

    swish_kernel<<<gridDim, blockDim>>>(d_a, d_result, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> output(size);
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost));

    return output;
}

std::vector<float> CUDAMatrixOps::gelu(const std::vector<float>& input) {
    size_t size = input.size();
    ensure_memory(size);

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, input.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);

    gelu_kernel<<<gridDim, blockDim>>>(d_a, d_result, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> output(size);
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost));

    return output;
}

std::vector<float> CUDAMatrixOps::layer_norm(
    const std::vector<float>& input, 
    const std::vector<float>& gamma, 
    const std::vector<float>& beta, 
    size_t embedding_dim, 
    float epsilon) {
    // 添加输入验证（与CPU版本保持一致）
    if (input.empty() || gamma.empty() || beta.empty()) {
        throw std::invalid_argument("Input vectors cannot be empty");
    }
    if (gamma.size() != embedding_dim || beta.size() != embedding_dim) {
        throw std::invalid_argument("Gamma and beta must match embedding dimension size");
    }

    float* d_input = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_output = nullptr;

    // 计算序列长度（与CPU版本逻辑一致）
    size_t seq_len = input.size() / embedding_dim;
    std::vector<float> output(input.size());

    // 向量转指针（添加这部分代码）
    const float* input_ptr = input.data();
    const float* gamma_ptr = gamma.data();
    const float* beta_ptr = beta.data();
    float* output_ptr = output.data();

    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_gamma, gamma.size() * sizeof(float));
    cudaMalloc(&d_beta, beta.size() * sizeof(float));
    cudaMalloc(&d_output, output.size() * sizeof(float));

    cudaMemcpy(d_input, input_ptr, input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma_ptr, gamma.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta_ptr, beta.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(1024);  // 每个block处理一个序列的所有特征， TODO: 当seq_len大于1024时，需要修改blockDim
    dim3 gridDim(seq_len);  // 每个序列分配一个block

    // 修改核函数调用参数
    layer_norm_kernel<<<gridDim, blockDim, embedding_dim * sizeof(float)>>>(
        d_input, d_gamma, d_beta, d_output, 
        seq_len, embedding_dim, epsilon
    );

    cudaMemcpy(output_ptr, d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);

    return output;  // 返回结果向量
}

std::vector<float> CUDAMatrixOps::softmax(const std::vector<float>& input, size_t axis, size_t dim_size) {
    if (axis != 1) {
        throw std::invalid_argument("Only axis=1 is supported for CUDA softmax");
    }

    size_t size = input.size();
    if (dim_size == 0) dim_size = 768; // 默认维度
    if (size % dim_size != 0) {
        throw std::invalid_argument("Input size must be divisible by dim_size");
    }

    ensure_memory(size);
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, input.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(dim_size);
    dim3 gridDim(size / dim_size);

    softmax_kernel<<<gridDim, blockDim>>>(d_a, d_result, size, dim_size, axis);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> output(size);
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost));

    return output;
}

__global__ void softmax_kernel_inplace(float* input, size_t seq_len) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int head = blockIdx.z;
    int idx = head * seq_len * seq_len + row * seq_len;

    if (row < seq_len) {
        // Find max value for numerical stability
        float max_val = input[idx];
        for (int col = 1; col < seq_len; col++) {
            if (input[idx + col] > max_val) max_val = input[idx + col];
        }

        // Compute softmax
        float sum = 0.0f;
        for (int col = 0; col < seq_len; col++) {
            input[idx + col] = expf(input[idx + col] - max_val);
            sum += input[idx + col];
        }

        // Normalize
        for (int col = 0; col < seq_len; col++) {
            input[idx + col] /= sum;
        }
    }
}

void CUDAMatrixOps::apply_softmax(float* input, size_t num_heads, size_t seq_len) {
    dim3 block(16, 16);
    dim3 grid(1, (seq_len + block.y - 1) / block.y, num_heads);
    softmax_kernel_inplace<<<grid, block>>>(input, seq_len);
    cudaDeviceSynchronize();
}

// CUDA RMSNorm核函数
__global__ void rms_norm_kernel(
    const float* input, 
    const float* weight, 
    float* output, 
    size_t seq_len, 
    size_t embedding_dim, 
    float epsilon) {
    int seq_idx = blockIdx.x;  // block索引对应序列索引
    int feature_idx = threadIdx.x;  // thread索引对应特征维度索引

    if (seq_idx < seq_len && feature_idx < embedding_dim) {
        // 计算当前token的RMSNorm归一化因子
        extern __shared__ float s_square_sums[];  // 共享内存存储平方和
        float val = input[seq_idx * embedding_dim + feature_idx];
        float square_val = val * val;
        
        // 使用共享内存进行归约计算RMS
        s_square_sums[feature_idx] = square_val;
        __syncthreads();

        // 简化实现：每个线程块的第一个线程计算整个token的RMS
        if (feature_idx == 0) {
            float sum = 0.0f;
            for (int i = 0; i < embedding_dim; ++i) {
                sum += s_square_sums[i];
            }
            sum /= embedding_dim;
            sum += epsilon;
            s_square_sums[0] = 1.0f / sqrtf(sum);  // 存储归一化因子
        }
        __syncthreads();

        // 应用RMSNorm: (input * weight) / RMS
        float norm_factor = s_square_sums[0];
        output[seq_idx * embedding_dim + feature_idx] = 
            norm_factor * weight[feature_idx] * val;
    }
}

std::vector<float> CUDAMatrixOps::rms_norm(
    const std::vector<float>& input, 
    const std::vector<float>& weight, 
    size_t embedding_dim, 
    float epsilon) {
    // 添加输入验证
    if (input.empty() || weight.empty()) {
        throw std::invalid_argument("Input vectors cannot be empty");
    }
    if (weight.size() != embedding_dim) {
        throw std::invalid_argument("Weight must match embedding dimension size");
    }

    // 计算序列长度
    size_t seq_len = input.size() / embedding_dim;
    if (seq_len * embedding_dim != input.size()) {
        throw std::invalid_argument("Input size is not divisible by embedding_dim");
    }

    std::vector<float> output(input.size());

    // 分配GPU内存
    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output.size() * sizeof(float)));

    // 复制数据到GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 配置CUDA执行参数
    dim3 blockSize(embedding_dim);
    dim3 gridSize(seq_len);
    size_t sharedMemSize = embedding_dim * sizeof(float);

    // 调用RMSNorm核函数
    rms_norm_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_input, d_weight, d_output, seq_len, embedding_dim, epsilon);

    // 检查核函数执行错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 将结果复制回主机
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // 释放GPU内存
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_weight));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    return output;
}
