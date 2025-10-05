#include "ops/cuda/cuda_attention_ops.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <cooperative_groups.h>

// 定义CUDA错误检查宏
#define CHECK_CUDA_ERROR(call)  \
    do { \
        cudaError_t err = call;    \
        if (err != cudaSuccess) {  \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +  \
                                     std::to_string(__LINE__) + ": " + cudaGetErrorString(err));  \
        } \
    } while (0)

// 将 [B, T, NH*HS] 重排为 [B, NH, T, HS]
__global__ void permute_kernel(float* input, float* output, int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * NH * HS) {
        int b = idx / (T * NH * HS);
        int t = (idx / (NH * HS)) % T;
        int nh = (idx / HS) % NH;
        int hs = idx % HS;
        
        int new_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;
        output[new_idx] = input[idx];
    }
}

// 将 [B, NH, T, HS] 重排为 [B, T, NH*HS]
__global__ void unpermute_kernel(const float* input, float* output, int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * NH * HS) {
        int b = idx / (T * NH * HS);
        int t = (idx / (NH * HS)) % T;
        int nh = (idx / HS) % NH;
        int hs = idx % HS;
        
        int input_idx = b * NH * T * HS + nh * T * HS + t * HS + hs;
        output[idx] = input[input_idx];
    }
}

// 实现数值稳定的softmax
__global__ void softmax_forward_kernel(const float* d_scores, float scale, float* d_preatt, int batch_heads, int seq_len) {
    // 解析三维grid索引
    int b = blockIdx.z;          // batch_heads维度
    int i = blockIdx.y;          // 序列行维度
    int j = threadIdx.x;         // 序列列维度

    // 边界检查
    if (b >= batch_heads || i >= seq_len || j >= seq_len) return;

    // 动态共享内存，分配2*seq_len以避免bank冲突
    extern __shared__ float s_data[];
    float* s_max = s_data;       // 前半段存max归约
    float* s_sum = s_data + seq_len; // 后半段存sum归约

    // 加载数据并应用缩放
    int idx = b * seq_len * seq_len + i * seq_len + j;
    float val = d_scores[idx] * scale;

    s_max[j] = val;
    __syncthreads();

    // 改进max归约（支持非2的幂次长度）
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { 
        if (j < k && j + k < seq_len) {
            s_max[j] = max(s_max[j], s_max[j + k]);
        }
        __syncthreads();
    }
    float max_val = s_max[0];
    __syncthreads();

    // 计算指数值（数值稳定性保障）
    float exp_val = expf(val - max_val);
    s_sum[j] = exp_val;
    __syncthreads();

    d_preatt[idx] = exp_val;

    // 改进sum归约（支持非2的幂次长度）
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (j < k && j + k < seq_len) {
            s_sum[j] += s_sum[j + k];
        }
        __syncthreads();
    }
    float sum_exp = s_sum[0];
    __syncthreads();

    // 计算softmax（防止除零保护）
    d_preatt[idx] = (sum_exp > 0) ? (exp_val / sum_exp) : 0.0f;
}

// 矩阵乘法 (A × B^T) 核函数
__global__ void matrix_multiply_transpose_kernel(const float* A, const float* B, float* C, 
                                               int m, int n, int k, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[col * n + i];
        }
        C[row * k + col] = alpha * sum + beta * C[row * k + col];
    }
}

// 带batch的矩阵乘法核函数，支持选择是否转置B矩阵
__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, 
                                    int batch_size, int m, int n, int k, bool transpose_b = false) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < batch_size && row < m && col < k) {
        const int batch_offset = batch * m * n;
        const int C_batch_offset = batch * m * k;
        
        float sum = 0.0f;
        if (transpose_b) {
            // 计算 A × B^T，B的原始形状为 [batch_size, k, n]
            const int B_batch_offset = batch * k * n;
            for (int i = 0; i < n; ++i) {
                sum += A[batch_offset + row * n + i] * B[B_batch_offset + col * n + i];
            }
        } else {
            // 计算 A × B，B的原始形状为 [batch_size, n, k]
            const int B_batch_offset = batch * n * k;
            for (int i = 0; i < n; ++i) {
                sum += A[batch_offset + row * n + i] * B[B_batch_offset + i * k + col];
            }
        }
        C[C_batch_offset + row * k + col] = sum;
    }
}

// 添加偏置核函数
__global__ void add_bias_kernel(float* data, const float* bias, int batch_size, int seq_len, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len * embed_dim) {
        // int batch = idx / (seq_len * embed_dim);
        // int seq = (idx / embed_dim) % seq_len;
        int dim = idx % embed_dim;
        
        data[idx] += bias[dim];
    }
}

CUDAAttentionOps::CUDAAttentionOps() {
    // 不需要初始化cuBLAS
}

CUDAAttentionOps::~CUDAAttentionOps() {
    // 不需要销毁cuBLAS上下文
}

std::vector<float> CUDAAttentionOps::multi_head_attention(
    const std::vector<float>& input,          // [batch, seq_len, embed_dim]
    const std::vector<float>& weight_q,       // [embed_dim, embed_dim]
    const std::vector<float>& weight_k,       // [embed_dim, embed_dim]
    const std::vector<float>& weight_v,       // [embed_dim, embed_dim]
    const std::vector<float>& query_bias,     // [embed_dim]
    const std::vector<float>& key_bias,       // [embed_dim]
    const std::vector<float>& value_bias,     // [embed_dim]
    size_t num_heads,                         // Number of attention heads
    size_t embedding_dim                      // Embedding dimension per token
) {
    // Calculate derived dimensions
    const size_t batch_size = 1;  
    const size_t seq_len = (input.size() / embedding_dim) / batch_size;
    const size_t head_dim = embedding_dim / num_heads;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Check input validity
    if (embedding_dim % num_heads != 0) {
        throw std::invalid_argument("Embedding dimension must be divisible by number of heads");
    }
    if (input.size() % (batch_size * seq_len * embedding_dim) != 0) {
        throw std::invalid_argument("Invalid input size for given dimensions");
    }

    // Device pointers
    float *d_input, *d_q, *d_k, *d_v, *d_output;
    float *d_wq, *d_wk, *d_wv, *d_bq, *d_bk, *d_bv;
    const size_t input_size = input.size() * sizeof(float);
    const size_t weight_size = weight_q.size() * sizeof(float);
    const size_t bias_size = query_bias.size() * sizeof(float);

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_wq, weight_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_wk, weight_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_wv, weight_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bq, bias_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bk, bias_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bv, bias_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_q, batch_size * seq_len * embedding_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_k, batch_size * seq_len * embedding_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v, batch_size * seq_len * embedding_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, input_size));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_wq, weight_q.data(), weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_wk, weight_k.data(), weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_wv, weight_v.data(), weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bq, query_bias.data(), bias_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bk, key_bias.data(), bias_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bv, value_bias.data(), bias_size, cudaMemcpyHostToDevice));

    // 矩阵乘法参数
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const size_t m = batch_size * seq_len;  // Rows of A and C
    const size_t n = embedding_dim;         // Columns of B and rows of A
    const size_t k = embedding_dim;         // Columns of B^T and C

    // 使用核函数替代cuBLAS计算矩阵乘法
    // 计算 Q = input * Wq^T + bq
    dim3 block_size(16, 16);
    dim3 grid_size((k + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);
    matrix_multiply_transpose_kernel<<<grid_size, block_size>>>(
        d_input, d_wq, d_q, m, n, k, alpha, beta);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 计算 K = input * Wk^T + bk
    matrix_multiply_transpose_kernel<<<grid_size, block_size>>>(
        d_input, d_wk, d_k, m, n, k, alpha, beta);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 计算 V = input * Wv^T + bv
    matrix_multiply_transpose_kernel<<<grid_size, block_size>>>(
        d_input, d_wv, d_v, m, n, k, alpha, beta);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 添加偏置
    int add_bias_block_size = 256;
    int add_bias_grid_size = (batch_size * seq_len * embedding_dim + add_bias_block_size - 1) / add_bias_block_size;
    add_bias_kernel<<<add_bias_grid_size, add_bias_block_size>>>(d_q, d_bq, batch_size, seq_len, embedding_dim);
    add_bias_kernel<<<add_bias_grid_size, add_bias_block_size>>>(d_k, d_bk, batch_size, seq_len, embedding_dim);
    add_bias_kernel<<<add_bias_grid_size, add_bias_block_size>>>(d_v, d_bv, batch_size, seq_len, embedding_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 分配中间结果内存
    float *d_q_permuted, *d_k_permuted, *d_v_permuted;
    CHECK_CUDA_ERROR(cudaMalloc(&d_q_permuted, batch_size * seq_len * embedding_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_k_permuted, batch_size * seq_len * embedding_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v_permuted, batch_size * seq_len * embedding_dim * sizeof(float)));

    // 维度重排
    int permute_block_size = 256;
    int permute_grid_size = (batch_size * seq_len * num_heads * head_dim + permute_block_size - 1) / permute_block_size;
    permute_kernel<<<permute_grid_size, permute_block_size>>>(d_q, d_q_permuted, batch_size, seq_len, num_heads, head_dim);
    permute_kernel<<<permute_grid_size, permute_block_size>>>(d_k, d_k_permuted, batch_size, seq_len, num_heads, head_dim);
    permute_kernel<<<permute_grid_size, permute_block_size>>>(d_v, d_v_permuted, batch_size, seq_len, num_heads, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 计算注意力分数 Q * K^T
    float *d_scores;
    CHECK_CUDA_ERROR(cudaMalloc(&d_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float)));
    
    dim3 matmul_block_size(32, 32);
    dim3 matmul_grid_size(
        (seq_len + matmul_block_size.x - 1) / matmul_block_size.x,
        (seq_len + matmul_block_size.y - 1) / matmul_block_size.y,
        batch_size * num_heads
    );
    batched_matmul_kernel<<<matmul_grid_size, matmul_block_size>>>(
        d_q_permuted, d_k_permuted, d_scores, 
        batch_size * num_heads, seq_len, head_dim, seq_len, true
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 应用softmax
    float *d_preatt;
    CHECK_CUDA_ERROR(cudaMalloc(&d_preatt, batch_size * num_heads * seq_len * seq_len * sizeof(float)));
    
    dim3 softmax_grid_size(1, seq_len, batch_size * num_heads);
    int softmax_block_size = (seq_len + 31) / 32 * 32;  // 向上取整到32的倍数
    softmax_forward_kernel<<<softmax_grid_size, softmax_block_size, 2 * softmax_block_size * sizeof(float)>>>(
        d_scores, scale, d_preatt, batch_size * num_heads, seq_len
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 计算注意力输出 (preatt * V)
    float *d_attn_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_attn_output, batch_size * num_heads * seq_len * head_dim * sizeof(float)));
    
    dim3 attn_output_grid_size(
        (head_dim + matmul_block_size.x - 1) / matmul_block_size.x,
        (seq_len + matmul_block_size.y - 1) / matmul_block_size.y,
        batch_size * num_heads
    );
    batched_matmul_kernel<<<attn_output_grid_size, matmul_block_size>>>(
        d_preatt, d_v_permuted, d_attn_output, 
        batch_size * num_heads, seq_len, seq_len, head_dim
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 打印attn_output以调试
    std::vector<float> attn_output(batch_size * num_heads * seq_len * head_dim);
    CHECK_CUDA_ERROR(cudaMemcpy(attn_output.data(), d_attn_output, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost));

    // 恢复原始维度
    unpermute_kernel<<<permute_grid_size, permute_block_size>>>(d_attn_output, d_output, batch_size, seq_len, num_heads, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 释放中间内存
    CHECK_CUDA_ERROR(cudaFree(d_q_permuted));
    CHECK_CUDA_ERROR(cudaFree(d_k_permuted));
    CHECK_CUDA_ERROR(cudaFree(d_v_permuted));
    CHECK_CUDA_ERROR(cudaFree(d_scores));
    CHECK_CUDA_ERROR(cudaFree(d_preatt));
    CHECK_CUDA_ERROR(cudaFree(d_attn_output));

    // 修复output_size未定义的错误
    // 拷贝结果回主机
    std::vector<float> output(input.size());
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_output, input_size, cudaMemcpyDeviceToHost));

    // 清理所有设备内存
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_wq));
    CHECK_CUDA_ERROR(cudaFree(d_wk));
    CHECK_CUDA_ERROR(cudaFree(d_wv));
    CHECK_CUDA_ERROR(cudaFree(d_bq));
    CHECK_CUDA_ERROR(cudaFree(d_bk));
    CHECK_CUDA_ERROR(cudaFree(d_bv));
    CHECK_CUDA_ERROR(cudaFree(d_q));
    CHECK_CUDA_ERROR(cudaFree(d_k));
    CHECK_CUDA_ERROR(cudaFree(d_v));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    return output;
}