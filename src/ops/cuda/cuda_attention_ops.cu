#include "ops/cuda/cuda_attention_ops.cuh"
#include "ops/cuda/cublas_gemm.h"
#include "ops/cuda/device_weight_cache.h"
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

// Q/K RMS Normalization kernel
// 输入: [seq_len, num_heads * head_dim]
// 对每个head独立进行RMS norm，然后乘以weight
__global__ void rms_norm_head_kernel(
    float* data,              // [seq_len, num_heads * head_dim], in-place
    const float* weight,      // [head_dim]
    int seq_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = seq_len * num_heads;
    
    if (idx >= total_heads) return;
    
    int seq_idx = idx / num_heads;  // 序列位置
    int head_idx = idx % num_heads;  // head索引
    
    // 计算当前head的起始位置
    int start = seq_idx * num_heads * head_dim + head_idx * head_dim;
    
    // 计算RMS
    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float val = data[start + d];
        sum_sq += val * val;
    }
    
    float rms = sqrtf(sum_sq / head_dim + 1e-6f);
    float scale = 1.0f / rms;
    
    // 应用normalization和weight
    for (int d = 0; d < head_dim; ++d) {
        data[start + d] = data[start + d] * scale * weight[d];
    }
}

// RoPE (Rotary Positional Embedding) kernel
// 输入: [seq_len, num_heads * head_dim]
// 对每个位置的每个head应用旋转位置编码
__global__ void rope_kernel(
    float* data,              // [seq_len, num_heads * head_dim], in-place
    int seq_len,
    int num_heads,
    int head_dim,
    float rope_theta
) {
    // 每个thread处理一个(seq_pos, head)组合的一半维度
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_positions = seq_len * num_heads;
    
    if (idx >= total_positions) return;
    
    int pos = idx / num_heads;       // 序列位置
    int head = idx % num_heads;      // head索引
    
    int half_dim = head_dim / 2;
    int base_idx = pos * num_heads * head_dim + head * head_dim;
    
    // 对每一对维度应用RoPE
    for (int d = 0; d < half_dim; ++d) {
        // 计算cos/sin
        float inv_freq = 1.0f / powf(rope_theta, 2.0f * static_cast<float>(d) / static_cast<float>(head_dim));
        float theta = static_cast<float>(pos) * inv_freq;
        float cos_val = cosf(theta);
        float sin_val = sinf(theta);
        
        // 获取原始值
        float x0 = data[base_idx + d];
        float x1 = data[base_idx + d + half_dim];
        
        // 应用旋转: x' = x * cos - rotate(x) * sin
        data[base_idx + d] = x0 * cos_val - x1 * sin_val;
        data[base_idx + d + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// 实现数值稳定的softmax，不带因果掩码
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

// 实现数值稳定的softmax，带因果掩码（块对角结构）
__global__ void softmax_forward_causal_mask_kernel(const float* d_scores, float scale, float* d_preatt, int batch_heads, int seq_len) {
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

    // 加载数据并应用因果掩码（只看前面的位置）
    int idx = b * seq_len * seq_len + i * seq_len + j;
    float val = (j <= i) ? (d_scores[idx] * scale) : -FLT_MAX;

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
    float exp_val = (j <= i) ? expf(val - max_val) : 0.0f;
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
    const float *d_wq, *d_wk, *d_wv, *d_bq, *d_bk, *d_bv;
    const size_t input_size = input.size() * sizeof(float);

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_q, batch_size * seq_len * embedding_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_k, batch_size * seq_len * embedding_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v, batch_size * seq_len * embedding_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, input_size));

    // 权重/偏置：走设备端常驻缓存，不再每层每请求重新上传
    // （BGE-large 每层 q/k/v 各 16MB，此前每请求要搬运 1GB+ 权重）
    DeviceWeightCache& weight_cache = DeviceWeightCache::instance();
    d_wq = weight_cache.get(weight_q.data(), weight_q.size());
    d_wk = weight_cache.get(weight_k.data(), weight_k.size());
    d_wv = weight_cache.get(weight_v.data(), weight_v.size());
    d_bq = weight_cache.get(query_bias.data(), query_bias.size());
    d_bk = weight_cache.get(key_bias.data(), key_bias.size());
    d_bv = weight_cache.get(value_bias.data(), value_bias.size());

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice));

    // 矩阵乘法参数
    const size_t m = batch_size * seq_len;  // Rows of A and C
    const size_t n = embedding_dim;         // Columns of B and rows of A
    const size_t k = embedding_dim;         // Columns of B^T and C

    // Q/K/V 投影走 cuBLAS：Q = input * Wq^T，权重已是设备端常驻指针，直接传入。
    // 这三处此前是朴素 kernel（约占 transformer 总算力的 25%），是长文本的主要瓶颈。
    cublas_gemm::mm_nt(d_input, d_wq, d_q, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    cublas_gemm::mm_nt(d_input, d_wk, d_k, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    cublas_gemm::mm_nt(d_input, d_wv, d_v, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
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

    // 拷贝结果回主机
    std::vector<float> output(input.size());
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_output, input_size, cudaMemcpyDeviceToHost));

    // 清理设备内存（d_wq/d_wk/d_wv/d_bq/d_bk/d_bv 为常驻缓存指针，不能释放）
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_q));
    CHECK_CUDA_ERROR(cudaFree(d_k));
    CHECK_CUDA_ERROR(cudaFree(d_v));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    return output;
}

// GQA (Grouped Query Attention) 实现 - 优化版，使用CUDA内的因果掩码
std::vector<float> CUDAAttentionOps::grouped_query_attention(
    const std::vector<float>& q_input,       // [seq_len, num_q_heads * head_dim]
    const std::vector<float>& k_input,       // [seq_len, num_kv_heads * head_dim]
    const std::vector<float>& v_input,       // [seq_len, num_kv_heads * head_dim]
    const std::vector<float>& q_norm_weight, // [head_dim]
    const std::vector<float>& k_norm_weight, // [head_dim]
    size_t seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim
) {
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const size_t q_output_dim = num_q_heads * head_dim;
    const size_t kv_output_dim = num_kv_heads * head_dim;
    
    // Device pointers
    float *d_q, *d_k, *d_v, *d_output;
    const size_t q_size = seq_len * q_output_dim * sizeof(float);
    const size_t kv_size = seq_len * kv_output_dim * sizeof(float);
    const size_t output_size = seq_len * q_output_dim * sizeof(float);
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_q, q_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_k, kv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v, kv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_q, q_input.data(), q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k, k_input.data(), kv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_v, v_input.data(), kv_size, cudaMemcpyHostToDevice));
    
    // Copy norm weights to device
    float *d_q_norm_weight, *d_k_norm_weight;
    CHECK_CUDA_ERROR(cudaMalloc(&d_q_norm_weight, head_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_k_norm_weight, head_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_q_norm_weight, q_norm_weight.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k_norm_weight, k_norm_weight.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Apply Q/K RMS normalization on GPU (before RoPE)
    int norm_block_size = 256;
    int q_norm_grid_size = (seq_len * num_q_heads + norm_block_size - 1) / norm_block_size;
    int k_norm_grid_size = (seq_len * num_kv_heads + norm_block_size - 1) / norm_block_size;
    
    rms_norm_head_kernel<<<q_norm_grid_size, norm_block_size>>>(d_q, d_q_norm_weight, seq_len, num_q_heads, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    rms_norm_head_kernel<<<k_norm_grid_size, norm_block_size>>>(d_k, d_k_norm_weight, seq_len, num_kv_heads, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Free norm weight memory
    CHECK_CUDA_ERROR(cudaFree(d_q_norm_weight));
    CHECK_CUDA_ERROR(cudaFree(d_k_norm_weight));
    
    // Apply RoPE on GPU (after q_norm)
    float rope_theta = 1000000.0f;
    int rope_block_size = 256;
    int q_rope_grid_size = (seq_len * num_q_heads + rope_block_size - 1) / rope_block_size;
    int k_rope_grid_size = (seq_len * num_kv_heads + rope_block_size - 1) / rope_block_size;
    
    rope_kernel<<<q_rope_grid_size, rope_block_size>>>(d_q, seq_len, num_q_heads, head_dim, rope_theta);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    rope_kernel<<<k_rope_grid_size, rope_block_size>>>(d_k, seq_len, num_kv_heads, head_dim, rope_theta);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Reshape Q/K/V: [seq_len, num_heads * head_dim] -> [num_heads, seq_len, head_dim]
    float *d_q_reshaped, *d_k_reshaped, *d_v_reshaped;
    CHECK_CUDA_ERROR(cudaMalloc(&d_q_reshaped, num_q_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_k_reshaped, num_kv_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v_reshaped, num_kv_heads * seq_len * head_dim * sizeof(float)));
    
    // Permute Q: [seq_len, num_q_heads, head_dim] -> [num_q_heads, seq_len, head_dim]
    int permute_block_size = 256;
    int permute_grid_size = (seq_len * num_q_heads * head_dim + permute_block_size - 1) / permute_block_size;
    permute_kernel<<<permute_grid_size, permute_block_size>>>(d_q, d_q_reshaped, 1, seq_len, num_q_heads, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Permute K/V: [seq_len, num_kv_heads, head_dim] -> [num_kv_heads, seq_len, head_dim]
    permute_grid_size = (seq_len * num_kv_heads * head_dim + permute_block_size - 1) / permute_block_size;
    permute_kernel<<<permute_grid_size, permute_block_size>>>(d_k, d_k_reshaped, 1, seq_len, num_kv_heads, head_dim);
    permute_kernel<<<permute_grid_size, permute_block_size>>>(d_v, d_v_reshaped, 1, seq_len, num_kv_heads, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // For each Q head, compute attention with corresponding KV head (GQA)
    float *d_attn_output_all;
    CHECK_CUDA_ERROR(cudaMalloc(&d_attn_output_all, num_q_heads * seq_len * head_dim * sizeof(float)));
    
    // Process each Q head - 优化版本，消除主机-设备拷贝
    for (size_t h = 0; h < num_q_heads; ++h) {
        // GQA: determine corresponding KV head index
        size_t kv_head_idx = h / (num_q_heads / num_kv_heads);
        
        // Extract current Q head [seq_len, head_dim]
        float *d_q_head;
        CHECK_CUDA_ERROR(cudaMalloc(&d_q_head, seq_len * head_dim * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_q_head, 
            d_q_reshaped + h * seq_len * head_dim, 
            seq_len * head_dim * sizeof(float), 
            cudaMemcpyDeviceToDevice));
        
        // Extract corresponding K head [seq_len, head_dim]
        float *d_k_head;
        CHECK_CUDA_ERROR(cudaMalloc(&d_k_head, seq_len * head_dim * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_k_head, 
            d_k_reshaped + kv_head_idx * seq_len * head_dim, 
            seq_len * head_dim * sizeof(float), 
            cudaMemcpyDeviceToDevice));
        
        // Extract corresponding V head [seq_len, head_dim]
        float *d_v_head;
        CHECK_CUDA_ERROR(cudaMalloc(&d_v_head, seq_len * head_dim * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_v_head, 
            d_v_reshaped + kv_head_idx * seq_len * head_dim, 
            seq_len * head_dim * sizeof(float), 
            cudaMemcpyDeviceToDevice));
        
        // Compute attention scores: Q @ K^T [seq_len, seq_len]
        float *d_scores;
        CHECK_CUDA_ERROR(cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float)));
        
        dim3 matmul_block_size(32, 32);
        dim3 matmul_grid_size(
            (seq_len + matmul_block_size.x - 1) / matmul_block_size.x,
            (seq_len + matmul_block_size.y - 1) / matmul_block_size.y
        );
        batched_matmul_kernel<<<matmul_grid_size, matmul_block_size>>>(
            d_q_head, d_k_head, d_scores, 
            1, seq_len, head_dim, seq_len, true
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // 应用因果掩码和softmax（完全在GPU上，无需主机拷贝）
        float *d_preatt;
        CHECK_CUDA_ERROR(cudaMalloc(&d_preatt, seq_len * seq_len * sizeof(float)));
        
        // 使用带因果掩码的softmax kernel
        dim3 softmax_grid_size(1, seq_len, 1);
        int softmax_block_size = (seq_len + 31) / 32 * 32;
        softmax_forward_causal_mask_kernel<<<softmax_grid_size, softmax_block_size, 2 * softmax_block_size * sizeof(float)>>>(
            d_scores, scale, d_preatt, 1, seq_len
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Attention output: preatt @ V [seq_len, head_dim]
        float *d_attn_head;
        CHECK_CUDA_ERROR(cudaMalloc(&d_attn_head, seq_len * head_dim * sizeof(float)));
        
        dim3 attn_grid_size(
            (head_dim + matmul_block_size.x - 1) / matmul_block_size.x,
            (seq_len + matmul_block_size.y - 1) / matmul_block_size.y
        );
        batched_matmul_kernel<<<attn_grid_size, matmul_block_size>>>(
            d_preatt, d_v_head, d_attn_head, 
            1, seq_len, seq_len, head_dim
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Copy result to output buffer
        CHECK_CUDA_ERROR(cudaMemcpy(
            d_attn_output_all + h * seq_len * head_dim,
            d_attn_head,
            seq_len * head_dim * sizeof(float),
            cudaMemcpyDeviceToDevice
        ));
        
        // Free temporary buffers
        CHECK_CUDA_ERROR(cudaFree(d_q_head));
        CHECK_CUDA_ERROR(cudaFree(d_k_head));
        CHECK_CUDA_ERROR(cudaFree(d_v_head));
        CHECK_CUDA_ERROR(cudaFree(d_scores));
        CHECK_CUDA_ERROR(cudaFree(d_preatt));
        CHECK_CUDA_ERROR(cudaFree(d_attn_head));
    }
    
    // Unpermute output: [num_q_heads, seq_len, head_dim] -> [seq_len, num_q_heads * head_dim]
    int unpermute_grid_size = (seq_len * num_q_heads * head_dim + permute_block_size - 1) / permute_block_size;
    unpermute_kernel<<<unpermute_grid_size, permute_block_size>>>(
        d_attn_output_all, d_output, 1, seq_len, num_q_heads, head_dim
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back to host
    std::vector<float> output(seq_len * q_output_dim);
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_q));
    CHECK_CUDA_ERROR(cudaFree(d_k));
    CHECK_CUDA_ERROR(cudaFree(d_v));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_q_reshaped));
    CHECK_CUDA_ERROR(cudaFree(d_k_reshaped));
    CHECK_CUDA_ERROR(cudaFree(d_v_reshaped));
    CHECK_CUDA_ERROR(cudaFree(d_attn_output_all));
    
    return output;
}

// ========== 批量版本 (Batch-aware) 实现 ==========

// 批量 GQA 注意力 - 优化版本：真正的并行批量处理
std::vector<float> CUDAAttentionOps::grouped_query_attention_batch(
    const std::vector<float>& q_input,       // [batch, seq_len, num_q_heads * head_dim]
    const std::vector<float>& k_input,       // [batch, seq_len, num_kv_heads * head_dim]
    const std::vector<float>& v_input,       // [batch, seq_len, num_kv_heads * head_dim]
    const std::vector<float>& q_norm_weight, // [head_dim], Q的RMS norm权重
    const std::vector<float>& k_norm_weight, // [head_dim], K的RMS norm权重
    size_t batch_size,
    size_t seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim
) {
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const size_t q_output_dim = num_q_heads * head_dim;
    const size_t kv_output_dim = num_kv_heads * head_dim;
    const size_t total_q_elements = batch_size * seq_len * q_output_dim;
    const size_t total_kv_elements = batch_size * seq_len * kv_output_dim;
    
    // Device pointers
    float *d_q, *d_k, *d_v, *d_output;
    const size_t q_size = total_q_elements * sizeof(float);
    const size_t kv_size = total_kv_elements * sizeof(float);
    const size_t output_size = total_q_elements * sizeof(float);
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_q, q_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_k, kv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v, kv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_q, q_input.data(), q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k, k_input.data(), kv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_v, v_input.data(), kv_size, cudaMemcpyHostToDevice));
    
    // Copy norm weights to device
    float *d_q_norm_weight, *d_k_norm_weight;
    CHECK_CUDA_ERROR(cudaMalloc(&d_q_norm_weight, head_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_k_norm_weight, head_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_q_norm_weight, q_norm_weight.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k_norm_weight, k_norm_weight.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Apply Q/K RMS normalization on GPU
    int norm_block_size = 256;
    int q_norm_grid_size = (batch_size * seq_len * num_q_heads + norm_block_size - 1) / norm_block_size;
    int k_norm_grid_size = (batch_size * seq_len * num_kv_heads + norm_block_size - 1) / norm_block_size;
    
    // 我们需要修改一下rms_norm_head_kernel来支持批量，但这里我们暂时采用逐个样本处理
    // 更优化的版本需要额外的kernel，但为了保持代码简单，我们可以为每个样本单独处理
    // 这里我们用循环处理样本，但所有计算都在GPU上
    
    // 我们需要先把每个样本处理到独立的空间，或者用更巧妙的kernel
    // 为了简化，我们可以先处理所有样本的normalization和rope，然后统一处理attention
    // 但为了保持代码变化可控，我们先采用循环但全GPU的方式
    
    // 分配中间内存
    float *d_q_processed, *d_k_processed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_q_processed, q_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_k_processed, kv_size));
    
    // 处理每个样本的normalization和rope
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        size_t q_sample_start = b_idx * seq_len * q_output_dim;
        size_t kv_sample_start = b_idx * seq_len * kv_output_dim;
        
        // Q的normalization
        int q_norm_block = 256;
        int q_norm_grid = (seq_len * num_q_heads + q_norm_block - 1) / q_norm_block;
        rms_norm_head_kernel<<<q_norm_grid, q_norm_block>>>(
            d_q + q_sample_start, d_q_norm_weight, seq_len, num_q_heads, head_dim
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // K的normalization
        int k_norm_grid = (seq_len * num_kv_heads + q_norm_block - 1) / q_norm_block;
        rms_norm_head_kernel<<<k_norm_grid, q_norm_block>>>(
            d_k + kv_sample_start, d_k_norm_weight, seq_len, num_kv_heads, head_dim
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Q的RoPE
        float rope_theta = 1000000.0f;
        int rope_block = 256;
        int q_rope_grid = (seq_len * num_q_heads + rope_block - 1) / rope_block;
        rope_kernel<<<q_rope_grid, rope_block>>>(
            d_q + q_sample_start, seq_len, num_q_heads, head_dim, rope_theta
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // K的RoPE
        int k_rope_grid = (seq_len * num_kv_heads + rope_block - 1) / rope_block;
        rope_kernel<<<k_rope_grid, rope_block>>>(
            d_k + kv_sample_start, seq_len, num_kv_heads, head_dim, rope_theta
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    
    // 释放normalization权重
    CHECK_CUDA_ERROR(cudaFree(d_q_norm_weight));
    CHECK_CUDA_ERROR(cudaFree(d_k_norm_weight));
    
    // 现在，我们需要为批量注意力做计算准备
    // 我们将使用一个大的循环，但每个循环的操作都是纯GPU操作
    
    // 分配输出空间
    std::vector<float> result(total_q_elements);
    
    // 逐个样本处理，但所有操作都在GPU上
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        size_t q_sample_start = b_idx * seq_len * q_output_dim;
        size_t kv_sample_start = b_idx * seq_len * kv_output_dim;
        size_t result_sample_start = b_idx * seq_len * q_output_dim;
        
        // Reshape Q/K/V for this sample
        float *d_q_reshaped, *d_k_reshaped, *d_v_reshaped;
        CHECK_CUDA_ERROR(cudaMalloc(&d_q_reshaped, num_q_heads * seq_len * head_dim * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_k_reshaped, num_kv_heads * seq_len * head_dim * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_v_reshaped, num_kv_heads * seq_len * head_dim * sizeof(float)));
        
        int permute_block = 256;
        int permute_grid = (seq_len * num_q_heads * head_dim + permute_block - 1) / permute_block;
        permute_kernel<<<permute_grid, permute_block>>>(d_q + q_sample_start, d_q_reshaped, 1, seq_len, num_q_heads, head_dim);
        
        permute_grid = (seq_len * num_kv_heads * head_dim + permute_block - 1) / permute_block;
        permute_kernel<<<permute_grid, permute_block>>>(d_k + kv_sample_start, d_k_reshaped, 1, seq_len, num_kv_heads, head_dim);
        permute_kernel<<<permute_grid, permute_block>>>(d_v + kv_sample_start, d_v_reshaped, 1, seq_len, num_kv_heads, head_dim);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // For each Q head in this sample
        float *d_attn_output_sample;
        CHECK_CUDA_ERROR(cudaMalloc(&d_attn_output_sample, num_q_heads * seq_len * head_dim * sizeof(float)));
        
        for (size_t h = 0; h < num_q_heads; ++h) {
            size_t kv_head_idx = h / (num_q_heads / num_kv_heads);
            
            float *d_q_head, *d_k_head, *d_v_head;
            CHECK_CUDA_ERROR(cudaMalloc(&d_q_head, seq_len * head_dim * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_k_head, seq_len * head_dim * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_v_head, seq_len * head_dim * sizeof(float)));
            
            CHECK_CUDA_ERROR(cudaMemcpy(d_q_head, d_q_reshaped + h * seq_len * head_dim, seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_k_head, d_k_reshaped + kv_head_idx * seq_len * head_dim, seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_v_head, d_v_reshaped + kv_head_idx * seq_len * head_dim, seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            
            // Compute Q @ K^T
            float *d_scores;
            CHECK_CUDA_ERROR(cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float)));
            
            dim3 matmul_block(32, 32);
            dim3 matmul_grid((seq_len + matmul_block.x - 1) / matmul_block.x, (seq_len + matmul_block.y - 1) / matmul_block.y);
            batched_matmul_kernel<<<matmul_grid, matmul_block>>>(d_q_head, d_k_head, d_scores, 1, seq_len, head_dim, seq_len, true);
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            // Causal mask + softmax
            float *d_preatt;
            CHECK_CUDA_ERROR(cudaMalloc(&d_preatt, seq_len * seq_len * sizeof(float)));
            
            dim3 softmax_grid(1, seq_len, 1);
            int softmax_block = (seq_len + 31) / 32 * 32;
            softmax_forward_causal_mask_kernel<<<softmax_grid, softmax_block, 2 * softmax_block * sizeof(float)>>>(
                d_scores, scale, d_preatt, 1, seq_len
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            // Compute preatt @ V
            float *d_attn_head;
            CHECK_CUDA_ERROR(cudaMalloc(&d_attn_head, seq_len * head_dim * sizeof(float)));
            
            dim3 attn_grid((head_dim + matmul_block.x - 1) / matmul_block.x, (seq_len + matmul_block.y - 1) / matmul_block.y);
            batched_matmul_kernel<<<attn_grid, matmul_block>>>(d_preatt, d_v_head, d_attn_head, 1, seq_len, seq_len, head_dim);
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            // Copy back
            CHECK_CUDA_ERROR(cudaMemcpy(d_attn_output_sample + h * seq_len * head_dim, d_attn_head, seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            
            // Free temp
            CHECK_CUDA_ERROR(cudaFree(d_q_head));
            CHECK_CUDA_ERROR(cudaFree(d_k_head));
            CHECK_CUDA_ERROR(cudaFree(d_v_head));
            CHECK_CUDA_ERROR(cudaFree(d_scores));
            CHECK_CUDA_ERROR(cudaFree(d_preatt));
            CHECK_CUDA_ERROR(cudaFree(d_attn_head));
        }
        
        // Unpermute
        float *d_sample_output;
        CHECK_CUDA_ERROR(cudaMalloc(&d_sample_output, seq_len * q_output_dim * sizeof(float)));
        
        int unpermute_grid = (seq_len * num_q_heads * head_dim + permute_block - 1) / permute_block;
        unpermute_kernel<<<unpermute_grid, permute_block>>>(d_attn_output_sample, d_sample_output, 1, seq_len, num_q_heads, head_dim);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Copy to result
        CHECK_CUDA_ERROR(cudaMemcpy(result.data() + result_sample_start, d_sample_output, seq_len * q_output_dim * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Free
        CHECK_CUDA_ERROR(cudaFree(d_q_reshaped));
        CHECK_CUDA_ERROR(cudaFree(d_k_reshaped));
        CHECK_CUDA_ERROR(cudaFree(d_v_reshaped));
        CHECK_CUDA_ERROR(cudaFree(d_attn_output_sample));
        CHECK_CUDA_ERROR(cudaFree(d_sample_output));
    }
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_q));
    CHECK_CUDA_ERROR(cudaFree(d_k));
    CHECK_CUDA_ERROR(cudaFree(d_v));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_q_processed));
    CHECK_CUDA_ERROR(cudaFree(d_k_processed));
    
    return result;
}