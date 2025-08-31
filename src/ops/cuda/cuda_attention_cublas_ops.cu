#include "ops/cuda/cuda_attention_cublas_ops.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <cooperative_groups.h>
#include <cublas_v2.h>


// 添加在文件开头的核函数定义区域
// 将 [B, T, NH*HS] 重排为 [B, NH, T, HS]
__global__ void permute_kernel(float* input, float* output, int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * NH * HS) {
        // 计算原始张量中的索引
        int b = idx / (T * NH * HS);
        int t = (idx / (NH * HS)) % T;
        int nh = (idx / HS) % NH;
        int hs = idx % HS;

        // 计算目标张量中的索引
        int new_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

        // 将原始张量中的值复制到目标张量中
        output[new_idx] = input[idx];
    }
}


// 将 [B, NH, T, HS] 重排为 [B, T, NH*HS] (与permute_kernel保持一致)
__global__ void unpermute_kernel(const float* input, float* output, int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * NH * HS) {
        // 计算当前索引在原始张量中的维度分解 (与permute_kernel对称)
        int b = idx / (T * NH * HS);
        int t = (idx / (NH * HS)) % T;
        int nh = (idx / HS) % NH;
        int hs = idx % HS;

        // 计算输入张量的索引 (与permute_kernel的new_idx计算对称)
        int input_idx = b * NH * T * HS + nh * T * HS + t * HS + hs;
        output[idx] = input[input_idx];
    }
}


__global__ void softmax_forward_kernel5(const float* d_scores, float scale, float* d_preatt, int batch_heads, int seq_len) {
    // 修正1: 正确解析三维grid索引
    int b = blockIdx.z;          // batch_heads维度
    int i = blockIdx.y;          // 序列行维度
    int j = threadIdx.x;         // 序列列维度

    // 强化边界检查
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

    // 修正2: 改进max归约（支持非2的幂次长度）
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

    // 修正3: 改进sum归约（支持非2的幂次长度）
    for (int k =  blockDim.x / 2; k > 0; k >>= 1) {
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


CUDAAttentionOps::CUDAAttentionOps() {
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize cuBLAS context");
    }
}

CUDAAttentionOps::~CUDAAttentionOps() {
    if (handle_ != nullptr) {
        cublasDestroy(handle_);
    }
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
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_wq, weight_size);
    cudaMalloc(&d_wk, weight_size);
    cudaMalloc(&d_wv, weight_size);
    cudaMalloc(&d_bq, bias_size);
    cudaMalloc(&d_bk, bias_size);
    cudaMalloc(&d_bv, bias_size);
    cudaMalloc(&d_q, batch_size * seq_len * embedding_dim * sizeof(float));
    cudaMalloc(&d_k, batch_size * seq_len * embedding_dim * sizeof(float));
    cudaMalloc(&d_v, batch_size * seq_len * embedding_dim * sizeof(float));
    cudaMalloc(&d_output, input_size);

    // Copy data to device
    cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wq, weight_q.data(), weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wk, weight_k.data(), weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wv, weight_v.data(), weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bq, query_bias.data(), bias_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bk, key_bias.data(), bias_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bv, value_bias.data(), bias_size, cudaMemcpyHostToDevice);

    // cuBLAS parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const size_t m = batch_size * seq_len;  // Rows of A and C
    const size_t n = embedding_dim;         // Columns of B and C
    const size_t k = embedding_dim;         // Columns of A and rows of B

    // Compute Q = input * Wq^T + bq
    cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha,
                d_wq, n, d_input, n, &beta, d_q, k);

    // Compute K = input * Wk^T + bk
    cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha,
                d_wk, n, d_input, n, &beta, d_k, k);

    // Compute V = input * Wv^T + bv
    cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha,
                d_wv, n, d_input, n, &beta, d_v, k);
    
    for (int i = 0; i < m; ++i) {
        cublasSaxpy(handle_, n, &alpha, d_bq, 1, d_q + i * n, 1);
        cublasSaxpy(handle_, n, &alpha, d_bk, 1, d_k + i * n, 1);
        cublasSaxpy(handle_, n, &alpha, d_bv, 1, d_v + i * n, 1);
   }

    int blockSize = 256;
    int numBlocks = (batch_size * seq_len * num_heads * head_dim + blockSize - 1) / blockSize;

    // 分配中间结果内存
    float *d_q_permuted, *d_k_permuted, *d_v_permuted;
    cudaMalloc(&d_q_permuted, batch_size * seq_len * embedding_dim * sizeof(float));
    cudaMalloc(&d_k_permuted, batch_size * seq_len * embedding_dim * sizeof(float));
    cudaMalloc(&d_v_permuted, batch_size * seq_len * embedding_dim * sizeof(float));

    permute_kernel<<<numBlocks, blockSize>>>(d_q, d_q_permuted, batch_size, seq_len, num_heads, head_dim);
    permute_kernel<<<numBlocks, blockSize>>>(d_k, d_k_permuted, batch_size, seq_len, num_heads, head_dim);
    permute_kernel<<<numBlocks, blockSize>>>(d_v, d_v_permuted, batch_size, seq_len, num_heads, head_dim);

    float *d_scores, *d_attn_output;
    cudaMalloc(&d_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float));
    cudaMalloc(&d_attn_output, batch_size * seq_len * embedding_dim * sizeof(float));
    
    cublasSgemmStridedBatched(
        handle_, 
        CUBLAS_OP_T, 
        CUBLAS_OP_N, 
        seq_len, seq_len, head_dim, 
        &alpha, 
        d_k_permuted, head_dim, seq_len * head_dim,
        d_q_permuted, head_dim, seq_len * head_dim,  
        &beta, 
        d_scores, seq_len, seq_len * seq_len,     
        batch_size  * num_heads
    );


    float* d_preatt;
    cudaMalloc(&d_preatt, batch_size * num_heads * seq_len * seq_len * sizeof(float));

    int softmax_block_size = (seq_len + 31) / 32 * 32; 
    // int grid_size = int(batch_size * num_heads * seq_len);
    int batch_heads = batch_size * num_heads;

    // 修正启动参数（gridDim设置）
    // 启动参数保持不变，但需确保grid_dim正确初始化
    dim3 grid_dim(1, seq_len, batch_heads); // 确保此代码已正确设置
    softmax_forward_kernel5<<<grid_dim, softmax_block_size, 2 * seq_len * sizeof(float)>>>(d_scores, scale, d_preatt, batch_size * num_heads, seq_len);
    cudaDeviceSynchronize();

    cublasSgemmStridedBatched(
        handle_, 
        CUBLAS_OP_N, 
        CUBLAS_OP_N, 
        head_dim, seq_len, seq_len, 
        &alpha, 
        d_v_permuted, head_dim, seq_len * head_dim, 
        d_preatt, seq_len, seq_len * seq_len,
        &beta, 
        d_output, head_dim, seq_len * head_dim, 
        batch_size * num_heads
    );

    unpermute_kernel<<<numBlocks, blockSize>>>(d_output, d_output, batch_size, seq_len, num_heads, head_dim);

    // 释放中间内存
    cudaFree(d_scores);

    // Copy result to host
    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), d_output, input_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_wq); cudaFree(d_wk); cudaFree(d_wv);
    cudaFree(d_bq); cudaFree(d_bk); cudaFree(d_bv);
    cudaFree(d_output);

    return output;
}