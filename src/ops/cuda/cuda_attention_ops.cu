#include "cuda_flash_attention.h"
#include "cuda_matrix_ops.h"
#include <vector>
#include <cuda_runtime.h>

// Include the header file with CUDAAttentionOps declaration
#include "ops/cuda_attention_ops.h"

// Implement the multi_head_attention function
std::vector<float> CUDAAttentionOps::multi_head_attention(
    const std::vector<float>& input,
    const std::vector<float>& weight_q,
    const std::vector<float>& weight_k,
    const std::vector<float>& weight_v,
    size_t num_heads,
    size_t embedding_dim
) {
    size_t seq_len = input.size() / embedding_dim;
    size_t head_dim = embedding_dim / num_heads;
    size_t batch_size = 1; // 假设 batch size 为 1

    // 计算 Q, K, V 矩阵
    CUDAMatrixOps matrix_ops;
    std::vector<float> Q = matrix_ops.matrix_multiply(input, weight_q, seq_len, embedding_dim, embedding_dim);
    std::vector<float> K = matrix_ops.matrix_multiply(input, weight_k, seq_len, embedding_dim, embedding_dim);
    std::vector<float> V = matrix_ops.matrix_multiply(input, weight_v, seq_len, embedding_dim, embedding_dim);

    // 分配输出缓冲区
    std::vector<float> output(seq_len * embedding_dim);

    // 分配 CUDA 设备内存
    float *d_q, *d_k, *d_v, *d_out;
    cudaMalloc((void**)&d_q, Q.size() * sizeof(float));
    cudaMalloc((void**)&d_k, K.size() * sizeof(float));
    cudaMalloc((void**)&d_v, V.size() * sizeof(float));
    cudaMalloc((void**)&d_out, output.size() * sizeof(float));

    // 将数据复制到设备
    cudaMemcpy(d_q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 调用 Flash Attention
    flash_attention(d_q, d_k, d_v, d_out, static_cast<int>(batch_size), static_cast<int>(seq_len), static_cast<int>(num_heads), static_cast<int>(head_dim));

    // 将结果复制回主机
    cudaMemcpy(output.data(), d_out, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);

    return output;
}
