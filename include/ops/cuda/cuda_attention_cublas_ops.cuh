#pragma once
#include <vector>
#include <cublas_v2.h>

class CUDAAttentionOps {
public:
    CUDAAttentionOps();
    ~CUDAAttentionOps();

    std::vector<float> multi_head_attention(
        const std::vector<float>& input,          // [batch, seq_len, embed_dim]
        const std::vector<float>& weight_q,       // [embed_dim, embed_dim]
        const std::vector<float>& weight_k,
        const std::vector<float>& weight_v,
        const std::vector<float>& query_bias,     // [embed_dim]
        const std::vector<float>& key_bias,
        const std::vector<float>& value_bias,
        size_t num_heads,
        size_t embedding_dim);

private:
    cublasHandle_t handle_ = nullptr;
};