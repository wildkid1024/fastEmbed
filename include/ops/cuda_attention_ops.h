#ifndef CUDA_ATTENTION_OPS_H
#define CUDA_ATTENTION_OPS_H

#include "attention_ops_interface.h"
#include <vector>

// 声明 Flash Attention 函数
void flash_attention(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

class CUDAAttentionOps : public AttentionOpsInterface {
public:
    CUDAAttentionOps();
    std::vector<float> multi_head_attention(
        const std::vector<float>& input,
        const std::vector<float>& weight_q,
        const std::vector<float>& weight_k,
        const std::vector<float>& weight_v,
        size_t num_heads,
        size_t embedding_dim
    ) override;
};

#endif // CUDA_ATTENTION_OPS_H
