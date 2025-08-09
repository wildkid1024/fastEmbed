#ifndef CUDA_ATTENTION_OPS_H
#define CUDA_ATTENTION_OPS_H

#include "attention_ops_interface.h"  // 包含头文件
#include <vector>

class CUDAAttentionOps : public AttentionOpsInterface {
public:
    std::vector<float> multi_head_attention(
        const std::vector<float>& input,
        const std::vector<float>& weight_q,
        const std::vector<float>& weight_k,
        const std::vector<float>& weight_v,
        const std::vector<float>& query_bias,  // Add query bias
        const std::vector<float>& key_bias,    // Add key bias
        const std::vector<float>& value_bias,  // Add value bias
        size_t num_heads,
        size_t embedding_dim
    ) override;
};

#endif // CUDA_ATTENTION_OPS_H