#ifndef ATTENTION_OPS_INTERFACE_H
#define ATTENTION_OPS_INTERFACE_H

#include <vector>

class AttentionOpsInterface {
public:
    virtual ~AttentionOpsInterface() = default;
    // Add multi_head_attention virtual function declaration
    virtual std::vector<float> multi_head_attention(
        const std::vector<float>& input,
        const std::vector<float>& weight_q,
        const std::vector<float>& weight_k,
        const std::vector<float>& weight_v,
        size_t num_heads,
        size_t embedding_dim
    ) = 0;
};

#endif // ATTENTION_OPS_INTERFACE_H
