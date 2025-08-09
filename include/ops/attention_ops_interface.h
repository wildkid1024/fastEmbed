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
        const std::vector<float>& query_bias,  // New: query bias
        const std::vector<float>& key_bias,    // New: key bias
        const std::vector<float>& value_bias,  // New: value bias
        size_t num_heads,
        size_t embedding_dim
    ) = 0;
};

#endif // ATTENTION_OPS_INTERFACE_H