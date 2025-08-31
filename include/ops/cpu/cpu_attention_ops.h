#ifndef CPU_ATTENTION_OPS_H
#define CPU_ATTENTION_OPS_H

#include "attention_ops_interface.h"
#include "cpu_matrix_ops.h"

class CPUAttentionOps : public AttentionOpsInterface {
public:
    CPUAttentionOps();
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
    );

    std::vector<float> multi_head_attentionv2(
        const std::vector<float>& input, 
        const std::vector<float>& weight_q, 
        const std::vector<float>& weight_k, 
        const std::vector<float>& weight_v, 
        const std::vector<float>& query_bias,  // Add query bias
        const std::vector<float>& key_bias,    // Add key bias
        const std::vector<float>& value_bias,  // Add value bias
        size_t num_heads, 
        size_t embedding_dim
    );

private:
    CPUMatrixOps matrix_ops;
};

#endif // CPU_ATTENTION_OPS_H