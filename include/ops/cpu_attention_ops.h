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
        size_t num_heads,
        size_t embedding_dim
    ) override;

private:
    CPUMatrixOps matrix_ops;
};

#endif // CPU_ATTENTION_OPS_H
