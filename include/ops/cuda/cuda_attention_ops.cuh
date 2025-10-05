#ifndef CUDA_ATTENTION_OPS_CUH
#define CUDA_ATTENTION_OPS_CUH

#include <vector>

class CUDAAttentionOps {
public:
    CUDAAttentionOps();
    ~CUDAAttentionOps();
    
    std::vector<float> multi_head_attention(
        const std::vector<float>& input,          // [batch, seq_len, embed_dim]
        const std::vector<float>& weight_q,       // [embed_dim, embed_dim]
        const std::vector<float>& weight_k,       // [embed_dim, embed_dim]
        const std::vector<float>& weight_v,       // [embed_dim, embed_dim]
        const std::vector<float>& query_bias,     // [embed_dim]
        const std::vector<float>& key_bias,       // [embed_dim]
        const std::vector<float>& value_bias,     // [embed_dim]
        size_t num_heads,                         // Number of attention heads
        size_t embedding_dim                      // Embedding dimension per token
    );
};

#endif // CUDA_ATTENTION_OPS_CUH