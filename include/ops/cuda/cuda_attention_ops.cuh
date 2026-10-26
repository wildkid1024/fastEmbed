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
    
    // GQA (Grouped Query Attention) 支持不同的Q/K/V头数
    // 输入格式: [seq_len, num_heads * head_dim] (与multi_head_attention保持一致)
    std::vector<float> grouped_query_attention(
        const std::vector<float>& q_input,       // [seq_len, num_q_heads * head_dim]
        const std::vector<float>& k_input,       // [seq_len, num_kv_heads * head_dim]
        const std::vector<float>& v_input,       // [seq_len, num_kv_heads * head_dim]
        const std::vector<float>& q_norm_weight, // [head_dim], Q的RMS norm权重
        const std::vector<float>& k_norm_weight, // [head_dim], K的RMS norm权重
        size_t seq_len,
        size_t num_q_heads,
        size_t num_kv_heads,
        size_t head_dim
    );
    
    // ========== 批量版本 (Batch-aware) ==========
    
    // 批量 GQA 注意力
    std::vector<float> grouped_query_attention_batch(
        const std::vector<float>& q_input,       // [batch, seq_len, num_q_heads * head_dim]
        const std::vector<float>& k_input,       // [batch, seq_len, num_kv_heads * head_dim]
        const std::vector<float>& v_input,       // [batch, seq_len, num_kv_heads * head_dim]
        const std::vector<float>& q_norm_weight, // [head_dim], Q的RMS norm权重
        const std::vector<float>& k_norm_weight, // [head_dim], K的RMS norm权重
        size_t batch_size,
        size_t seq_len,
        size_t num_q_heads,
        size_t num_kv_heads,
        size_t head_dim
    );

    // ========== 编码器（双向）注意力 ==========
    // 用于 Gemma / EmbeddingGemma 等 encoder-only 模型：
    //   1. Q/K/V 各自按 head 做 RMSNorm（Q/K 带可学习权重，V 无权重，由 normalize_v 控制）
    //   2. 仅对 Q/K 应用 RoPE，rope_theta 可配置（不同层类型可不同）
    //   3. 双向（非因果）softmax，可选的滑动窗口（|i-j| <= sliding_window）
    // 输入布局均为 [seq_len, num_heads * head_dim]
    std::vector<float> encoder_attention(
        const std::vector<float>& q_input,       // [seq_len, num_q_heads * head_dim]
        const std::vector<float>& k_input,       // [seq_len, num_kv_heads * head_dim]
        const std::vector<float>& v_input,       // [seq_len, num_kv_heads * head_dim]
        const std::vector<float>& q_norm_weight, // [head_dim]，为空则跳过 Q 的归一化
        const std::vector<float>& k_norm_weight, // [head_dim]，为空则跳过 K 的归一化
        size_t seq_len,
        size_t num_q_heads,
        size_t num_kv_heads,
        size_t head_dim,
        float rope_theta,        // RoPE base
        float attn_scale,        // 注意力分数缩放（EmbeddingGemma2 用 1.0）
        bool normalize_v,        // 是否对 V 做无权重 RMSNorm
        int sliding_window,      // <=0 表示全注意力；否则 |i-j| <= sliding_window
        float rms_eps            // RMSNorm 的 epsilon
    );
};

#endif // CUDA_ATTENTION_OPS_CUH