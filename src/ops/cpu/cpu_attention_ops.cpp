#include "ops/cpu/cpu_attention_ops.h"
#include <algorithm>
#include <cmath>

CPUAttentionOps::CPUAttentionOps() = default;

std::vector<float> CPUAttentionOps::multi_head_attention(
    const std::vector<float>& input,
    const std::vector<float>& weight_q,
    const std::vector<float>& weight_k,
    const std::vector<float>& weight_v,
    size_t num_heads,
    size_t embedding_dim
) {
    size_t seq_len = input.size() / embedding_dim;
    size_t head_dim = embedding_dim / num_heads;

    // 计算 Q, K, V 矩阵
    std::vector<float> Q = matrix_ops.matrix_multiply(input, weight_q, seq_len, embedding_dim, embedding_dim);
    std::vector<float> K = matrix_ops.matrix_multiply(input, weight_k, seq_len, embedding_dim, embedding_dim);
    std::vector<float> V = matrix_ops.matrix_multiply(input, weight_v, seq_len, embedding_dim, embedding_dim);

    std::vector<float> output(seq_len * embedding_dim, 0.0f);

    // 遍历每个头
    for (size_t h = 0; h < num_heads; ++h) {
        std::vector<float> Q_head(seq_len * head_dim);
        std::vector<float> K_head(seq_len * head_dim);
        std::vector<float> V_head(seq_len * head_dim);

        // 提取每个头的 Q, K, V
        for (size_t i = 0; i < seq_len; ++i) {
            std::copy(Q.begin() + i * embedding_dim + h * head_dim, 
                    Q.begin() + i * embedding_dim + (h + 1) * head_dim,
                    Q_head.begin() + i * head_dim);
            std::copy(K.begin() + i * embedding_dim + h * head_dim, 
                    K.begin() + i * embedding_dim + (h + 1) * head_dim,
                    K_head.begin() + i * head_dim);
            std::copy(V.begin() + i * embedding_dim + h * head_dim, 
                    V.begin() + i * embedding_dim + (h + 1) * head_dim,
                    V_head.begin() + i * head_dim);
        }

        // 计算注意力分数
        std::vector<float> scores = matrix_ops.matrix_multiply(Q_head, K_head, seq_len, head_dim, seq_len);
        // 缩放
        float scale = std::sqrt(static_cast<float>(head_dim));
        for (auto& score : scores) {
            score /= scale;
        }

        // Softmax 处理
        for (size_t i = 0; i < seq_len; ++i) {
            std::vector<float> score_chunk(scores.begin() + i * seq_len, scores.begin() + (i + 1) * seq_len);
            matrix_ops.softmax(score_chunk);
            std::copy(score_chunk.begin(), score_chunk.end(), scores.begin() + i * seq_len);
        }

        // 应用注意力分数
        std::vector<float> attn_output = matrix_ops.matrix_multiply(scores, V_head, seq_len, seq_len, head_dim);

        // 拼接多头结果
        for (size_t i = 0; i < seq_len; ++i) {
            std::copy(attn_output.begin() + i * head_dim, 
                    attn_output.begin() + (i + 1) * head_dim,
                    output.begin() + i * embedding_dim + h * head_dim);
        }
    }

    return output;
}
