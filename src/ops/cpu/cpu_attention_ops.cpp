#include "ops/cpu/cpu_attention_ops.h"
#include <algorithm>
#include <cmath>
#include <iostream>

CPUAttentionOps::CPUAttentionOps() = default;

std::vector<float> CPUAttentionOps::multi_head_attention(
    const std::vector<float>& input,
    const std::vector<float>& weight_q,
    const std::vector<float>& weight_k,
    const std::vector<float>& weight_v,
    const std::vector<float>& query_bias,
    const std::vector<float>& key_bias,
    const std::vector<float>& value_bias,
    size_t num_heads,
    size_t embedding_dim
) {
    // 计算输入序列长度（input形状: [seq_len, embedding_dim]）
    size_t seq_len = input.size() / embedding_dim;
    size_t head_dim = embedding_dim / num_heads;

    // 计算Q、K、V矩阵
    std::vector<float> Q = matrix_ops.matrix_multiply_transpose(input, weight_q, seq_len, embedding_dim, embedding_dim);
    std::vector<float> K = matrix_ops.matrix_multiply_transpose(input, weight_k, seq_len, embedding_dim, embedding_dim);
    std::vector<float> V = matrix_ops.matrix_multiply_transpose(input, weight_v, seq_len, embedding_dim, embedding_dim);

    // 添加偏置
    for (size_t i = 0; i < Q.size(); ++i) Q[i] += query_bias[i % embedding_dim];
    for (size_t i = 0; i < K.size(); ++i) K[i] += key_bias[i % embedding_dim];
    for (size_t i = 0; i < V.size(); ++i) V[i] += value_bias[i % embedding_dim];

    // 计算注意力分数 (num_heads, seq_len, seq_len)
    std::vector<float> scores(num_heads * seq_len * seq_len);
    float scale = std::sqrt(static_cast<float>(head_dim));

    // 应用注意力分数到V的输出
    std::vector<float> attn_output(num_heads * seq_len * head_dim);

    for (size_t h = 0; h < num_heads; ++h) {
        // 为当前头创建Q_head、K_head、V_head的临时存储
        std::vector<float> Q_head(seq_len * head_dim);
        std::vector<float> K_head(head_dim * seq_len);
        std::vector<float> V_head(seq_len * head_dim);

        // 直接从原始Q、K、V矩阵提取当前头的参数
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t d = 0; d < head_dim; ++d) {
                size_t q_idx = i * embedding_dim + h * head_dim + d;
                Q_head[i * head_dim + d] = Q[q_idx];

                size_t k_idx = i * embedding_dim + h * head_dim + d;
                K_head[d * seq_len + i] = K[k_idx];

                size_t v_idx = i * embedding_dim + h * head_dim + d;
                V_head[i * head_dim + d] = V[v_idx];
            }
        }

        // 计算注意力分数 Q[seq_len, head_dim] × K^T[head_dim, seq_len] = scores[seq_len, seq_len]
        std::vector<float> temp = matrix_ops.matrix_multiply(
            Q_head, K_head, seq_len, head_dim, seq_len
        );

        // 缩放分数并应用softmax
        float* score_head = &scores[h * seq_len * seq_len];
        std::copy(temp.begin(), temp.end(), score_head);
        for (size_t i = 0; i < seq_len * seq_len; ++i) {
            score_head[i] /= scale;
        }

        // Softmax处理
        std::vector<float> score_matrix(score_head, score_head + seq_len * seq_len);
        std::vector<float> softmax_result = matrix_ops.softmax(score_matrix, 1, seq_len);
        std::copy(softmax_result.begin(), softmax_result.end(), score_head);

        // 应用注意力分数到V
        float* output_head = &attn_output[h * seq_len * head_dim];
        std::vector<float> temp_output = matrix_ops.matrix_multiply(
            std::vector<float>(score_head, score_head + seq_len * seq_len),
            V_head,
            seq_len, seq_len, head_dim
        );
        std::copy(temp_output.begin(), temp_output.end(), output_head);
    }

    // 重塑回[seq_len, embedding_dim]
    std::vector<float> output(seq_len * embedding_dim, 0.0f);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t h = 0; h < num_heads; ++h) {
            const float* head_data = &attn_output[h * seq_len * head_dim + i * head_dim];
            std::copy(head_data, head_data + head_dim,
                      output.begin() + i * embedding_dim + h * head_dim);
        }
    }

    return output;
}