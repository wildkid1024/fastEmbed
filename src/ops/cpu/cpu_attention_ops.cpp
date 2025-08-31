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

// CPU版本维度重排函数，功能与CUDA permute_kernel完全一致
void cpu_permute(const float* input, float* output, int B, int T, int NH, int HS) {
    // 三重循环遍历batch、sequence、head维度
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                // 计算输入索引: [B, T, NH*HS] 布局
                const int input_idx = b * T * NH * HS + t * NH * HS + h * HS;
                // 计算输出索引: [B, NH, T, HS] 布局
                const int output_idx = b * NH * T * HS + h * T * HS + t * HS;
                // 复制整个head维度的数据
                for (int s = 0; s < HS; ++s) {
                    output[output_idx + s] = input[input_idx + s];
                }
            }
        }
    }
}

static std::vector<float> transpose(const std::vector<float>& input, int rows, int cols) {
        std::vector<float> output(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output[j * rows + i] = input[i * cols + j];
            }
        }
        return output;
}

std::vector<float> CPUAttentionOps::multi_head_attentionv2(
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
    const size_t batch_size = 1;  // 保持与现有实现一致
    const size_t seq_len = input.size() / embedding_dim / batch_size;
    const size_t head_dim = embedding_dim / num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // 1. 计算QKV矩阵 (batch_size, seq_len, embedding_dim)
    std::vector<float> Q = matrix_ops.matrix_multiply_transpose(input, weight_q, seq_len, embedding_dim, embedding_dim);
    std::vector<float> K = matrix_ops.matrix_multiply_transpose(input, weight_k, seq_len, embedding_dim, embedding_dim);
    std::vector<float> V = matrix_ops.matrix_multiply_transpose(input, weight_v, seq_len, embedding_dim, embedding_dim);

    // 添加偏置
    for (size_t i = 0; i < Q.size(); ++i) Q[i] += query_bias[i % embedding_dim];
    for (size_t i = 0; i < K.size(); ++i) K[i] += key_bias[i % embedding_dim];
    for (size_t i = 0; i < V.size(); ++i) V[i] += value_bias[i % embedding_dim];

    // 2. 维度重排: [B, T, NH*HS] -> [B, NH, T, HS]
    std::vector<float> Q_permuted(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> K_permuted(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> V_permuted(batch_size * num_heads * seq_len * head_dim);
    cpu_permute(Q.data(), Q_permuted.data(), batch_size, seq_len, num_heads, head_dim);
    cpu_permute(K.data(), K_permuted.data(), batch_size, seq_len, num_heads, head_dim);
    cpu_permute(V.data(), V_permuted.data(), batch_size, seq_len, num_heads, head_dim);

    // 3. 批量矩阵乘法计算注意力分数 (Q*K^T)
    const size_t M = seq_len;          // Q的行数
    const size_t N = seq_len;          // K的列数
    const size_t K_dim = head_dim;     // Q的列数 = K的行数
    std::vector<float> scores(batch_size * num_heads * seq_len * head_dim);

    // 对每个batch和head执行Q*K^T
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            // Calculate offset for current head
            size_t head_offset = h * head_dim * seq_len;
            
            // Extract Q head vector from permuted Q matrix
            std::vector<float> Q_head_vector(Q_permuted.begin() + head_offset, 
                                            Q_permuted.begin() + head_offset + head_dim * seq_len);
            
            // Extract K head vector and transpose
            std::vector<float> K_head_vector(K_permuted.begin() + head_offset, 
                                            K_permuted.begin() + head_offset + head_dim * seq_len);
            
            std::vector<float> V_head_vector(V_permuted.begin() + head_offset, 
                                            V_permuted.begin() + head_offset + head_dim * seq_len);
            
            // Matrix multiplication Q*K^T
            std::vector<float> attention_scores = matrix_ops.matrix_multiply_transpose(Q_head_vector, K_head_vector, M, K_dim, N);

            // 缩放注意力分数
            for (size_t i = 0; i < attention_scores.size(); ++i)
                attention_scores[i] *= scale;

            // 应用 softmax
            attention_scores = matrix_ops.softmax(attention_scores, 1, seq_len); 

            // 4. 批量矩阵乘法计算输出 (Attention * V)
            std::vector<float> attn_output = matrix_ops.matrix_multiply(attention_scores, V_head_vector, seq_len, seq_len, head_dim);
            
            // 将结果存入scores矩阵的对应位置
            int offset = b * num_heads * seq_len * head_dim + h * seq_len * head_dim;
            std::copy(attn_output.begin(), attn_output.end(), 
                     scores.begin() + offset);

        }
    }
    
    // 5. 重排回原始维度并合并heads
    std::vector<float> output(batch_size * seq_len * embedding_dim);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t src_offset = (b * num_heads + h) * seq_len * head_dim + t * head_dim;
                const size_t dst_offset = (b * seq_len + t) * embedding_dim + h * head_dim;
                std::copy(
                    &scores[src_offset],
                    &scores[src_offset] + head_dim,
                    &output[dst_offset]
                );
            }
        }
    }
    
    return output;
}
