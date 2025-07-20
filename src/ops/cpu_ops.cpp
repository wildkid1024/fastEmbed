#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// 矩阵乘法函数
std::vector<float> matrix_multiply(const std::vector<float>& mat1, const std::vector<float>& mat2, size_t m, size_t n, size_t k) {
    std::vector<float> result(m * k, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t p = 0; p < n; ++p) {
                result[i * k + j] += mat1[i * n + p] * mat2[p * k + j];
            }
        }
    }
    return result;
}

// Softmax 函数
void softmax(std::vector<float>& input) {
    float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    for (auto& val : input) {
        val = std::exp(val - max_val);
        sum += val;
    }
    for (auto& val : input) {
        val /= sum;
    }
}

// GELU 激活函数
float gelu(float x) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
}

// GELU 激活函数
std::vector<float> gelu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = gelu(input[i]);
    }
    return output;
}

// 提取每个头的 Q, K, V 矩阵
void extract_heads(const std::vector<float>& src, std::vector<float>& dst, size_t seq_len, size_t embedding_dim, size_t head_dim, size_t h) {
    for (size_t i = 0; i < seq_len; ++i) {
        std::copy(src.begin() + i * embedding_dim + h * head_dim, 
                src.begin() + i * embedding_dim + (h + 1) * head_dim,
                dst.begin() + i * head_dim);
    }
}

// 多头注意力函数
std::vector<float> multi_head_attention(
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
    std::vector<float> Q = matrix_multiply(input, weight_q, seq_len, embedding_dim, embedding_dim);
    std::vector<float> K = matrix_multiply(input, weight_k, seq_len, embedding_dim, embedding_dim);
    std::vector<float> V = matrix_multiply(input, weight_v, seq_len, embedding_dim, embedding_dim);

    std::vector<float> output(seq_len * embedding_dim, 0.0f);

    // 遍历每个头
    for (size_t h = 0; h < num_heads; ++h) {
        std::vector<float> Q_head(seq_len * head_dim);
        std::vector<float> K_head(seq_len * head_dim);
        std::vector<float> V_head(seq_len * head_dim);

        // 提取每个头的 Q, K, V
        extract_heads(Q, Q_head, seq_len, embedding_dim, head_dim, h);
        extract_heads(K, K_head, seq_len, embedding_dim, head_dim, h);
        extract_heads(V, V_head, seq_len, embedding_dim, head_dim, h);

        // 计算注意力分数
        std::vector<float> scores = matrix_multiply(Q_head, K_head, seq_len, head_dim, seq_len);
        // 缩放
        float scale = std::sqrt(static_cast<float>(head_dim));
        for (auto& score : scores) {
            score /= scale;
        }

        // Softmax 处理
        for (size_t i = 0; i < seq_len; ++i) {
            std::vector<float> score_chunk(scores.begin() + i * seq_len, scores.begin() + (i + 1) * seq_len);
            softmax(score_chunk);
            std::copy(score_chunk.begin(), score_chunk.end(), scores.begin() + i * seq_len);
        }

        // 应用注意力分数
        std::vector<float> attn_output = matrix_multiply(scores, V_head, seq_len, seq_len, head_dim);

        // 拼接多头结果
        for (size_t i = 0; i < seq_len; ++i) {
            std::copy(attn_output.begin() + i * head_dim, 
                    attn_output.begin() + (i + 1) * head_dim,
                    output.begin() + i * embedding_dim + h * head_dim);
        }
    }

    return output;
}
