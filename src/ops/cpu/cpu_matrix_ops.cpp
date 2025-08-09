#include "ops/cpu/cpu_matrix_ops.h"
#include <cmath>
#include <algorithm>
#include <stdexcept> // Add this line to include the stdexcept header


std::vector<float> CPUMatrixOps::matrix_multiply_transpose(const std::vector<float>& a, const std::vector<float>& b, 
                                                 size_t m, size_t n, size_t k) {
    std::vector<float> result(m * k, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t p = 0; p < n; ++p) {
                result[i * k + j] += a[i * n + p] * b[j * n + p];
            }
        }
    }
    return result;
}

// 矩阵乘法实现
std::vector<float> CPUMatrixOps::matrix_multiply(const std::vector<float>& a, const std::vector<float>& b, 
                                                 size_t m, size_t n, size_t k) {
    std::vector<float> result(m * k, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t p = 0; p < n; ++p) {
                result[i * k + j] += a[i * n + p] * b[p * k + j];
            }
        }
    }
    return result;
}

// GELU 激活函数实现
std::vector<float> CPUMatrixOps::gelu(const std::vector<float>& input) {
    std::vector<float> output = input;
    const float sqrt_2_over_pi = std::sqrt(2.0f / 3.14159265358979323846f);
    for (size_t i = 0; i < output.size(); ++i) {
        float x = output[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    }
    return output;
}

// 层归一化实现
std::vector<float> CPUMatrixOps::layer_norm(const std::vector<float>& input, 
                                            const std::vector<float>& gamma, 
                                            const std::vector<float>& beta, 
                                            size_t embedding_dim, 
                                            float epsilon) {
    std::vector<float> output = input;
    for (size_t pos = 0; pos < output.size() / embedding_dim; ++pos) {
        float mean = 0.0f;
        float variance = 0.0f;
        size_t start = pos * embedding_dim;
        // 计算均值
        for (size_t i = 0; i < embedding_dim; ++i) {
            mean += output[start + i];
        }
        mean /= embedding_dim;
        // 计算方差
        for (size_t i = 0; i < embedding_dim; ++i) {
            float diff = output[start + i] - mean;
            variance += diff * diff;
        }
        variance /= embedding_dim;
        float std_dev = std::sqrt(variance + epsilon);
        // 应用 LayerNorm
        for (size_t i = 0; i < embedding_dim; ++i) {
            output[start + i] = (output[start + i] - mean) / std_dev * gamma[i] + beta[i];
        }
    }
    return output;
}

// Softmax 实现
std::vector<float> CPUMatrixOps::softmax(const std::vector<float>& input, size_t axis, size_t dim_size) {
    if (axis == 0) {
        // 沿第 0 维计算 Softmax（向量）
        std::vector<float> output = input;
        float max_val = *std::max_element(input.begin(), input.end());
        float sum = 0.0f;

        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }

        for (size_t i = 0; i < output.size(); ++i) {
            output[i] /= sum;
        }

        return output;
    } else if (axis == 1) {
        // 使用传入的dim_size，不再硬编码768
        size_t actual_dim = (dim_size > 0) ? dim_size : 768; // 兼容旧调用
        size_t num_rows = input.size() / actual_dim;
        std::vector<float> output = input;

        for (size_t i = 0; i < num_rows; ++i) {
            float max_val = *std::max_element(input.begin() + i * actual_dim, input.begin() + (i + 1) * actual_dim);
            float sum = 0.0f;

            for (size_t j = 0; j < actual_dim; ++j) {
                size_t idx = i * actual_dim + j;
                output[idx] = std::exp(input[idx] - max_val);
                sum += output[idx];
            }

            for (size_t j = 0; j < actual_dim; ++j) {
                size_t idx = i * actual_dim + j;
                output[idx] /= sum;
            }
        }

        return output;
    } else {
        throw std::invalid_argument("Unsupported axis for softmax");
    }
}