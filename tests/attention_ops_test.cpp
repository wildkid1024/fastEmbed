#include <vector>
#include <random>
#include <cmath>
#include "ops/cpu/cpu_attention_ops.h"
#include "ops/cuda/cuda_attention_cublas_ops.cuh"
#include <gtest/gtest.h>

// 测试参数配置
const float TOLERANCE = 1e-3f;  // 浮点比较容差
const int RANDOM_SEED = 42;     // 随机种子，保证测试可复现

// 生成三维随机张量 (batch_size, seq_len, dim)
std::vector<float> generate_random_tensor(size_t seq_len, size_t dim) {
    std::vector<float> tensor(seq_len * dim);
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : tensor) val = dist(gen);
    return tensor;
}

// 生成随机矩阵
std::vector<float> generate_random_matrix(size_t size) {
    std::vector<float> mat(size);
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : mat) val = dist(gen);
    return mat;
}

// 比较两个矩阵是否在容差范围内相等
void compare_matrices(const std::vector<float>& cpu_result, const std::vector<float>& cuda_result) {
    ASSERT_EQ(cpu_result.size(), cuda_result.size()) << "矩阵尺寸不匹配";
    for (size_t i = 0; i < cpu_result.size(); ++i) {
        EXPECT_NEAR(cpu_result[i], cuda_result[i], TOLERANCE) << 
            "元素不匹配 at index: " << i << 
            " CPU: " << cpu_result[i] << 
            " CUDA: " << cuda_result[i];
    }
}

// 验证注意力权重每行之和是否为1.0
void verify_attention_weights_sum(const std::vector<float>& weights, size_t batch_size, size_t seq_len) {
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < seq_len; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < seq_len; ++j) {
                sum += weights[b * seq_len * seq_len + i * seq_len + j];
            }
            EXPECT_NEAR(sum, 1.0f, TOLERANCE) << "注意力权重之和不为1.0";
        }
    }
}

class AttentionOpsTest : public ::testing::Test {
protected:
    CPUAttentionOps cpu_ops;
    CUDAAttentionOps cuda_ops;
};

// 测试多头注意力
TEST_F(AttentionOpsTest, MultiHeadAttention) {
    const std::vector<std::tuple<size_t, size_t, size_t>> PARAMS = {
        {10, 64, 4},   // seq_len=10, embed_dim=64, heads=4
        {10, 128, 8},  // seq_len=16, embed_dim=128, heads=8
    };

    for (const auto& [seq_len, embed_dim, num_heads] : PARAMS) {
        SCOPED_TRACE("seq_len: " + std::to_string(seq_len) + ", embed_dim: " + std::to_string(embed_dim) + ", heads: " + std::to_string(num_heads));

        // 生成输入和权重矩阵
        auto input = generate_random_tensor(seq_len, embed_dim);  // batch_size固定为1
        auto wq = generate_random_matrix(embed_dim * embed_dim);
        auto wk = generate_random_matrix(embed_dim * embed_dim);
        auto wv = generate_random_matrix(embed_dim * embed_dim);
        
        // 生成偏置向量
        auto query_bias = generate_random_matrix(embed_dim);
        auto key_bias = generate_random_matrix(embed_dim);
        auto value_bias = generate_random_matrix(embed_dim);

        // 计算多头注意力 (移除了output_weight参数)
        auto cpu_output = cpu_ops.multi_head_attention(
            input, wq, wk, wv, query_bias, key_bias, value_bias, num_heads, embed_dim
        );
        auto cuda_output = cuda_ops.multi_head_attention(
            input, wq, wk, wv, query_bias, key_bias, value_bias, num_heads, embed_dim
        );

        // 比较结果
        compare_matrices(cpu_output, cuda_output);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}