#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include "ops/cpu/cpu_matrix_ops.h"
#include "ops/cuda/cuda_matrix_ops.h"

// 测试参数配置
const float TOLERANCE = 1e-5f;  // 浮点比较容差
const int RANDOM_SEED = 42;     // 随机种子，保证测试可复现
const std::vector<std::tuple<int, int, int>> MATRIX_SIZES = {
    {2, 2, 2},   // 小型方阵
    {4, 5, 6},   // 非方阵
    {16, 16, 16}, // 中型方阵
    {1, 1, 1},   // 标量情况
    {100, 200, 300} // 大型矩阵
};

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

class MatrixOpsTest : public ::testing::Test {
protected:
    CPUMatrixOps cpu_ops;
    CUDAMatrixOps cuda_ops;
};

// 测试矩阵乘法
TEST_F(MatrixOpsTest, MatrixMultiplication) {
    for (const auto& [m, n, k] : MATRIX_SIZES) {
        SCOPED_TRACE("矩阵尺寸: " + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k));

        auto a = generate_random_matrix(m * n);
        auto b = generate_random_matrix(n * k);

        auto cpu_result = cpu_ops.matrix_multiply(a, b, m, n, k);
        auto cuda_result = cuda_ops.matrix_multiply(a, b, m, n, k);

        compare_matrices(cpu_result, cuda_result);
    }
}

// 测试激活函数
TEST_F(MatrixOpsTest, ActivationFunctions) {
    const std::vector<size_t> SIZES = {1, 10, 1000, 1024*1024};
    const std::vector<float> TEST_VALUES = {-2.5f, -1.0f, 0.0f, 0.5f, 1.0f, 3.14f};

    // 测试ReLU
    for (size_t size : SIZES) {
        SCOPED_TRACE("ReLU测试，尺寸: " + std::to_string(size));
        auto input = generate_random_matrix(size);
        auto cpu_result = cpu_ops.relu(input);
        auto cuda_result = cuda_ops.relu(input);
        compare_matrices(cpu_result, cuda_result);
    }

    // 测试GELU
    for (float val : TEST_VALUES) {
        SCOPED_TRACE("GELU测试值: " + std::to_string(val));
        std::vector<float> input = {val};
        auto cpu_result = cpu_ops.gelu(input);
        auto cuda_result = cuda_ops.gelu(input);
        compare_matrices(cpu_result, cuda_result);
    }
}

// 测试层归一化
TEST_F(MatrixOpsTest, LayerNorm) {
    const std::vector<size_t> EMBEDDING_DIMS = {16, 32, 64, 128, 256, 1024};
    const std::vector<float> EPS_VALUES = {1e-5f, 1e-10f};

    for (size_t dim : EMBEDDING_DIMS) {
        for (float eps : EPS_VALUES) {
            SCOPED_TRACE("嵌入维度: " + std::to_string(dim) + ", EPS: " + std::to_string(eps));
            auto input = generate_random_matrix(10 * dim);  // 10个样本
            auto gamma = generate_random_matrix(dim);
            auto beta = generate_random_matrix(dim);

            auto cpu_result = cpu_ops.layer_norm(input, gamma, beta, dim, eps);
            auto cuda_result = cuda_ops.layer_norm(input, gamma, beta, dim, eps);
            compare_matrices(cpu_result, cuda_result);
        }
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}