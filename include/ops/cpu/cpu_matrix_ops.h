#pragma once
#include <vector>

class CPUMatrixOps {
public:
    // 矩阵乘法方法
    std::vector<float> matrix_multiply(const std::vector<float>& a, const std::vector<float>& b, 
                                       size_t m, size_t n, size_t k);

    std::vector<float> matrix_multiply_transpose(const std::vector<float>& a, const std::vector<float>& b, 
                                       size_t m, size_t n, size_t k);
    // GELU 激活函数方法
    std::vector<float> gelu(const std::vector<float>& input);
    // 添加ReLU和Swish函数声明
    std::vector<float> relu(const std::vector<float>& input);
    std::vector<float> swish(const std::vector<float>& input);
    // 层归一化方法
    std::vector<float> layer_norm(const std::vector<float>& input, 
                                  const std::vector<float>& gamma, 
                                  const std::vector<float>& beta, 
                                  size_t embedding_dim, 
                                  float epsilon = 1e-5);
    // Softmax 方法声明 - 添加维度参数
    std::vector<float> softmax(const std::vector<float>& input, size_t axis = 1, size_t dim_size = 0);
};