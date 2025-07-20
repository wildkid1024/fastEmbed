#include "cpu_matrix_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

std::vector<float> CPUMatrixOps::matrix_multiply(const std::vector<float>& mat1, const std::vector<float>& mat2, size_t m, size_t n, size_t k) {
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

void CPUMatrixOps::softmax(std::vector<float>& input) {
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
