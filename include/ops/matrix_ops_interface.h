#ifndef MATRIX_OPS_INTERFACE_H
#define MATRIX_OPS_INTERFACE_H

#include <vector>

class MatrixOpsInterface {
public:
    virtual ~MatrixOpsInterface() = default;
    // 矩阵乘法操作接口
    virtual std::vector<float> matrix_multiply(
        const std::vector<float>& mat1,
        const std::vector<float>& mat2,
        size_t m,
        size_t n,
        size_t k
    ) = 0;

    // Add softmax virtual function declaration
    virtual void softmax(std::vector<float>& input) = 0;
};

#endif // MATRIX_OPS_INTERFACE_H
