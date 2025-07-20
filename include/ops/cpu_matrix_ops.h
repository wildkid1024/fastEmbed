#ifndef CPU_MATRIX_OPS_H
#define CPU_MATRIX_OPS_H

#include "matrix_ops_interface.h"

class CPUMatrixOps : public MatrixOpsInterface {
public:
    std::vector<float> matrix_multiply(const std::vector<float>& mat1, const std::vector<float>& mat2, size_t m, size_t n, size_t k) override;
    void softmax(std::vector<float>& input) override;
};

#endif // CPU_MATRIX_OPS_H
