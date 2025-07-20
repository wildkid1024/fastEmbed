#include "cuda_matrix_ops.h"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_matrix_ops_kernel.cu"

std::vector<float> CUDAMatrixOps::matrix_multiply(const std::vector<float>& mat1, const std::vector<float>& mat2, size_t m, size_t n, size_t k) {
    float *d_mat1, *d_mat2, *d_result;
    std::vector<float> result(m * k, 0.0f);

    // 分配设备内存
    cudaMalloc((void**)&d_mat1, mat1.size() * sizeof(float));
    cudaMalloc((void**)&d_mat2, mat2.size() * sizeof(float));
    cudaMalloc((void**)&d_result, result.size() * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_mat1, mat1.data(), mat1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2.data(), mat2.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 设置网格和线程块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // 调用 CUDA 核函数
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_mat1, d_mat2, d_result, m, n, k);

    // 将结果从设备复制到主机
    cudaMemcpy(result.data(), d_result, result.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);

    return result;
}


void CUDAMatrixOps::softmax(std::vector<float>& input) {
    float *d_input;
    cudaMalloc((void**)&d_input, input.size() * sizeof(float));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    // compute_softmax(d_input, input.size());
    dim3 block(256);
    dim3 grid((num_cols + block.x - 1) / block.x, num_rows);
    softmax_kernel<<<grid, block>>>(input, num_rows, num_cols);
    cudaDeviceSynchronize();

    cudaMemcpy(input.data(), d_input, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
}
