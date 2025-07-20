#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// 矩阵乘法核函数
__global__ void matrix_multiply_kernel(const float* mat1, const float* mat2, float* result, size_t m, size_t n, size_t k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (size_t p = 0; p < n; ++p) {
            sum += mat1[row * n + p] * mat2[p * k + col];
        }
        result[row * k + col] = sum;
    }
}

// 矩阵乘法函数
std::vector<float> matrix_multiply(const std::vector<float>& mat1, const std::vector<float>& mat2, size_t m, size_t n, size_t k) {
    float *d_mat1, *d_mat2, *d_result;
    std::vector<float> result(m * k, 0.0f);

    cudaMalloc((void**)&d_mat1, mat1.size() * sizeof(float));
    cudaMalloc((void**)&d_mat2, mat2.size() * sizeof(float));
    cudaMalloc((void**)&d_result, result.size() * sizeof(float));

    cudaMemcpy(d_mat1, mat1.data(), mat1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2.data(), mat2.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    matrix_multiply_kernel<<<gridSize, blockSize>>>(d_mat1, d_mat2, d_result, m, n, k);

    cudaMemcpy(result.data(), d_result, result.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);

    return result;
}

// Softmax 核函数
__global__ void softmax_kernel(float* input, size_t n) {
    __shared__ float max_val;
    __shared__ float sum;

    // 计算最大值
    if (threadIdx.x == 0) {
        max_val = input[0];
        for (size_t i = 1; i < n; ++i) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }
    }
    __syncthreads();

    // 计算指数和
    if (threadIdx.x == 0) {
        sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += expf(input[i] - max_val);
        }
    }
    __syncthreads();

    // 计算 Softmax
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        input[i] = expf(input[i] - max_val) / sum;
    }
}

// Softmax 函数
void softmax(std::vector<float>& input) {
    float *d_input;
    cudaMalloc((void**)&d_input, input.size() * sizeof(float));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    softmax_kernel<<<1, 256>>>(d_input, input.size());

    cudaMemcpy(input.data(), d_input, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
}

// GELU 激活函数核函数
__global__ void gelu_kernel(float* input, size_t n) {
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        float x = input[i];
        input[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }
}

// GELU 激活函数
std::vector<float> gelu(const std::vector<float>& input) {
    std::vector<float> output = input;
    float *d_output;
    cudaMalloc((void**)&d_output, output.size() * sizeof(float));
    cudaMemcpy(d_output, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice);

    gelu_kernel<<<1, 256>>>(d_output, output.size());

    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    return output;
}

// Flash Attention 核函数
__global__ void flash_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scale
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (seq_idx >= seq_len) return;

    const float* q_ptr = q + (batch_idx * num_heads * seq_len * head_dim) + (head_idx * seq_len * head_dim) + (seq_idx * head_dim);
    const float* k_ptr = k + (batch_idx * num_heads * seq_len * head_dim) + (head_idx * seq_len * head_dim);
    const float* v_ptr = v + (batch_idx * num_heads * seq_len * head_dim) + (head_idx * seq_len * head_dim);

    float* out_ptr = out + (batch_idx * num_heads * seq_len * head_dim) + (head_idx * seq_len * head_dim) + (seq_idx * head_dim);

    float local_sum[128];
    float local_max = -INFINITY;
    float local_output[128];
    for (int i = 0; i < 128; ++i) {
        local_sum[i] = 0.0f;
        local_output[i] = 0.0f;
    }

    for (int block_start = 0; block_start < seq_len; block_start += 128) {
        int block_end = min(block_start + 128, seq_len);
        int block_size = block_end - block_start;

        float scores[128];
        for (int i = 0; i < block_size; ++i) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_ptr[d] * k_ptr[(block_start + i) * head_dim + d];
            }
            score *= scale;
            scores[i] = score;
            if (score > local_max) {
                local_max = score;
            }
        }

        float local_denom = 0.0f;
        for (int i = 0; i < block_size; ++i) {
            float exp_score = expf(scores[i] - local_max);
            local_denom += exp_score;
            scores[i] = exp_score;
        }

        for (int i = 0; i < block_size; ++i) {
            float attn_weight = scores[i] / local_denom;
            for (int d = 0; d < head_dim; ++d) {
                local_output[d] += attn_weight * v_ptr[(block_start + i) * head_dim + d];
            }
        }
    }

    for (int d = 0; d < head_dim; ++d) {
        out_ptr[d] = local_output[d];
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
    size_t batch_size = 1;

    std::vector<float> Q = matrix_multiply(input, weight_q, seq_len, embedding_dim, embedding_dim);
    std::vector<float> K = matrix_multiply(input, weight_k, seq_len, embedding_dim, embedding_dim);
    std::vector<float> V = matrix_multiply(input, weight_v, seq_len, embedding_dim, embedding_dim);

    std::vector<float> output(seq_len * embedding_dim);

    float *d_q, *d_k, *d_v, *d_out;
    cudaMalloc((void**)&d_q, Q.size() * sizeof(float));
    cudaMalloc((void**)&d_k, K.size() * sizeof(float));
    cudaMalloc((void**)&d_v, V.size() * sizeof(float));
    cudaMalloc((void**)&d_out, output.size() * sizeof(float));

    cudaMemcpy(d_q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);

    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid((seq_len + 128 - 1) / 128, num_heads, batch_size);
    dim3 block(128);

    flash_attention_kernel<<<grid, block>>>(d_q, d_k, d_v, d_out, batch_size, seq_len, num_heads, head_dim, scale);

    cudaMemcpy(output.data(), d_out, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);

    return output;
}
