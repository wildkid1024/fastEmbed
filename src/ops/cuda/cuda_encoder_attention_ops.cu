#include "ops/cuda/cuda_attention_ops.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>

// 编码器（双向）注意力实现，用于 Gemma / EmbeddingGemma 等 encoder-only 模型。
// 与 grouped_query_attention 的区别：
//   1. 不做因果掩码（双向可见）
//   2. RoPE base 可配置（Gemma 的 full/sliding 层使用不同的 theta）
//   3. 支持对 V 做无权重 RMSNorm
//   4. 支持可选滑动窗口（|i-j| <= window）

#define CHECK_CUDA_ERROR_ENC(call)  \
    do { \
        cudaError_t err = call;    \
        if (err != cudaSuccess) {  \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +  \
                                     std::to_string(__LINE__) + ": " + cudaGetErrorString(err));  \
        } \
    } while (0)

namespace {

#define ENC_BLOCK_SIZE 256

// 每个 head 独立做 RMSNorm（in-place）：data [seq_len, num_heads * head_dim]
__global__ void encoder_rms_norm_head_kernel(
    float* data,
    const float* weight,   // [head_dim]，has_weight=0 时忽略
    int seq_len,
    int num_heads,
    int head_dim,
    float eps,
    int has_weight
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * num_heads;
    if (idx >= total) return;

    const int pos = idx / num_heads;
    const int head = idx % num_heads;
    const long base = static_cast<long>(pos) * num_heads * head_dim +
                      static_cast<long>(head) * head_dim;

    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        const float val = data[base + d];
        sum_sq += val * val;
    }
    const float scale = 1.0f / sqrtf(sum_sq / static_cast<float>(head_dim) + eps);

    if (has_weight) {
        for (int d = 0; d < head_dim; ++d) {
            data[base + d] = data[base + d] * scale * weight[d];
        }
    } else {
        for (int d = 0; d < head_dim; ++d) {
            data[base + d] *= scale;
        }
    }
}

// RoPE（in-place，仅用于 Q/K）：data [seq_len, num_heads * head_dim]
__global__ void encoder_rope_kernel(
    float* data,
    int seq_len,
    int num_heads,
    int head_dim,
    float rope_theta
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * num_heads;
    if (idx >= total) return;

    const int pos = idx / num_heads;
    const int head = idx % num_heads;
    const int half_dim = head_dim / 2;
    const long base = static_cast<long>(pos) * num_heads * head_dim +
                      static_cast<long>(head) * head_dim;

    for (int d = 0; d < half_dim; ++d) {
        // inv_freq[d] = 1 / theta^(2d/head_dim)
        const float inv_freq = 1.0f / powf(rope_theta, 2.0f * static_cast<float>(d) / static_cast<float>(head_dim));
        const float theta = static_cast<float>(pos) * inv_freq;
        const float cos_val = cosf(theta);
        const float sin_val = sinf(theta);

        const float x0 = data[base + d];
        const float x1 = data[base + d + half_dim];
        data[base + d] = x0 * cos_val - x1 * sin_val;
        data[base + d + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// scores[head][i][j] = scale * dot(q[i][head], k[j][kv_head])
// blockIdx.x = 查询位置 i，blockIdx.y = 查询 head
__global__ void encoder_qk_kernel(
    const float* q,      // [seq_len, num_q_heads * head_dim]
    const float* k,      // [seq_len, num_kv_heads * head_dim]
    float* scores,       // [num_q_heads, seq_len, seq_len]
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int window           // <=0 表示全注意力
) {
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    if (i >= seq_len || h >= num_q_heads) return;

    const int group = num_q_heads / num_kv_heads;
    const int kv_head = h / group;   // GQA 映射

    __shared__ float reduce_buf[ENC_BLOCK_SIZE];

    float* s_row = scores + (static_cast<long>(h) * seq_len + i) * seq_len;
    const float* q_row = q + (static_cast<long>(i) * num_q_heads + h) * head_dim;

    // 先全部置为 -inf（窗口外的位置不参与注意力）
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        s_row[j] = -1e30f;
    }
    __syncthreads();

    int j_start = 0;
    int j_end = seq_len;
    if (window > 0) {
        j_start = i - window > 0 ? i - window : 0;
        j_end = i + window + 1 < seq_len ? i + window + 1 : seq_len;
    }

    for (int j = j_start; j < j_end; ++j) {
        const float* k_row = k + (static_cast<long>(j) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q_row[d] * k_row[d];
        }

        reduce_buf[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            s_row[j] = reduce_buf[0] * scale;
        }
        __syncthreads();
    }
}

// 行级 softmax（in-place）：scores [num_q_heads * seq_len, seq_len]
__global__ void encoder_softmax_kernel(float* scores, int seq_len) {
    const int row = blockIdx.x;
    float* s_row = scores + static_cast<long>(row) * seq_len;

    __shared__ float reduce_buf[ENC_BLOCK_SIZE];

    float max_val = -1e30f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        max_val = fmaxf(max_val, s_row[j]);
    }
    reduce_buf[threadIdx.x] = max_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_buf[threadIdx.x] = fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = reduce_buf[0];
    __syncthreads();

    float sum_val = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        const float e = expf(s_row[j] - row_max);
        s_row[j] = e;
        sum_val += e;
    }
    reduce_buf[threadIdx.x] = sum_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float total = reduce_buf[0];
    const float inv_total = total > 0.0f ? 1.0f / total : 0.0f;

    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        s_row[j] *= inv_total;
    }
}

// 注意力输出：out[i][h][d] = sum_j probs[h][i][j] * v[j][kv_head][d]
// blockIdx.x = 查询位置 i，blockIdx.y = 查询 head
__global__ void encoder_av_kernel(
    const float* probs,   // [num_q_heads, seq_len, seq_len]
    const float* v,       // [seq_len, num_kv_heads * head_dim]
    float* out,           // [seq_len, num_q_heads * head_dim]
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    if (i >= seq_len || h >= num_q_heads) return;

    const int group = num_q_heads / num_kv_heads;
    const int kv_head = h / group;

    const float* p_row = probs + (static_cast<long>(h) * seq_len + i) * seq_len;
    const int out_offset = (i * num_q_heads + h) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            acc += p_row[j] * v[(static_cast<long>(j) * num_kv_heads + kv_head) * head_dim + d];
        }
        out[out_offset + d] = acc;
    }
}

// 视觉塔的轴向 2D RoPE（in-place，仅用于 Q/K）：data [num_patches, num_heads * head_dim]
// 频率按 head_dim/2 计算（每半 head_dim/4 个频率），前一半通道用 x 坐标、后一半用 y 坐标。
__global__ void vision_axial_rope_kernel(
    float* data,
    const int* positions,   // [num_patches * 2] (x, y)
    int num_patches,
    int num_heads,
    int head_dim,
    float rope_theta
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_patches * num_heads;
    if (idx >= total) return;

    const int pos = idx / num_heads;
    const int head = idx % num_heads;

    const int quarter = head_dim / 4;     // 每半区内的频率个数
    if (quarter <= 0) return;
    const int spatial_dim = head_dim / 2;  // 频率分母的维度
    const int part = head_dim / 2;         // 每半区的通道数

    const long base = static_cast<long>(pos) * num_heads * head_dim +
                      static_cast<long>(head) * head_dim;
    const float x_pos = static_cast<float>(positions[pos * 2]);
    const float y_pos = static_cast<float>(positions[pos * 2 + 1]);

    for (int i = 0; i < quarter; ++i) {
        const float inv_freq = 1.0f / powf(rope_theta, 2.0f * static_cast<float>(i) / static_cast<float>(spatial_dim));

        // 半区 0：使用 x 坐标
        const float theta_x = x_pos * inv_freq;
        const float cos_x = cosf(theta_x);
        const float sin_x = sinf(theta_x);
        const float x0 = data[base + i];
        const float x1 = data[base + i + quarter];
        data[base + i] = x0 * cos_x - x1 * sin_x;
        data[base + i + quarter] = x0 * sin_x + x1 * cos_x;

        // 半区 1：使用 y 坐标
        const float theta_y = y_pos * inv_freq;
        const float cos_y = cosf(theta_y);
        const float sin_y = sinf(theta_y);
        const long base_y = base + part;
        const float y0 = data[base_y + i];
        const float y1 = data[base_y + i + quarter];
        data[base_y + i] = y0 * cos_y - y1 * sin_y;
        data[base_y + i + quarter] = y0 * sin_y + y1 * cos_y;
    }
}

// 视觉塔 QK^T：padding patch 作为 key 时置为 -inf
// blockIdx.x = 查询位置 i，blockIdx.y = 查询 head
__global__ void vision_qk_kernel(
    const float* q,        // [num_patches, num_q_heads * head_dim]
    const float* k,        // [num_patches, num_kv_heads * head_dim]
    float* scores,         // [num_q_heads, num_patches, num_patches]
    const uint8_t* valid,  // [num_patches]
    int num_patches,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    if (i >= num_patches || h >= num_q_heads) return;

    const int group = num_q_heads / num_kv_heads;
    const int kv_head = h / group;

    __shared__ float reduce_buf[ENC_BLOCK_SIZE];

    float* s_row = scores + (static_cast<long>(h) * num_patches + i) * num_patches;
    const float* q_row = q + (static_cast<long>(i) * num_q_heads + h) * head_dim;

    for (int j = threadIdx.x; j < num_patches; j += blockDim.x) {
        s_row[j] = -1e30f;
    }
    __syncthreads();

    for (int j = 0; j < num_patches; ++j) {
        if (valid[j] == 0) continue;  // padding patch 不参与注意力
        const float* k_row = k + (static_cast<long>(j) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q_row[d] * k_row[d];
        }

        reduce_buf[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            s_row[j] = reduce_buf[0] * scale;
        }
        __syncthreads();
    }
}

}  // namespace

std::vector<float> CUDAAttentionOps::encoder_attention(
    const std::vector<float>& q_input,
    const std::vector<float>& k_input,
    const std::vector<float>& v_input,
    const std::vector<float>& q_norm_weight,
    const std::vector<float>& k_norm_weight,
    size_t seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float rope_theta,
    float attn_scale,
    bool normalize_v,
    int sliding_window,
    float rms_eps
) {
    if (seq_len == 0 || num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("encoder_attention: 非法维度参数");
    }
    const size_t q_stride = num_q_heads * head_dim;
    const size_t kv_stride = num_kv_heads * head_dim;
    if (q_input.size() != seq_len * q_stride ||
        k_input.size() != seq_len * kv_stride ||
        v_input.size() != seq_len * kv_stride) {
        throw std::invalid_argument("encoder_attention: 输入张量尺寸不匹配");
    }

    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_out = nullptr;
    float* d_q_norm = nullptr;
    float* d_k_norm = nullptr;

    const size_t q_size = seq_len * q_stride * sizeof(float);
    const size_t kv_size = seq_len * kv_stride * sizeof(float);

    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_q, q_size));
    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_k, kv_size));
    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_v, kv_size));
    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_out, q_size));

    CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_q, q_input.data(), q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_k, k_input.data(), kv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_v, v_input.data(), kv_size, cudaMemcpyHostToDevice));

    // Q/K/V 的 per-head RMSNorm（Q/K 带权重，V 无权重）
    const int q_norm_blocks = static_cast<int>((seq_len * num_q_heads + ENC_BLOCK_SIZE - 1) / ENC_BLOCK_SIZE);
    const int kv_norm_blocks = static_cast<int>((seq_len * num_kv_heads + ENC_BLOCK_SIZE - 1) / ENC_BLOCK_SIZE);

    if (!q_norm_weight.empty()) {
        CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_q_norm, head_dim * sizeof(float)));
        CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_q_norm, q_norm_weight.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (!k_norm_weight.empty()) {
        CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_k_norm, head_dim * sizeof(float)));
        CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_k_norm, k_norm_weight.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice));
    }

    encoder_rms_norm_head_kernel<<<q_norm_blocks, ENC_BLOCK_SIZE>>>(
        d_q, d_q_norm, static_cast<int>(seq_len), static_cast<int>(num_q_heads),
        static_cast<int>(head_dim), rms_eps, q_norm_weight.empty() ? 0 : 1);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    encoder_rms_norm_head_kernel<<<kv_norm_blocks, ENC_BLOCK_SIZE>>>(
        d_k, d_k_norm, static_cast<int>(seq_len), static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim), rms_eps, k_norm_weight.empty() ? 0 : 1);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    if (normalize_v) {
        encoder_rms_norm_head_kernel<<<kv_norm_blocks, ENC_BLOCK_SIZE>>>(
            d_v, nullptr, static_cast<int>(seq_len), static_cast<int>(num_kv_heads),
            static_cast<int>(head_dim), rms_eps, 0);
        CHECK_CUDA_ERROR_ENC(cudaGetLastError());
    }

    // 仅对 Q/K 应用 RoPE
    encoder_rope_kernel<<<q_norm_blocks, ENC_BLOCK_SIZE>>>(
        d_q, static_cast<int>(seq_len), static_cast<int>(num_q_heads),
        static_cast<int>(head_dim), rope_theta);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    encoder_rope_kernel<<<kv_norm_blocks, ENC_BLOCK_SIZE>>>(
        d_k, static_cast<int>(seq_len), static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim), rope_theta);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    // QK^T -> 双向 softmax -> AV
    float* d_scores = nullptr;
    const size_t scores_size = num_q_heads * seq_len * seq_len * sizeof(float);
    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_scores, scores_size));

    dim3 attn_grid(static_cast<unsigned int>(seq_len), static_cast<unsigned int>(num_q_heads));
    encoder_qk_kernel<<<attn_grid, ENC_BLOCK_SIZE>>>(
        d_q, d_k, d_scores,
        static_cast<int>(seq_len), static_cast<int>(num_q_heads),
        static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
        attn_scale, sliding_window);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    const int softmax_blocks = static_cast<int>(num_q_heads * seq_len);
    encoder_softmax_kernel<<<softmax_blocks, ENC_BLOCK_SIZE>>>(d_scores, static_cast<int>(seq_len));
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    encoder_av_kernel<<<attn_grid, ENC_BLOCK_SIZE>>>(
        d_scores, d_v, d_out,
        static_cast<int>(seq_len), static_cast<int>(num_q_heads),
        static_cast<int>(num_kv_heads), static_cast<int>(head_dim));
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());
    CHECK_CUDA_ERROR_ENC(cudaDeviceSynchronize());

    std::vector<float> output(seq_len * q_stride);
    CHECK_CUDA_ERROR_ENC(cudaMemcpy(output.data(), d_out, q_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR_ENC(cudaFree(d_q));
    CHECK_CUDA_ERROR_ENC(cudaFree(d_k));
    CHECK_CUDA_ERROR_ENC(cudaFree(d_v));
    CHECK_CUDA_ERROR_ENC(cudaFree(d_out));
    CHECK_CUDA_ERROR_ENC(cudaFree(d_scores));
    if (d_q_norm != nullptr) CHECK_CUDA_ERROR_ENC(cudaFree(d_q_norm));
    if (d_k_norm != nullptr) CHECK_CUDA_ERROR_ENC(cudaFree(d_k_norm));

    return output;
}

std::vector<float> CUDAAttentionOps::vision_attention(
    const std::vector<float>& q_input,
    const std::vector<float>& k_input,
    const std::vector<float>& v_input,
    const std::vector<float>& q_norm_weight,
    const std::vector<float>& k_norm_weight,
    const std::vector<int32_t>& positions,
    const std::vector<uint8_t>& valid,
    size_t num_patches,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float rope_theta,
    float attn_scale,
    float rms_eps
) {
    if (num_patches == 0 || num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("vision_attention: 非法维度参数");
    }
    const size_t q_stride = num_q_heads * head_dim;
    const size_t kv_stride = num_kv_heads * head_dim;
    if (q_input.size() != num_patches * q_stride ||
        k_input.size() != num_patches * kv_stride ||
        v_input.size() != num_patches * kv_stride ||
        positions.size() != num_patches * 2 ||
        valid.size() != num_patches) {
        throw std::invalid_argument("vision_attention: 输入张量尺寸不匹配");
    }

    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_out = nullptr;
    float* d_q_norm = nullptr;
    float* d_k_norm = nullptr;
    int* d_positions = nullptr;
    uint8_t* d_valid = nullptr;

    const size_t q_size = num_patches * q_stride * sizeof(float);
    const size_t kv_size = num_patches * kv_stride * sizeof(float);

    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_q, q_size));
    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_k, kv_size));
    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_v, kv_size));
    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_out, q_size));
    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_positions, positions.size() * sizeof(int32_t)));
    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_valid, valid.size() * sizeof(uint8_t)));

    CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_q, q_input.data(), q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_k, k_input.data(), kv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_v, v_input.data(), kv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_positions, positions.data(), positions.size() * sizeof(int32_t),
                                    cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_valid, valid.data(), valid.size() * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice));

    const int q_norm_blocks = static_cast<int>((num_patches * num_q_heads + ENC_BLOCK_SIZE - 1) / ENC_BLOCK_SIZE);
    const int kv_norm_blocks = static_cast<int>((num_patches * num_kv_heads + ENC_BLOCK_SIZE - 1) / ENC_BLOCK_SIZE);

    if (!q_norm_weight.empty()) {
        CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_q_norm, head_dim * sizeof(float)));
        CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_q_norm, q_norm_weight.data(), head_dim * sizeof(float),
                                        cudaMemcpyHostToDevice));
    }
    if (!k_norm_weight.empty()) {
        CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_k_norm, head_dim * sizeof(float)));
        CHECK_CUDA_ERROR_ENC(cudaMemcpy(d_k_norm, k_norm_weight.data(), head_dim * sizeof(float),
                                        cudaMemcpyHostToDevice));
    }

    encoder_rms_norm_head_kernel<<<q_norm_blocks, ENC_BLOCK_SIZE>>>(
        d_q, d_q_norm, static_cast<int>(num_patches), static_cast<int>(num_q_heads),
        static_cast<int>(head_dim), rms_eps, q_norm_weight.empty() ? 0 : 1);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    encoder_rms_norm_head_kernel<<<kv_norm_blocks, ENC_BLOCK_SIZE>>>(
        d_k, d_k_norm, static_cast<int>(num_patches), static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim), rms_eps, k_norm_weight.empty() ? 0 : 1);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    // V 的无权重 RMSNorm
    encoder_rms_norm_head_kernel<<<kv_norm_blocks, ENC_BLOCK_SIZE>>>(
        d_v, nullptr, static_cast<int>(num_patches), static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim), rms_eps, 0);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    // 轴向 2D RoPE（仅 Q/K）
    vision_axial_rope_kernel<<<q_norm_blocks, ENC_BLOCK_SIZE>>>(
        d_q, d_positions, static_cast<int>(num_patches), static_cast<int>(num_q_heads),
        static_cast<int>(head_dim), rope_theta);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    vision_axial_rope_kernel<<<kv_norm_blocks, ENC_BLOCK_SIZE>>>(
        d_k, d_positions, static_cast<int>(num_patches), static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim), rope_theta);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    float* d_scores = nullptr;
    const size_t scores_size = num_q_heads * num_patches * num_patches * sizeof(float);
    CHECK_CUDA_ERROR_ENC(cudaMalloc(&d_scores, scores_size));

    dim3 attn_grid(static_cast<unsigned int>(num_patches), static_cast<unsigned int>(num_q_heads));
    vision_qk_kernel<<<attn_grid, ENC_BLOCK_SIZE>>>(
        d_q, d_k, d_scores, d_valid,
        static_cast<int>(num_patches), static_cast<int>(num_q_heads),
        static_cast<int>(num_kv_heads), static_cast<int>(head_dim), attn_scale);
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    const int softmax_blocks = static_cast<int>(num_q_heads * num_patches);
    encoder_softmax_kernel<<<softmax_blocks, ENC_BLOCK_SIZE>>>(d_scores, static_cast<int>(num_patches));
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());

    encoder_av_kernel<<<attn_grid, ENC_BLOCK_SIZE>>>(
        d_scores, d_v, d_out,
        static_cast<int>(num_patches), static_cast<int>(num_q_heads),
        static_cast<int>(num_kv_heads), static_cast<int>(head_dim));
    CHECK_CUDA_ERROR_ENC(cudaGetLastError());
    CHECK_CUDA_ERROR_ENC(cudaDeviceSynchronize());

    std::vector<float> output(num_patches * q_stride);
    CHECK_CUDA_ERROR_ENC(cudaMemcpy(output.data(), d_out, q_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR_ENC(cudaFree(d_q));
    CHECK_CUDA_ERROR_ENC(cudaFree(d_k));
    CHECK_CUDA_ERROR_ENC(cudaFree(d_v));
    CHECK_CUDA_ERROR_ENC(cudaFree(d_out));
    CHECK_CUDA_ERROR_ENC(cudaFree(d_scores));
    CHECK_CUDA_ERROR_ENC(cudaFree(d_positions));
    CHECK_CUDA_ERROR_ENC(cudaFree(d_valid));
    if (d_q_norm != nullptr) CHECK_CUDA_ERROR_ENC(cudaFree(d_q_norm));
    if (d_k_norm != nullptr) CHECK_CUDA_ERROR_ENC(cudaFree(d_k_norm));

    return output;
}
