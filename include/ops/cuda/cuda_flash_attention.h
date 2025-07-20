#ifndef CUDA_FLASH_ATTENTION_H
#define CUDA_FLASH_ATTENTION_H

#include <vector>

void flash_attention(float* d_q, float* d_k, float* d_v, float* d_out, int batch_size, int seq_len, int num_heads, int head_dim);

#endif // CUDA_FLASH_ATTENTION_H
