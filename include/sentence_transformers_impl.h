#pragma once
#include <vector>
#include <string>
#include <sentencepiece_processor.h>
#include <nlohmann/json.hpp>
#include "tokenizer.h"  // 引入 tokenizer 头文件

#include "ops/cpu/cpu_attention_ops.h"
#include "ops/cpu/cpu_matrix_ops.h"

#include "ops/cuda/cuda_matrix_ops.h"
#include "ops/cuda/cuda_attention_ops.h"

using json = nlohmann::json;

class SentenceTransformerImpl {
private:
    // 假设使用之前实现的 BGETokenizer
    BGETokenizer tokenizer; 
    std::unordered_map<std::string, int64_t> vocab;
    std::unordered_map<std::string, std::vector<float>> weights;
    size_t embedding_dim;
    // 定义 CPUAttentionOps 实例
    CPUAttentionOps cpu_attention_ops; 
    // 定义 CPUMatrixOps 实例
    CPUMatrixOps cpu_matrix_ops; 

    // Private helper functions
    void load_vocab(const std::string& config_path);
    std::vector<int32_t> tokenize(const std::string& text);
    std::vector<float> embedding_layer(const std::vector<int32_t>& tokens);
    std::vector<float> add_position_encoding(const std::vector<float>& input, size_t seq_len, size_t embedding_dim);
    std::vector<float> multi_head_attention(
        const std::vector<float>& input,
        const std::vector<float>& weight_q,
        const std::vector<float>& weight_k,
        const std::vector<float>& weight_v,
        const std::vector<float>& query_bias,  // New: query bias
        const std::vector<float>& key_bias,    // New: key bias
        const std::vector<float>& value_bias,  // New: value bias
        size_t num_heads,
        size_t embedding_dim
    );
    // std::vector<float> feed_forward_network(const std::vector<float>& input, const std::vector<float>& weight_ff_1, const std::vector<float>& weight_ff_2, size_t embedding_dim, size_t intermediate_dim);
    std::vector<float> feed_forward_network(const std::vector<float>& input, 
                                                                  const std::vector<float>& weight_ff_1, 
                                                                  const std::vector<float>& bias_ff_1, 
                                                                  const std::vector<float>& weight_ff_2, 
                                                                  const std::vector<float>& bias_ff_2, 
                                                                  const std::vector<float>& ln_gamma,  // 新增LayerNorm参数
                                                                  const std::vector<float>& ln_beta,   // 新增LayerNorm参数
                                                                  size_t embedding_dim, 
                                                                  size_t intermediate_dim,
                                                                  float dropout_prob = 0.1f);
    std::vector<float> bge_forward(const std::vector<int32_t>& tokens);

public:
    SentenceTransformerImpl(const std::string& model_path, const std::string& tokenizer_path, const std::string& config_path);
    std::vector<float> encode(const std::string& text);
    std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts);
    size_t get_embedding_dimension() const;
};