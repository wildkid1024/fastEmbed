#include "sentence_transformers_impl.h"
#include <safetensors.h>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>

// 从 config.json 加载词汇表
void SentenceTransformerImpl::load_vocab(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("无法打开 config.json 文件: " + config_path);
    }

    json config;
    config_file >> config;

    // 取['model']['vocab']中的数据表
    // json vocab_json = config["model"]["vocab"];
    config = config["model"];
    
    if (config.contains("vocab") && config["vocab"].is_object()) {
        for (auto& [key, value] : config["vocab"].items()) {
            if (value.is_number_integer()) {
                vocab[key] = value.get<int64_t>();
            }
        }
    } else {
        throw std::runtime_error("config.json 中未找到有效的词汇表字段");
    }
}

// 使用新的 tokenizer 进行分词
std::vector<int32_t> SentenceTransformerImpl::tokenize(const std::string& text) {
    auto tokens = tokenizer.tokenize(text);
    auto token_ids = tokenizer.encode(tokens);
    return std::vector<int32_t>(token_ids.begin(), token_ids.end());
}

// 简单的嵌入层前向传播
std::vector<float> SentenceTransformerImpl::embedding_layer(const std::vector<int32_t>& tokens) {
    // 调整输出向量大小为 tokens.size() * embedding_dim
    std::vector<float> output(tokens.size() * embedding_dim, 0.0f);
    const std::vector<float>& embedding_weight = weights["embeddings.word_embeddings.weight"];

    for (size_t i = 0; i < tokens.size(); ++i) {
        int32_t token = tokens[i];
        size_t start_idx = token * embedding_dim;
        // 复制当前 token 对应的嵌入向量到输出中
        for (size_t j = 0; j < embedding_dim; ++j) {
            output[i * embedding_dim + j] = embedding_weight[start_idx + j];
        }
    }
    return output;
}

SentenceTransformerImpl::SentenceTransformerImpl(const std::string& model_path, const std::string& tokenizer_path, const std::string& config_path)
    : tokenizer(model_path + "/vocab.txt") { // 假设词汇表文件在 model_path 下
    // 加载词汇表
    load_vocab(config_path);

    std::string tensor_path = model_path + "/model.safetensors";

    if (!load_single_safetensors_file(tensor_path, weights)) {  // 使用函数
        throw std::runtime_error("加载模型权重失败");
    }

    // 打印safetensor unordered_map 权重的key
    for (const auto& key : weights) {
        std::cout << key.first << std::endl;
    }

    // 假设嵌入层权重名为 "embedding.weight"
    if (weights.find("embeddings.word_embeddings.weight") != weights.end()) {
        // 假设嵌入维度是权重矩阵的列数
        embedding_dim = weights["embeddings.word_embeddings.weight"].size() / vocab.size();
        // 打印vocab size 和 embedding_dim
        std::cout << "vocab size: " << vocab.size() << std::endl;
        std::cout << "embedding_dim: " << embedding_dim << std::endl;
    } else {
        throw std::runtime_error("未找到嵌入层权重");
    }
}

// 完整的 BGE 前向传播
std::vector<float> SentenceTransformerImpl::bge_forward(const std::vector<int32_t>& tokens) {
    // 先通过嵌入层
    std::vector<float> embedded = embedding_layer(tokens);
    size_t seq_len = tokens.size();

    // 添加位置编码
    embedded = add_position_encoding(embedded, seq_len, embedding_dim);

    // 假设 BGE 模型有 4 层 Transformer 编码器
    const int num_layers = 4;
    for (int layer = 0; layer < num_layers; ++layer) {
        std::string q_weight_name = "encoder.layer." + std::to_string(layer) + ".attention.self.query.weight";
        std::string k_weight_name = "encoder.layer." + std::to_string(layer) + ".attention.self.key.weight";
        std::string v_weight_name = "encoder.layer." + std::to_string(layer) + ".attention.self.value.weight";
        std::string ff_weight_1_name = "encoder.layer." + std::to_string(layer) + ".intermediate.dense.weight";
        std::string ff_weight_2_name = "encoder.layer." + std::to_string(layer) + ".output.dense.weight";
        std::string ln_gamma_name = "encoder.layer." + std::to_string(layer) + ".attention.output.LayerNorm.weight";
        std::string ln_beta_name = "encoder.layer." + std::to_string(layer) + ".attention.output.LayerNorm.bias";
        std::string ln_ff_gamma_name = "encoder.layer." + std::to_string(layer) + ".output.LayerNorm.weight";
        std::string ln_ff_beta_name = "encoder.layer." + std::to_string(layer) + ".output.LayerNorm.bias";

        // 获取对应层的权重
        const std::vector<float>& weight_q = weights[q_weight_name];
        const std::vector<float>& weight_k = weights[k_weight_name];
        const std::vector<float>& weight_v = weights[v_weight_name];
        const std::vector<float>& weight_ff_1 = weights[ff_weight_1_name];
        const std::vector<float>& weight_ff_2 = weights[ff_weight_2_name];
        const std::vector<float>& ln_gamma = weights[ln_gamma_name];
        const std::vector<float>& ln_beta = weights[ln_beta_name];
        const std::vector<float>& ln_ff_gamma = weights[ln_ff_gamma_name];
        const std::vector<float>& ln_ff_beta = weights[ln_ff_beta_name];

        // 多头注意力机制
        std::vector<float> attn_output = multi_head_attention(embedded, weight_q, weight_k, weight_v, 12, embedding_dim);

        // 残差连接
        for (size_t i = 0; i < embedded.size(); ++i) {
            attn_output[i] += embedded[i];
        }

        // 层归一化，调用 cpu_matrix_ops 中的实现
        attn_output = cpu_matrix_ops.layer_norm(attn_output, ln_gamma, ln_beta, embedding_dim);

        // 前馈神经网络
        std::vector<float> ff_output = feed_forward_network(attn_output, weight_ff_1, weight_ff_2, embedding_dim, 3072);

        // 残差连接
        for (size_t i = 0; i < attn_output.size(); ++i) {
            ff_output[i] += attn_output[i];
        }

        // 层归一化，调用 cpu_matrix_ops 中的实现
        ff_output = cpu_matrix_ops.layer_norm(ff_output, ln_ff_gamma, ln_ff_beta, embedding_dim);

        embedded = ff_output;
    }

    // 池化层，取 [CLS] 标记的输出作为句子表示
    std::vector<float> pooled(embedding_dim);
    std::copy_n(embedded.begin(), embedding_dim, pooled.begin());

    // 归一化
    float norm = 0.0f;
    for (float val : pooled) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    for (float& val : pooled) {
        val /= norm;
    }

    return pooled;
}

std::vector<float> SentenceTransformerImpl::encode(const std::string& text) {
    // 分词
    std::vector<int32_t> tokens = tokenize(text);
    // 打印tokens
    std::cout << "tokens: " << std::endl;
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    // BGE 模型前向传播
    return bge_forward(tokens);
}

std::vector<std::vector<float>> SentenceTransformerImpl::encode_batch(const std::vector<std::string>& texts) {
    std::vector<std::vector<float>> result;
    for (const auto& text : texts) {
        result.push_back(encode(text));
    }
    return result;
}

size_t SentenceTransformerImpl::get_embedding_dimension() const {
    return embedding_dim;
}

// 实现 SentenceTransformerImpl 的 multi_head_attention 函数
std::vector<float> SentenceTransformerImpl::multi_head_attention(const std::vector<float>& input, const std::vector<float>& weight_q, const std::vector<float>& weight_k, const std::vector<float>& weight_v, size_t num_heads, size_t embedding_dim) {
    // 调用 CPUAttentionOps 的 multi_head_attention 方法
    return cpu_attention_ops.multi_head_attention(input, weight_q, weight_k, weight_v, num_heads, embedding_dim);
}

// 实现 SentenceTransformerImpl 的 feed_forward_network 函数
std::vector<float> SentenceTransformerImpl::feed_forward_network(const std::vector<float>& input, const std::vector<float>& weight_ff_1, const std::vector<float>& weight_ff_2, size_t embedding_dim, size_t intermediate_dim) {
    // 计算第一层线性变换
    // input 维度: [batch_size * seq_len, embedding_dim]
    // weight_ff_1 维度: [embedding_dim, intermediate_dim]
    // 输出维度: [batch_size * seq_len, intermediate_dim]
    size_t batch_seq_len = input.size() / embedding_dim;
    std::vector<float> hidden = cpu_matrix_ops.matrix_multiply(input, weight_ff_1, batch_seq_len, embedding_dim, intermediate_dim);

    // 应用 GELU 激活函数
    hidden = cpu_matrix_ops.gelu(hidden);

    // 计算第二层线性变换
    // hidden 维度: [batch_size * seq_len, intermediate_dim]
    // weight_ff_2 维度: [intermediate_dim, embedding_dim]
    // 输出维度: [batch_size * seq_len, embedding_dim]
    std::vector<float> output = cpu_matrix_ops.matrix_multiply(hidden, weight_ff_2, batch_seq_len, intermediate_dim, embedding_dim);

    return output;
}

std::vector<float> SentenceTransformerImpl::add_position_encoding(const std::vector<float>& input, 
                                                                  size_t seq_len, 
                                                                  size_t embedding_dim) {
    if (input.size() != seq_len * embedding_dim) {
        std::cout << "input.size(): " << input.size() << std::endl;
        std::cout << "seq_len * embedding_dim: " << seq_len * embedding_dim << std::endl;
        throw std::runtime_error("Input size does not match seq_len * embedding_dim");
    }
    std::vector<float> output = input;
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < embedding_dim; ++i) {
            float angle = static_cast<float>(pos) / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(embedding_dim));
            if (i % 2 == 0) {
                output[pos * embedding_dim + i] += std::sin(angle);
            } else {
                output[pos * embedding_dim + i] += std::cos(angle);
            }
        }
    }
    return output;
}