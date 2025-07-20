#include "sentence_transformers_impl.h"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include "ops/cpu_attention_ops.h"
#include "ops/cuda_attention_ops.h"
#include "ops/cpu_matrix_ops.h"
#include "ops/cuda_matrix_ops.h"

// 从 config.json 加载词汇表
void SentenceTransformerImpl::load_vocab(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("无法打开 config.json 文件: " + config_path);
    }

    json config;
    config_file >> config;

    // 假设 config.json 中有 "vocab" 字段，其为键值对形式
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

// 使用 tokenizers 库进行分词，并结合词汇表
std::vector<int64_t> SentenceTransformerImpl::tokenize(const std::string& text) {
    auto encoding = tokenizer.encode(text, true);
    auto tokens = encoding.get_tokens();
    std::vector<int64_t> ids;
    for (const auto& token : tokens) {
        if (vocab.find(token) != vocab.end()) {
            ids.push_back(vocab[token]);
        } else {
            // 处理未登录词，假设 [UNK] 的 ID 为 100
            ids.push_back(100); 
        }
    }
    return ids;
}

// 简单的嵌入层前向传播
std::vector<float> SentenceTransformerImpl::embedding_layer(const std::vector<int64_t>& tokens) {
    std::vector<float> output(embedding_dim, 0.0f);
    const std::vector<float>& embedding_weight = weights["embedding.weight"];
    for (int64_t token : tokens) {
        size_t start_idx = token * embedding_dim;
        for (size_t i = 0; i < embedding_dim; ++i) {
            output[i] += embedding_weight[start_idx + i];
        }
    }
    return output;
}

SentenceTransformerImpl::SentenceTransformerImpl(const std::string& model_path, const std::string& tokenizer_path, const std::string& config_path) {
    try {
        // 加载分词器
        tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path);
    } catch (const std::exception& e) {
        throw std::runtime_error("加载分词器失败: " + std::string(e.what()));
    }

    // 加载词汇表
    load_vocab(config_path);

    if (!load_single_safetensors_file(model_path, weights)) {
        throw std::runtime_error("加载模型权重失败");
    }
    // 假设嵌入层权重名为 "embedding.weight"
    if (weights.find("embedding.weight") != weights.end()) {
        // 假设嵌入维度是权重矩阵的列数
        embedding_dim = weights["embedding.weight"].size() / vocab.size();
    } else {
        throw std::runtime_error("未找到嵌入层权重");
    }
}

// 完整的 BGE 前向传播
std::vector<float> SentenceTransformerImpl::bge_forward(const std::vector<int64_t>& tokens) {
    // 先通过嵌入层
    std::vector<float> embedded = embedding_layer(tokens);
    size_t seq_len = tokens.size();

    // 添加位置编码
    embedded = add_position_encoding(embedded, seq_len, embedding_dim);

    // 假设 BGE 模型有 12 层 Transformer 编码器
    const int num_layers = 12;
    for (int layer = 0; layer < num_layers; ++layer) {
        std::string q_weight_name = "transformer.layers." + std::to_string(layer) + ".attention.self.query.weight";
        std::string k_weight_name = "transformer.layers." + std::to_string(layer) + ".attention.self.key.weight";
        std::string v_weight_name = "transformer.layers." + std::to_string(layer) + ".attention.self.value.weight";
        std::string ff_weight_1_name = "transformer.layers." + std::to_string(layer) + ".intermediate.dense.weight";
        std::string ff_weight_2_name = "transformer.layers." + std::to_string(layer) + ".output.dense.weight";

        // 获取对应层的权重
        const std::vector<float>& weight_q = weights[q_weight_name];
        const std::vector<float>& weight_k = weights[k_weight_name];
        const std::vector<float>& weight_v = weights[v_weight_name];
        const std::vector<float>& weight_ff_1 = weights[ff_weight_1_name];
        const std::vector<float>& weight_ff_2 = weights[ff_weight_2_name];

        // 多头注意力机制
        std::vector<float> attn_output = multi_head_attention(embedded, weight_q, weight_k, weight_v, 12, embedding_dim);

        // 残差连接
        for (size_t i = 0; i < embedded.size(); ++i) {
            attn_output[i] += embedded[i];
        }

        // 层归一化（简化处理）
        float mean = 0.0f;
        for (float val : attn_output) {
            mean += val;
        }
        mean /= attn_output.size();
        for (float& val : attn_output) {
            val -= mean;
        }

        // 前馈神经网络
        std::vector<float> ff_output = feed_forward_network(attn_output, weight_ff_1, weight_ff_2, embedding_dim, 3072);

        // 残差连接
        for (size_t i = 0; i < attn_output.size(); ++i) {
            ff_output[i] += attn_output[i];
        }

        // 层归一化（简化处理）
        mean = 0.0f;
        for (float val : ff_output) {
            mean += val;
        }
        mean /= ff_output.size();
        for (float& val : ff_output) {
            val -= mean;
        }

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
    std::vector<int64_t> tokens = tokenize(text);
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
