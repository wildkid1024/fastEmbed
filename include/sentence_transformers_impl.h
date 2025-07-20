#ifndef SENTENCE_TRANSFORMERS_IMPL_H
#define SENTENCE_TRANSFORMERS_IMPL_H

#include <vector>
#include <string>
#include <unordered_map>
#include "safetensors.h"
#include <tokenizers_cpp.h>
#include <nlohmann/json.hpp>  // 包含 JSON 库头文件

using json = nlohmann::json;

class SentenceTransformerImpl {
public:
    explicit SentenceTransformerImpl(const std::string& model_path, const std::string& tokenizer_path, const std::string& config_path);
    ~SentenceTransformerImpl() = default;

    std::vector<float> encode(const std::string& text);
    std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts);
    size_t get_embedding_dimension() const;

private:
    std::unordered_map<std::string, std::vector<float>> weights;
    size_t embedding_dim;
    tokenizers::Tokenizer tokenizer;
    std::unordered_map<std::string, int64_t> vocab;  // 存储词汇表
    // 分词函数
    std::vector<int64_t> tokenize(const std::string& text);
    // 简单的嵌入层前向传播
    std::vector<float> embedding_layer(const std::vector<int64_t>& tokens);
    // 从 config.json 加载词汇表
    void load_vocab(const std::string& config_path);
};

#endif // SENTENCE_TRANSFORMERS_IMPL_H
