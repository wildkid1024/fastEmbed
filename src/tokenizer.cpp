#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

BGETokenizer::BGETokenizer(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开词汇表文件: " + vocab_file);
    }
    std::string token;
    int id = 0;
    while (std::getline(file, token)) {
        vocab[token] = id;
        id_to_token[id] = token;
        id++;
    }
    // 检查 [UNK] 标记是否存在
    if (vocab.find("[UNK]") == vocab.end()) {
        throw std::runtime_error("词汇表中缺少 [UNK] 标记");
    }
}

std::vector<std::string> BGETokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

std::vector<int> BGETokenizer::encode(const std::vector<std::string>& tokens) {
    std::vector<int> token_ids;
    for (const auto& token : tokens) {
        auto it = vocab.find(token);
        if (it != vocab.end()) {
            token_ids.push_back(it->second);
        } else {
            // 处理未知标记
            token_ids.push_back(vocab["[UNK]"]);
        }
    }
    return token_ids;
}

std::string BGETokenizer::decode(const std::vector<int>& token_ids) {
    std::string text;
    for (int id : token_ids) {
        text += id_to_token[id] + " ";
    }
    // 移除末尾多余的空格
    if (!text.empty()) {
        text.pop_back();
    }
    return text;
}
