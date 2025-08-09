#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <iostream>  // 添加此头文件以使用 std::cout
#include <cstdint>  // 添加此头文件以使用 uint8_t

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
    std::string current_token;

    for (char c : text) {
        // 跳过空格
        if (isspace(c)) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            continue;
        }

        // 判断是否为中文字符（Unicode：0x4E00-0x9FA5）
        bool is_chinese = (static_cast<uint8_t>(c) & 0x80) != 0;  // 非ASCII字符
        if (is_chinese) {
            // 中文单字拆分：先保存当前token，再添加中文字符
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            tokens.push_back(std::string(1, c));
            continue;
        }

        // 非中文：判断是否为字母/数字或标点符号
        bool is_alnum = isalnum(static_cast<unsigned char>(c));
        if (is_alnum) {
            current_token += tolower(c);  // 将字母转换为小写
        } else {
            // 标点符号：先保存当前token，再添加标点
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            tokens.push_back(std::string(1, c));
        }
    }

    // 添加最后一个未处理的token
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    return tokens;
}

// {{ 新增：WordPiece 子词拆分辅助函数 }}
std::vector<std::string> BGETokenizer::wordpiece_split(const std::string& token) {
    std::vector<std::string> subwords;
    std::string current_token = token;

    // 如果 token 本身在词汇表中，直接返回
    if (vocab.find(current_token) != vocab.end()) {
        subwords.push_back(current_token);
        return subwords;
    }

    // 子词拆分：从最长前缀开始匹配
    size_t start = 0;
    while (start < current_token.size()) {
        bool found = false;
        // 尝试匹配最长可能的子词（从当前位置到结尾）
        for (size_t end = current_token.size(); end > start; --end) {
            std::string subword = current_token.substr(start, end - start);
            // 非起始子词需添加 "##" 前缀
            if (start > 0) {
                subword = "##" + subword;
            }
            // 检查子词是否在词汇表中
            if (vocab.find(subword) != vocab.end()) {
                subwords.push_back(subword);
                start = end;  // 继续处理剩余部分
                found = true;
                break;
            }
        }
        if (!found) {
            // 未找到匹配的子词，返回 [UNK]
            return {"[UNK]"};
        }
    }
    return subwords;
}
// {{ 辅助函数结束 }}

// {{ 修改：encode 方法，支持子词拆分 }}
std::vector<int> BGETokenizer::encode(const std::vector<std::string>& tokens) {
    std::vector<int> token_ids;
    for (const auto& token : tokens) {
        // 对每个 token 进行 WordPiece 子词拆分
        std::vector<std::string> subwords = wordpiece_split(token);
        // 将子词转换为 ID
        for (const auto& subword : subwords) {
            auto it = vocab.find(subword);
            if (it != vocab.end()) {
                token_ids.push_back(it->second);
            } else {
                token_ids.push_back(vocab["[UNK]"]);
            }
        }
    }
    return token_ids;
}
// {{ 修改结束 }}

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