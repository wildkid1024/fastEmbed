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
    size_t i = 0;
    while (i < text.size()) {
        char c = text[i];
        uint8_t byte = static_cast<uint8_t>(c);

        // 跳过空格
        if (isspace(c)) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            i++;
            continue;
        }

        // 判断是否为中文字符（Unicode：0x4E00-0x9FA5）
        bool is_chinese = (static_cast<uint8_t>(c) & 0x80) != 0;  // 非ASCII字符
        if (is_chinese) {
           // 判断UTF-8字节长度
            int bytes_needed = 0;
            if ((byte & 0xE0) == 0xC0) bytes_needed = 2;  // 双字节 (110xxxxx)
            else if ((byte & 0xF0) == 0xE0) bytes_needed = 3;  // 三字节 (1110xxxx)
            else if ((byte & 0xF8) == 0xF0) bytes_needed = 4;  // 四字节 (11110xxx)
            else {  // 无效UTF-8首字节
                tokens.push_back("[UNK]");
                i++;
                continue;
            }

            // 检查剩余字节是否足够
            if (i + bytes_needed > text.size()) {
                tokens.push_back("[UNK]");
                i++;
                continue;
            }

            // 提取完整UTF-8字符
            std::string chinese_char;
            for (int j = 0; j < bytes_needed; j++) {
                chinese_char += text[i + j];
            }
            i += bytes_needed;

            // 添加中文字符到分词结果
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            tokens.push_back(chinese_char);
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
        i++;
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

// XLMRobertaTokenizer implementation
XLMRobertaTokenizer::XLMRobertaTokenizer(const std::string& tokenizer_json_path, const std::string& sentencepiece_model_path) {
    // 1. 初始化SentencePiece处理器
    sp_processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto status = sp_processor->Load(sentencepiece_model_path);
    if (!status.ok()) {
        throw std::runtime_error("无法加载SentencePiece模型: " + status.ToString());
    }
    
    // 2. 从tokenizer.json加载词汇表和特殊token信息
    std::ifstream tokenizer_file(tokenizer_json_path);
    if (!tokenizer_file.is_open()) {
        throw std::runtime_error("无法打开tokenizer.json文件: " + tokenizer_json_path);
    }
    
    nlohmann::json tokenizer_config;
    tokenizer_file >> tokenizer_config;
    
    // 3. 加载词汇表
     if (tokenizer_config.contains("model") && tokenizer_config["model"].contains("vocab")) {
        const auto& vocab_json = tokenizer_config["model"]["vocab"];
        
        if (vocab_json.is_array()) {
            // 数组格式: [ [token1, score1], [token2, score2], ... ]
            int id = 0;
            for (const auto& item : vocab_json) {
                if (item.is_array() && item.size() >= 1 && item[0].is_string()) {
                    std::string token = item[0].get<std::string>();
                    vocab[token] = id;
                    id_to_token[id] = token;
                    id++;
                }
            }
        } else if (vocab_json.is_object()) {
            // 兼容旧的对象格式
            for (auto& [key, value] : vocab_json.items()) {
                if (value.is_number_integer()) {
                    vocab[key] = value.get<int>();
                    id_to_token[value.get<int>()] = key;
                }
            }
        }
    }
    
    // 4. 设置特殊token ID
    if (tokenizer_config.contains("bos_token_id")) {
        bos_token_id = tokenizer_config["bos_token_id"].get<int>();
    } else {
        bos_token_id = 0; // 默认值，通常为0
    }
    
    if (tokenizer_config.contains("eos_token_id")) {
        eos_token_id = tokenizer_config["eos_token_id"].get<int>();
    } else {
        eos_token_id = 2; // 默认值，通常为2
    }
    
    if (tokenizer_config.contains("unk_token_id")) {
        unk_token_id = tokenizer_config["unk_token_id"].get<int>();
    } else {
        unk_token_id = 3; // 默认值，通常为3
    }
    
    if (tokenizer_config.contains("pad_token_id")) {
        pad_token_id = tokenizer_config["pad_token_id"].get<int>();
    } else {
        pad_token_id = unk_token_id; // 默认使用unk_token_id作为pad_token_id
    }
}

std::vector<std::string> XLMRobertaTokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::vector<std::string> sp_tokens;
    
    // 使用SentencePiece进行分词
    auto status = sp_processor->Encode(text, &sp_tokens);
    if (!status.ok()) {
        throw std::runtime_error("SentencePiece分词失败: " + status.ToString());
    }
    
    // 处理特殊token，移除可能的<s>和</s>
    for (const auto& token : sp_tokens) {
        if (token != "<s>" && token != "</s>") {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

std::vector<int> XLMRobertaTokenizer::encode(const std::vector<std::string>& tokens) {
    std::vector<int> token_ids;
    
    // 添加开始token
    token_ids.push_back(bos_token_id);
    
    // 将tokens转换为ID
    for (const auto& token : tokens) {
        // 尝试直接在词汇表中查找
        auto it = vocab.find(token);
        if (it != vocab.end()) {
            token_ids.push_back(it->second);
        } else {
            // 如果找不到，使用SentencePiece处理器获取ID
            std::vector<int> sp_ids;
            auto status = sp_processor->Encode({token}, &sp_ids);
            if (!status.ok() || sp_ids.empty()) {
                // 如果仍然失败，使用unk_token_id
                token_ids.push_back(unk_token_id);
            } else {
                // 添加SentencePiece生成的ID
                for (int id : sp_ids) {
                    token_ids.push_back(id);
                }
            }
        }
    }
    
    // 添加结束token
    token_ids.push_back(eos_token_id);
    
    // 打印token_ids以进行调试
    std::cout << "Token IDs: ";
    for (int id : token_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    return token_ids;
}

std::string XLMRobertaTokenizer::decode(const std::vector<int>& token_ids) {
    std::string text;
    
    // 使用SentencePiece处理器进行解码
    auto status = sp_processor->Decode(token_ids, &text);
    if (!status.ok()) {
        // 如果解码失败，尝试手动解码
        for (int id : token_ids) {
            if (id_to_token.find(id) != id_to_token.end()) {
                text += id_to_token[id];
            } else {
                text += "<unk>";
            }
        }
    }
    
    return text;
}