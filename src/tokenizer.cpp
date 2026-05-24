#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <iostream>  // 添加此头文件以使用 std::cout
#include <cstdint>  // 添加此头文件以使用 uint8_t
#include <filesystem>  // 添加filesystem头文件
#include <iomanip>  // 添加iomanip头文件以使用setw和setfill
#include <chrono>  // 添加chrono头文件以进行性能测量

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

// Qwen3Tokenizer implementation - Hugging Face format constructor
Qwen3Tokenizer::Qwen3Tokenizer(const std::string& tokenizer_json_path, const std::string& vocab_json_path) {
    // 首先尝试从tokenizer.json加载完整的tokenizer配置
    std::ifstream tokenizer_file(tokenizer_json_path);
    if (!tokenizer_file.is_open()) {
        throw std::runtime_error("无法打开tokenizer.json文件: " + tokenizer_json_path);
    }
    
    nlohmann::json tokenizer_config;
    tokenizer_file >> tokenizer_config;
    
    // 从tokenizer.json的model部分加载词汇表和merges
    if (tokenizer_config.contains("model")) {
        const auto& model = tokenizer_config["model"];
        
        // 加载词汇表
        if (model.contains("vocab")) {
            for (auto& [token, value] : model["vocab"].items()) {
                int id = value.get<int>();
                vocab[token] = id;
                id_to_token[id] = token;
            }
            std::cout << "从tokenizer.json加载了 " << vocab.size() << " 个词汇" << std::endl;
        }
        
        // 加载BPE merges
        if (model.contains("merges")) {
            int merge_count = 0;
            for (const auto& merge : model["merges"]) {
                if (merge.is_array() && merge.size() == 2) {
                    // 数组格式: ["token1", "token2"]
                    std::string first = merge[0].get<std::string>();
                    std::string second = merge[1].get<std::string>();
                    merges.push_back(first + " " + second);
                    merge_count++;
                } else if (merge.is_string()) {
                    // 字符串格式: "token1 token2"
                    merges.push_back(merge.get<std::string>());
                    merge_count++;
                }
            }
            std::cout << "从tokenizer.json加载了 " << merge_count << " 个merges" << std::endl;
        }
    } else if (std::filesystem::exists(vocab_json_path)) {
        // 如果tokenizer.json中没有model，尝试从vocab.json加载
        std::ifstream vocab_file(vocab_json_path);
        if (!vocab_file.is_open()) {
            throw std::runtime_error("无法打开vocab.json文件: " + vocab_json_path);
        }
        
        nlohmann::json vocab_json;
        vocab_file >> vocab_json;
        
        if (vocab_json.is_object()) {
            for (auto& [token, value] : vocab_json.items()) {
                int id = value.get<int>();
                vocab[token] = id;
                id_to_token[id] = token;
            }
            std::cout << "从vocab.json加载了 " << vocab.size() << " 个词汇" << std::endl;
        }
    } else {
        throw std::runtime_error("无法找到有效的词汇表文件");
    }
    
    // 设置特殊token ID - 从added_tokens中获取
    bos_token_id = -1;  // Qwen3默认没有BOS token
    eos_token_id = 151645;  // Qwen3 EOS token (|im_end|>)
    unk_token_id = 151643;  // UNK token (|endoftext|>)
    pad_token_id = 151643;  // PAD token (|endoftext|>)
        
    if (tokenizer_config.contains("added_tokens")) {
        const auto& added_tokens = tokenizer_config["added_tokens"];
        for (const auto& token_info : added_tokens) {
            if (token_info.contains("content") && token_info.contains("id")) {
                std::string content = token_info["content"];
                int token_id = token_info["id"];
                    
                // 添加到词汇表中
                vocab[content] = token_id;
                id_to_token[token_id] = content;
                    
                // 识别特殊token
                if (content == "<|endoftext|>") {
                    // <|endoftext|> 用作 PAD 和 UNK，但不是 BOS
                    unk_token_id = token_id;
                    pad_token_id = token_id;
                    // bos_token_id 保持为 -1（没有BOS）
                    // 注意：Qwen3的post_processor使用<|endoftext|>作为结束token，而不是<|im_end|>
                    eos_token_id = token_id;  // 用于tokenizer编码时添加的结束token
                } else if (content == "<|im_start|>") {
                    im_start_token_id = token_id;
                } else if (content == "<|im_end|>") {
                    im_end_token_id = token_id;
                    // 不将<|im_end|>设为eos_token_id，因为post_processor使用的是<|endoftext|>
                }
            }
        }
    }
    
    std::cout << "特殊Token IDs - BOS: " << bos_token_id << ", EOS: " << eos_token_id 
              << ", UNK: " << unk_token_id << ", PAD: " << pad_token_id << std::endl;
    std::cout << "词汇表大小: " << vocab.size() << std::endl;
}

// Pre-tokenization: 使用正则表达式分割文本
// 正则模式: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
static std::vector<std::string> pre_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    
    if (text.empty()) {
        return tokens;
    }
    
    // 简化的pre-tokenization实现：
    // 1. 按空白字符分割，但将空格附加到下一个token前面
    // 2. 保留标点符号作为独立tokens
    // 3. 将空格字符(0x20)替换为特殊字符Ġ(U+0120)
    
    size_t i = 0;
    size_t len = text.size();
    std::string pending_spaces;  // 待处理的空格
    
    while (i < len) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        
        // 跳过回车和换行符的特殊处理
        if (c == '\r' || c == '\n') {
            size_t start = i;
            while (i < len && (text[i] == '\r' || text[i] == '\n')) {
                i++;
            }
            std::string newline_token = pending_spaces + text.substr(start, i - start);
            tokens.push_back(newline_token);
            pending_spaces.clear();
            continue;
        }
        
        // 处理空白字符 - 收集但不立即添加
        if (std::isspace(c)) {
            size_t start = i;
            while (i < len && std::isspace(static_cast<unsigned char>(text[i]))) {
                i++;
            }
            // 注意：文本已经在tokenize函数中应用了bytes-to-unicode映射
            // 空格(0x20)已经被映射为Ġ(U+0120)，所以这里直接使用原始字符
            pending_spaces = text.substr(start, i - start);
            continue;
        }
        
        // 处理字母（包括UTF-8多字节字符）
        if (std::isalpha(c) || (c > 127)) {
            size_t start = i;
            
            // 收集连续的字母字符
            while (i < len) {
                unsigned char ch = static_cast<unsigned char>(text[i]);
                if (std::isalpha(ch) || ch > 127 || ch == '\'') {
                    // 处理UTF-8多字节字符
                    if (ch > 127) {
                        size_t char_len = 1;
                        if ((ch & 0xE0) == 0xC0) char_len = 2;
                        else if ((ch & 0xF0) == 0xE0) char_len = 3;
                        else if ((ch & 0xF8) == 0xF0) char_len = 4;
                        i += char_len;
                    } else {
                        i++;
                    }
                } else {
                    break;
                }
            }
            // 将待处理的空格作为前缀添加
            std::string token = pending_spaces + text.substr(start, i - start);
            tokens.push_back(token);
            pending_spaces.clear();
            continue;
        }
        
        // 处理数字
        if (std::isdigit(c)) {
            size_t start = i;
            while (i < len && std::isdigit(static_cast<unsigned char>(text[i]))) {
                i++;
            }
            std::string token = pending_spaces + text.substr(start, i - start);
            tokens.push_back(token);
            pending_spaces.clear();
            continue;
        }
        
        // 处理标点符号和其他字符
        {
            size_t start = i;
            
            // 收集连续的标点符号
            while (i < len) {
                unsigned char ch = static_cast<unsigned char>(text[i]);
                if (!std::isalnum(ch) && !std::isspace(ch) && ch != '\r' && ch != '\n') {
                    if (ch > 127) {
                        // UTF-8多字节字符
                        size_t char_len = 1;
                        if ((ch & 0xE0) == 0xC0) char_len = 2;
                        else if ((ch & 0xF0) == 0xE0) char_len = 3;
                        else if ((ch & 0xF8) == 0xF0) char_len = 4;
                        i += char_len;
                    } else {
                        i++;
                    }
                } else {
                    break;
                }
            }
            if (i > start) {
                // 标点符号前可能有一个空格（根据正则表达式）
                std::string token = pending_spaces + text.substr(start, i - start);
                tokens.push_back(token);
                pending_spaces.clear();
            }
            continue;
        }
    }
    
    // 如果最后还有未处理的空格，添加到末尾
    if (!pending_spaces.empty()) {
        tokens.push_back(pending_spaces);
    }
    
    return tokens;
}

// BPE编码辅助函数
// 将文本转换为Unicode字符级别的tokens
static std::vector<std::string> text_to_unicode_chars(const std::string& text) {
    std::vector<std::string> tokens;
    
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t char_len = 1;
        
        // 确定UTF-8字符的字节数
        if ((c & 0x80) == 0) {
            char_len = 1;  // ASCII
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }
        
        if (i + char_len > text.size()) {
            char_len = text.size() - i;
        }
        
        tokens.push_back(text.substr(i, char_len));
        i += char_len;
    }
    
    return tokens;
}

// Bytes-to-Unicode映射（GPT-2风格）
static std::unordered_map<int, char32_t> build_bytes_to_unicode() {
    std::unordered_map<int, char32_t> mapping;
    
    // !到~ (0x21-0x7E)
    for (int i = 0x21; i <= 0x7E; ++i) {
        mapping[i] = static_cast<char32_t>(i);
    }
    
    // ¡到¬ (0xA1-0xAC)
    for (int i = 0xA1; i <= 0xAC; ++i) {
        mapping[i] = static_cast<char32_t>(i);
    }
    
    // ®到ÿ (0xAE-0xFF)
    for (int i = 0xAE; i <= 0xFF; ++i) {
        mapping[i] = static_cast<char32_t>(i);
    }
    
    // 其余字节映射到0x100+
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (mapping.find(b) == mapping.end()) {
            mapping[b] = static_cast<char32_t>(256 + n);
            n++;
        }
    }
    
    return mapping;
}

// 正确的 BPE 合并实现：按照 merges 顺序依次应用
static std::vector<std::string> apply_bpe_merges_fast(std::vector<std::string> tokens,
                                                      const std::vector<std::string>& merges_list,
                                                      const std::unordered_map<std::string, int>& vocab) {
    if (merges_list.empty() || tokens.size() <= 1) {
        return tokens;
    }

    // 对每个 merge 规则，依次尝试应用到 tokens
    for (const auto& merge_str : merges_list) {
        size_t pos = merge_str.find(' ');
        if (pos == std::string::npos) continue;
        
        std::string first = merge_str.substr(0, pos);
        std::string second = merge_str.substr(pos + 1);
        
        // 尝试在 tokens 中找到 first + second 的 pair 并合并
        std::vector<std::string> new_tokens;
        bool merged = false;
        
        for (size_t i = 0; i < tokens.size();) {
            if (i + 1 < tokens.size() && tokens[i] == first && tokens[i+1] == second) {
                // 找到匹配的 pair，合并它们
                new_tokens.push_back(first + second);
                i += 2;
                merged = true;
            } else {
                new_tokens.push_back(tokens[i]);
                i++;
            }
        }
        
        if (merged) {
            tokens.swap(new_tokens);
            if (tokens.size() <= 1) break;  // 如果只剩一个token，提前结束
        }
    }

    return tokens;
}

std::vector<std::string> Qwen3Tokenizer::tokenize(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    
    // 步骤0: 对整个文本应用bytes-to-unicode映射
    static auto bytes_to_unicode = build_bytes_to_unicode();
    std::string mapped_text;
    for (unsigned char c : text) {
        char32_t mapped_char = bytes_to_unicode[c];
        
        // 将char32_t转换为UTF-8字符串
        if (mapped_char < 0x80) {
            mapped_text += static_cast<char>(mapped_char);
        } else if (mapped_char < 0x800) {
            mapped_text += static_cast<char>(0xC0 | (mapped_char >> 6));
            mapped_text += static_cast<char>(0x80 | (mapped_char & 0x3F));
        } else if (mapped_char < 0x10000) {
            mapped_text += static_cast<char>(0xE0 | (mapped_char >> 12));
            mapped_text += static_cast<char>(0x80 | ((mapped_char >> 6) & 0x3F));
            mapped_text += static_cast<char>(0x80 | (mapped_char & 0x3F));
        }
    }
    
    // 步骤1: Pre-tokenization - 按正则表达式分割文本
    std::vector<std::string> pre_tokens = pre_tokenize(mapped_text);
    
    static bool debug_enabled = false;
    if (debug_enabled) {
        printf("DEBUG tokenize: text='%s', pre_tokens count=%zu\n", text.c_str(), pre_tokens.size());
        for (size_t i = 0; i < pre_tokens.size(); ++i) {
            printf("  pre_token[%zu] = '%s'\n", i, pre_tokens[i].c_str());
        }
    }
    
    // 步骤2: 对每个pre-token应用BPE
    std::vector<std::string> final_tokens;
    for (const auto& pre_token : pre_tokens) {
        // 优化：如果pre-token直接在词汇表中，直接使用
        if (vocab.find(pre_token) != vocab.end()) {
            final_tokens.push_back(pre_token);
            continue;
        }
        
        // 将pre-token转换为字符级别（每个UTF-8字符作为一个token）
        std::vector<std::string> char_tokens;
        
        // 解析UTF-8字符串为单个字符
        size_t idx = 0;
        while (idx < pre_token.size()) {
            unsigned char c = static_cast<unsigned char>(pre_token[idx]);
            size_t char_len = 1;
            if ((c & 0x80) == 0) {
                char_len = 1;
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                char_len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4;
            }
            
            char_tokens.push_back(pre_token.substr(idx, char_len));
            idx += char_len;
        }
        
        // 应用BPE merges
        if (!merges.empty() && char_tokens.size() > 1) {
            char_tokens = apply_bpe_merges_fast(char_tokens, merges, vocab);
        }
        
        // 合并到最终结果
        final_tokens.insert(final_tokens.end(), char_tokens.begin(), char_tokens.end());
    }
    
    return final_tokens;
}

std::vector<int> Qwen3Tokenizer::encode(const std::vector<std::string>& tokens) {
    std::vector<int> token_ids;
    for (const auto& token : tokens) {
        auto it = vocab.find(token);
        if (it != vocab.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(unk_token_id);
        }
    }
    return token_ids;
}

std::string Qwen3Tokenizer::decode(const std::vector<int>& token_ids) {
    // 使用词汇表解码
    std::string decoded_text;
    for (int id : token_ids) {
        auto it = id_to_token.find(id);
        if (it != id_to_token.end()) {
            decoded_text += it->second;
        }
    }
    return decoded_text;
}
