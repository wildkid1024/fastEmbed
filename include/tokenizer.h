#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <memory>  // 确保包含memory头文件
#include "sentencepiece_processor.h"

class BGETokenizer {
private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;
    // WordPiece subword splitting helper function
    std::vector<std::string> wordpiece_split(const std::string& token);

public:
    BGETokenizer(const std::string& vocab_file);
    std::vector<std::string> tokenize(const std::string& text);
    std::vector<int> encode(const std::vector<std::string>& tokens);
    std::string decode(const std::vector<int>& token_ids);
};

// XLMRobertaTokenizer类定义
class XLMRobertaTokenizer {
private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;
    
    // 特殊token ID
    int bos_token_id;  // Beginning of sentence token ID
    int eos_token_id;  // End of sentence token ID
    int unk_token_id;  // Unknown token ID
    int pad_token_id;  // Padding token ID
    
    // SentencePiece处理器
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;

public:
    XLMRobertaTokenizer(const std::string& tokenizer_json_path, const std::string& sentencepiece_model_path);
    std::vector<std::string> tokenize(const std::string& text);
    std::vector<int> encode(const std::vector<std::string>& tokens);
    std::string decode(const std::vector<int>& token_ids);
    
    // 添加获取特殊token ID的方法
    int get_bos_token_id() const { return bos_token_id; }
    int get_eos_token_id() const { return eos_token_id; }
    int get_unk_token_id() const { return unk_token_id; }
    int get_pad_token_id() const { return pad_token_id; }
};

// Qwen3Tokenizer类定义，专门用于处理Qwen3模型的分词
class Qwen3Tokenizer {
private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;
    
    // 特殊token ID
    int bos_token_id;  // Beginning of sentence token ID
    int eos_token_id;  // End of sentence token ID
    int unk_token_id;  // Unknown token ID
    int pad_token_id;  // Padding token ID
    int chat_start_token_id;  // Chat start token ID for Qwen3
    int im_start_token_id;    // Instruction start token ID
    int im_end_token_id;      // Instruction end token ID
    
    // 分词相关参数（用于Hugging Face格式）
    std::vector<std::string> merges;

public:
    // Hugging Face格式构造函数
    Qwen3Tokenizer(const std::string& tokenizer_json_path, const std::string& vocab_json_path);
    
    std::vector<std::string> tokenize(const std::string& text);
    std::vector<int> encode(const std::vector<std::string>& tokens);
    std::string decode(const std::vector<int>& token_ids);
    
    // 添加获取特殊token ID的方法
    int get_bos_token_id() const { return bos_token_id; }
    int get_eos_token_id() const { return eos_token_id; }
    int get_unk_token_id() const { return unk_token_id; }
    int get_pad_token_id() const { return pad_token_id; }
    int get_chat_start_token_id() const { return chat_start_token_id; }
    int get_im_start_token_id() const { return im_start_token_id; }
    int get_im_end_token_id() const { return im_end_token_id; }
};

#endif // TOKENIZER_H