#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <vector>
#include <string>
#include <unordered_map>

class BGETokenizer {
private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;

public:
    BGETokenizer(const std::string& vocab_file);
    std::vector<std::string> tokenize(const std::string& text);
    std::vector<int> encode(const std::vector<std::string>& tokens);
    std::string decode(const std::vector<int>& token_ids);
};

#endif // TOKENIZER_H
