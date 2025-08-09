#include "sentence_transformers.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    try {
        std::string model_path = "/home/wildkid1024/Public/Models/bge-large-zh-v1.5";

        // 假设 SentenceTransformer 构造函数支持传入分词器路径
        SentenceTransformer model(model_path);

        std::vector<std::string> texts = {"This is a test sentence.", "Another example sentence."};
        std::vector<std::vector<float>> embeddings = model.encode_batch(texts);

        for (const auto& embedding : embeddings) {
            // 打印每个文本的嵌入向量shape
            std::cout << "embedding shape: " << embedding.size() << std::endl;
            for (float val : embedding) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
