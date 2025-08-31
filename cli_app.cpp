#include "sentence_transformers.h"
#include <iostream>
#include <vector>
#include <string>

// 修改main函数以支持命令行参数和多轮输入
int main(int argc, char* argv[]) {
    try {
        // 命令行参数检查
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
            return 1;
        }

        // 加载模型（仅加载一次）
        std::string model_path = argv[1];
        SentenceTransformer model(model_path);
        std::cout << "模型加载成功，可开始输入文本（输入exit退出）...\n" << std::endl;

        // 多轮输入循环
        std::string input_text;
        while (true) {
            std::cout << "请输入文本: ";
            if (!std::getline(std::cin, input_text)) {
                break; // 处理输入错误
            }

            // 退出条件
            if (input_text == "exit") {
                std::cout << "程序退出中...\n";
                break;
            }

            // 跳过空输入
            if (input_text.empty()) {
                continue;
            }

            // 处理单轮文本
            std::vector<std::string> texts = {input_text};
            std::vector<std::vector<float>> embeddings = model.encode_batch(texts);

            // 输出结果
            std::cout << "embedding shape: " << embeddings[0].size() << std::endl;
            for (float val : embeddings[0]) {
                std::cout << val << " ";
            }
            std::cout << "\n\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
