#include "sentence_transformers.h"
#include <iostream>
#include <vector>
#include <string>

namespace {

void print_embedding(const std::vector<float>& embedding) {
    std::cout << "embedding shape: " << embedding.size() << std::endl;
    for (float val : embedding) {
        std::cout << val << " ";
    }
    std::cout << "\n\n";
}

}  // namespace

// 支持文本、提示词与图像三类输入
int main(int argc, char* argv[]) {
    try {
        std::string model_path;
        std::string prompt_name;
        std::string image_path;

        // 解析参数：位置参数为 <model_path> [prompt_name]，也支持 --image / --prompt
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--image" && i + 1 < argc) {
                image_path = argv[++i];
            } else if (arg == "--prompt" && i + 1 < argc) {
                prompt_name = argv[++i];
            } else if (model_path.empty()) {
                model_path = arg;
            } else if (prompt_name.empty()) {
                prompt_name = arg;
            }
        }

        if (model_path.empty()) {
            std::cerr << "Usage: " << argv[0] << " <model_path> [prompt_name] [--image <image_path>]"
                      << std::endl;
            return 1;
        }

        // 加载模型（仅加载一次）
        SentenceTransformer model(model_path);

        // 单次图像嵌入模式
        if (!image_path.empty()) {
            if (!model.supports_image()) {
                std::cerr << "当前模型不支持图像编码（需要带视觉塔的 EmbeddingGemma2）" << std::endl;
                return 1;
            }
            std::vector<float> embedding = model.encode_image(image_path);
            std::cout << "图像: " << image_path << std::endl;
            print_embedding(embedding);
            return 0;
        }

        std::cout << "模型加载成功，可开始输入文本（输入 exit 退出，输入 @图片路径 编码图像）...\n"
                  << std::endl;

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

            // "@路径" 直接编码图像
            if (input_text[0] == '@') {
                if (!model.supports_image()) {
                    std::cerr << "当前模型不支持图像编码" << std::endl;
                    continue;
                }
                std::vector<float> embedding = model.encode_image(input_text.substr(1));
                print_embedding(embedding);
                continue;
            }

            // 处理单轮文本
            std::vector<std::string> texts = {input_text};
            std::vector<std::vector<float>> embeddings = model.encode_batch(texts, prompt_name);
            print_embedding(embeddings[0]);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
