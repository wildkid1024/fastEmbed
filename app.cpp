#include "sentence_transformers.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    try {
        momdel_path = ""
        SentenceTransformer model("path/to/your/model");
        std::vector<std::string> texts = {"This is a test sentence.", "Another example sentence."};
        std::vector<std::vector<float>> embeddings = model.encode_batch(texts);

        for (const auto& embedding : embeddings) {
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
