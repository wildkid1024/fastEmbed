#include "safetensors.h"
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <cmath>

// 加载 safetensors 文件
bool load_single_safetensors_file(const std::string& file_path, std::unordered_map<std::string, std::vector<float>>& weights) {
    try {
        // 使用 safetensors-cpp 库加载文件
        safetensors::SafeTensors st = safetensors::load(file_path);

        // 遍历所有张量并存储
        for (const auto& [name, tensor] : st.tensors()) {
            // 将张量数据复制到我们的存储结构中
            std::vector<float> data(tensor.num_elements());
            std::memcpy(data.data(), tensor.data<float>(), 
                       tensor.num_elements() * sizeof(float));
            weights[name] = std::move(data);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "加载 safetensors 文件失败: " << e.what() << std::endl;
        return false;
    }
}
