#include "safetensors.h"
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <cmath>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

using json = nlohmann::json;

// 读取文件头部的长度信息
uint64_t read_header_length(std::ifstream& file) {
    uint64_t header_length;
    file.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
    return header_length;
}

// 将bf16转换为f32
float bf16_to_f32(uint16_t bf16_value) {
    // 将bf16值扩展到f32格式（将bf16的16位放在高16位）
    uint32_t f32_bits = static_cast<uint32_t>(bf16_value) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// 加载 safetensors 文件
bool load_single_safetensors_file(const std::string& file_path, std::unordered_map<std::string, std::vector<float>>& weights) {
    try {
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("无法打开 safetensors 文件: " + file_path);
        }

        // 读取头部长度
        uint64_t header_length = read_header_length(file);

        // 读取头部 JSON 数据
        std::string header_json(header_length, '\0');
        file.read(&header_json[0], header_length);

        // 解析 JSON
        json header = json::parse(header_json);

        std::cout << "模型权重名称列表:" << std::endl;
        // 遍历所有张量
        for (const auto& [name, tensor_info] : header.items()) {
            if (name == "__metadata__") {
                continue;
            }
            
            std::string dtype = tensor_info["dtype"];
            
            // 计算元素数量
            size_t num_elements = 1;
            for (const auto& dim : tensor_info["shape"]) {
                num_elements *= dim.get<size_t>();
            }

            // 定位到数据块位置
            size_t data_offset = tensor_info["data_offsets"][0].get<size_t>();
            file.seekg(sizeof(uint64_t) + header_length + data_offset, std::ios::beg);

            if (dtype == "BF16") {
                // 处理BF16数据
                std::vector<uint16_t> bf16_data(num_elements);
                file.read(reinterpret_cast<char*>(bf16_data.data()), num_elements * sizeof(uint16_t));

                // 转换BF16到F32
                std::vector<float> f32_data(num_elements);
                for (size_t i = 0; i < num_elements; ++i) {
                    f32_data[i] = bf16_to_f32(bf16_data[i]);
                }

                weights[name] = std::move(f32_data);
            } else if (dtype == "F32") {
                // 处理F32数据
                std::vector<float> f32_data(num_elements);
                file.read(reinterpret_cast<char*>(f32_data.data()), num_elements * sizeof(float));

                weights[name] = std::move(f32_data);
            } else {
                std::cout << "  - 跳过数据类型: " << dtype << " (非BF16或F32)" << std::endl;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "加载 safetensors 文件失败: " << e.what() << std::endl;
        return false;
    }
}
