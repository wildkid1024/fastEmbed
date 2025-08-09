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

        // 遍历所有张量
        for (const auto& [name, tensor_info] : header.items()) {
            if (name == "__metadata__") {
                continue;
            }
            // 检查数据类型是否为 float32
            if (tensor_info["dtype"] != "F32") {
                continue;
                throw std::runtime_error("仅支持 float32 数据类型，当前张量 " + name + " 类型为 " + tensor_info["dtype"].get<std::string>());
            }

            // 计算元素数量
            size_t num_elements = 1;
            for (const auto& dim : tensor_info["shape"]) {
                num_elements *= dim.get<size_t>();
            }

            // 定位到数据块位置
            size_t data_offset = tensor_info["data_offsets"][0].get<size_t>();
            file.seekg(sizeof(uint64_t) + header_length + data_offset, std::ios::beg);

            // 读取数据
            std::vector<float> data(num_elements);
            file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));

            weights[name] = std::move(data);
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "加载 safetensors 文件失败: " << e.what() << std::endl;
        return false;
    }
}
