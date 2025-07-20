#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <string>
#include <unordered_map>
#include <vector>

// 声明加载 safetensors 文件的函数
bool load_single_safetensors_file(const std::string& file_path, 
                                 std::unordered_map<std::string, std::vector<float>>& weights);

#endif // SAFETENSORS_H
