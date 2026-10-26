#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <string>
#include <unordered_map>
#include <vector>

// 声明加载 safetensors 文件的函数
// skip_prefixes: 名称以任一前缀开头的张量会被跳过（用于只加载需要的子模型，如仅文本塔）
bool load_single_safetensors_file(const std::string& file_path,
                                 std::unordered_map<std::string, std::vector<float>>& weights,
                                 const std::vector<std::string>& skip_prefixes = {});

#endif // SAFETENSORS_H
