#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Gemma4 图像预处理：解码 -> 等比缩放 -> /255 -> patchify -> 位置 id
// 对齐 transformers 的 Gemma4ImageProcessor（patch_size=16, pooling_kernel_size=3, max_soft_tokens=280）

struct GemmaImageInput {
    // [num_patches, patch_size * patch_size * 3]，patch 内顺序为 (c, py, px)
    std::vector<float> pixel_values;
    // [num_patches * 2]，每组为 (x, y)，padding patch 为 (-1, -1)
    std::vector<int32_t> position_ids;
    // [num_patches]，1=有效 patch，0=padding
    std::vector<uint8_t> valid;
    size_t num_patches = 0;       // 含 padding 的总数（= max_patches）
    size_t num_real_patches = 0;  // 有效 patch 数
    size_t patch_grid_h = 0;
    size_t patch_grid_w = 0;
    std::vector<size_t> resized_hw;  // {h, w}，便于日志/调试
};

// 读取 PNG/JPEG 等图片为 RGB8（失败抛异常）
std::vector<uint8_t> decode_image_rgb(const std::string& path, int* height, int* width);

// 从内存中的图片字节解码为 RGB8（失败抛异常）
std::vector<uint8_t> decode_image_rgb_from_memory(const uint8_t* data, size_t size, int* height, int* width);

// 按 Gemma4 规则计算等比缩放后的目标尺寸（height/width 均向下取整到 pooling*patch 的倍数）
void gemma_target_size(int height, int width, int patch_size, int max_patches,
                       int pooling_kernel_size, int* target_h, int* target_w);

// 双三次（Keys, a=-0.5）缩放，带抗锯齿，对齐 torchvision/PIL 的 bicubic
std::vector<uint8_t> resize_bicubic_rgb8(const std::vector<uint8_t>& src, int src_h, int src_w,
                                         int dst_h, int dst_w);

// 完整预处理：RGB8 原图 -> GemmaImageInput
GemmaImageInput preprocess_gemma_image(const std::vector<uint8_t>& rgb, int height, int width,
                                       int patch_size = 16, int max_patches = 2520,
                                       int pooling_kernel_size = 3);

// 便捷入口：文件路径 -> GemmaImageInput
GemmaImageInput load_and_preprocess_image(const std::string& path,
                                          int patch_size = 16, int max_patches = 2520,
                                          int pooling_kernel_size = 3);

// 宽松的 base64 解码（忽略空白与换行，支持 data URI 前缀）
std::vector<uint8_t> base64_decode(const std::string& input);
