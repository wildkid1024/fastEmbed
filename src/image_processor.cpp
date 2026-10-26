#include "image_processor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_WEBP
#include "stb_image.h"

namespace {

// Keys 三次卷积核（a = -0.5），与 PIL/torchvision 的 BICUBIC 一致
inline double bicubic_kernel(double x) {
    x = std::fabs(x);
    if (x < 1.0) {
        return (1.5 * x - 2.5) * x * x + 1.0;
    }
    if (x < 2.0) {
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    }
    return 0.0;
}

// PIL 风格的权重预计算：包含 antialias 时的 filter scale 与归一化
struct ResampleCoeffs {
    std::vector<int> bounds;      // [dst + 1]，每个输出位置在 weights 中的区间
    std::vector<int> src_starts;  // [dst]，每个输出位置对应的第一个源下标
    std::vector<double> weights;  // 展平后的权重
};

ResampleCoeffs compute_coeffs(int src_size, int dst_size) {
    ResampleCoeffs coeffs;
    if (src_size <= 0 || dst_size <= 0) {
        return coeffs;
    }
    const double scale = static_cast<double>(src_size) / static_cast<double>(dst_size);
    const double filter_scale = std::max(1.0, scale);
    const double support = 2.0 * filter_scale;  // bicubic 的 support 为 2.0
    const double inv_filter_scale = 1.0 / filter_scale;

    coeffs.bounds.resize(static_cast<size_t>(dst_size) + 1);
    coeffs.src_starts.resize(static_cast<size_t>(dst_size), 0);
    coeffs.weights.reserve(static_cast<size_t>(dst_size) * 4);

    for (int dst = 0; dst < dst_size; ++dst) {
        coeffs.bounds[dst] = static_cast<int>(coeffs.weights.size());
        const double center = (dst + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        int xmax = static_cast<int>(center + support + 0.5);
        xmin = std::max(0, std::min(xmin, src_size));
        xmax = std::max(0, std::min(xmax, src_size));
        if (xmax <= xmin) {
            // 退化情况：至少取最近的源像素
            xmin = std::min(std::max(0, static_cast<int>(center)), src_size - 1);
            xmax = xmin + 1;
        }
        coeffs.src_starts[dst] = xmin;

        double total = 0.0;
        const int start = static_cast<int>(coeffs.weights.size());
        for (int x = xmin; x < xmax; ++x) {
            const double w = bicubic_kernel((x - center + 0.5) * inv_filter_scale);
            coeffs.weights.push_back(w);
            total += w;
        }
        if (total != 0.0) {
            for (size_t i = static_cast<size_t>(start); i < coeffs.weights.size(); ++i) {
                coeffs.weights[i] /= total;
            }
        }
    }
    coeffs.bounds[static_cast<size_t>(dst_size)] = static_cast<int>(coeffs.weights.size());
    return coeffs;
}

// 先水平后垂直，两趟都在 float 中计算，最后统一取整（与 torchvision 一致）
std::vector<uint8_t> resample_rgb8(const std::vector<uint8_t>& src, int src_h, int src_w,
                                   int dst_h, int dst_w) {
    std::vector<float> tmp(static_cast<size_t>(dst_w) * src_h * 3);
    const ResampleCoeffs cx = compute_coeffs(src_w, dst_w);

    for (int y = 0; y < src_h; ++y) {
        const uint8_t* row = src.data() + static_cast<size_t>(y) * src_w * 3;
        for (int x = 0; x < dst_w; ++x) {
            const int begin = cx.bounds[x];
            const int end = cx.bounds[x + 1];
            const int src_start = cx.src_starts[x];
            double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;
            for (int k = begin; k < end; ++k) {
                const double w = cx.weights[k];
                const uint8_t* px = row + static_cast<size_t>(src_start + (k - begin)) * 3;
                acc0 += w * px[0];
                acc1 += w * px[1];
                acc2 += w * px[2];
            }
            float* out = tmp.data() + (static_cast<size_t>(y) * dst_w + x) * 3;
            out[0] = static_cast<float>(acc0);
            out[1] = static_cast<float>(acc1);
            out[2] = static_cast<float>(acc2);
        }
    }

    std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * dst_h * 3);
    const ResampleCoeffs cy = compute_coeffs(src_h, dst_h);
    for (int y = 0; y < dst_h; ++y) {
        const int begin = cy.bounds[y];
        const int end = cy.bounds[y + 1];
        const int src_start = cy.src_starts[y];
        for (int x = 0; x < dst_w; ++x) {
            double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;
            for (int k = begin; k < end; ++k) {
                const double w = cy.weights[k];
                const float* px = tmp.data() + (static_cast<size_t>(src_start + (k - begin)) * dst_w + x) * 3;
                acc0 += w * px[0];
                acc1 += w * px[1];
                acc2 += w * px[2];
            }
            uint8_t* out = dst.data() + (static_cast<size_t>(y) * dst_w + x) * 3;
            // 四舍五入并 clamp 到 [0, 255]
            for (int c = 0; c < 3; ++c) {
                const double v = (c == 0) ? acc0 : (c == 1 ? acc1 : acc2);
                const double rounded = std::floor(v + 0.5);
                out[c] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, rounded)));
            }
        }
    }
    return dst;
}

}  // namespace

std::vector<uint8_t> decode_image_rgb(const std::string& path, int* height, int* width) {
    int h = 0, w = 0, channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (data == nullptr) {
        throw std::runtime_error("无法解码图片: " + path + " (" + std::string(stbi_failure_reason()) + ")");
    }
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    std::memcpy(rgb.data(), data, rgb.size());
    stbi_image_free(data);
    if (height != nullptr) *height = h;
    if (width != nullptr) *width = w;
    return rgb;
}

std::vector<uint8_t> decode_image_rgb_from_memory(const uint8_t* data, size_t size, int* height, int* width) {
    int h = 0, w = 0, channels = 0;
    unsigned char* pixels = stbi_load_from_memory(data, static_cast<int>(size), &w, &h, &channels, 3);
    if (pixels == nullptr) {
        throw std::runtime_error(std::string("无法从内存解码图片 (") + stbi_failure_reason() + ")");
    }
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    std::memcpy(rgb.data(), pixels, rgb.size());
    stbi_image_free(pixels);
    if (height != nullptr) *height = h;
    if (width != nullptr) *width = w;
    return rgb;
}

void gemma_target_size(int height, int width, int patch_size, int max_patches,
                       int pooling_kernel_size, int* target_h, int* target_w) {
    const double total_px = static_cast<double>(height) * static_cast<double>(width);
    const double target_px = static_cast<double>(max_patches) * patch_size * patch_size;
    const double factor = std::sqrt(target_px / total_px);
    const int side_mult = pooling_kernel_size * patch_size;

    int th = static_cast<int>(std::floor(factor * height / side_mult)) * side_mult;
    int tw = static_cast<int>(std::floor(factor * width / side_mult)) * side_mult;

    if (th == 0 && tw == 0) {
        throw std::runtime_error("图像过小，等比缩放后尺寸为 0（需要至少一个 pooling*patch 的边长）");
    }
    const int max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if (th == 0) {
        th = side_mult;
        tw = std::min(static_cast<int>(std::floor(static_cast<double>(width) / height)) * side_mult,
                      max_side_length);
        tw = std::max(tw, side_mult);
    } else if (tw == 0) {
        tw = side_mult;
        th = std::min(static_cast<int>(std::floor(static_cast<double>(height) / width)) * side_mult,
                      max_side_length);
        th = std::max(th, side_mult);
    }
    if (static_cast<long long>(th) * tw > static_cast<long long>(target_px)) {
        throw std::runtime_error("等比缩放后 patch 数超过上限");
    }
    if (target_h != nullptr) *target_h = th;
    if (target_w != nullptr) *target_w = tw;
}

std::vector<uint8_t> resize_bicubic_rgb8(const std::vector<uint8_t>& src, int src_h, int src_w,
                                         int dst_h, int dst_w) {
    if (src.size() != static_cast<size_t>(src_h) * src_w * 3) {
        throw std::invalid_argument("resize_bicubic_rgb8: 输入尺寸不匹配");
    }
    if (src_h == dst_h && src_w == dst_w) {
        return src;
    }
    return resample_rgb8(src, src_h, src_w, dst_h, dst_w);
}

GemmaImageInput preprocess_gemma_image(const std::vector<uint8_t>& rgb, int height, int width,
                                       int patch_size, int max_patches, int pooling_kernel_size) {
    if (rgb.size() != static_cast<size_t>(height) * width * 3) {
        throw std::invalid_argument("preprocess_gemma_image: 像素数据尺寸不匹配");
    }

    int target_h = 0, target_w = 0;
    gemma_target_size(height, width, patch_size, max_patches, pooling_kernel_size, &target_h, &target_w);
    std::vector<uint8_t> resized = resize_bicubic_rgb8(rgb, height, width, target_h, target_w);

    const size_t patch_h = static_cast<size_t>(target_h) / patch_size;
    const size_t patch_w = static_cast<size_t>(target_w) / patch_size;
    const size_t real_patches = patch_h * patch_w;
    if (real_patches == 0 || real_patches > static_cast<size_t>(max_patches)) {
        throw std::runtime_error("preprocess_gemma_image: patch 数非法");
    }

    const size_t patch_pixels = static_cast<size_t>(patch_size) * patch_size * 3;
    GemmaImageInput out;
    out.num_patches = static_cast<size_t>(max_patches);
    out.num_real_patches = real_patches;
    out.patch_grid_h = patch_h;
    out.patch_grid_w = patch_w;
    out.resized_hw = {static_cast<size_t>(target_h), static_cast<size_t>(target_w)};
    out.pixel_values.assign(out.num_patches * patch_pixels, 0.0f);
    out.position_ids.assign(out.num_patches * 2, -1);
    out.valid.assign(out.num_patches, 0);

    // patch 内按 (c, py, px) 展开，与 convert_image_to_patches 的 permute(1,3,2,4,0) 一致
    for (size_t py = 0; py < patch_h; ++py) {
        for (size_t px = 0; px < patch_w; ++px) {
            const size_t patch_idx = py * patch_w + px;
            float* dst = out.pixel_values.data() + patch_idx * patch_pixels;
            for (size_t c = 0; c < 3; ++c) {
                for (size_t dy = 0; dy < static_cast<size_t>(patch_size); ++dy) {
                    const size_t src_y = py * patch_size + dy;
                    for (size_t dx = 0; dx < static_cast<size_t>(patch_size); ++dx) {
                        const size_t src_x = px * patch_size + dx;
                        const size_t src_idx = (src_y * static_cast<size_t>(target_w) + src_x) * 3 + c;
                        const size_t dst_idx = (c * patch_size * patch_size) + dy * patch_size + dx;
                        dst[dst_idx] = static_cast<float>(resized[src_idx]) / 255.0f;
                    }
                }
            }
            out.position_ids[patch_idx * 2] = static_cast<int32_t>(px);
            out.position_ids[patch_idx * 2 + 1] = static_cast<int32_t>(py);
            out.valid[patch_idx] = 1;
        }
    }
    return out;
}

GemmaImageInput load_and_preprocess_image(const std::string& path, int patch_size, int max_patches,
                                          int pooling_kernel_size) {
    int h = 0, w = 0;
    std::vector<uint8_t> rgb = decode_image_rgb(path, &h, &w);
    return preprocess_gemma_image(rgb, h, w, patch_size, max_patches, pooling_kernel_size);
}

std::vector<uint8_t> base64_decode(const std::string& input) {
    // 去掉 data URI 前缀
    size_t start = 0;
    const size_t comma = input.find(',');
    if (input.rfind("data:", 0) == 0 && comma != std::string::npos) {
        start = comma + 1;
    }

    auto value_of = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+' || c == '-') return 62;
        if (c == '/' || c == '_') return 63;
        return -1;
    };

    std::vector<uint8_t> out;
    out.reserve(input.size() * 3 / 4);
    int buffer = 0;
    int bits = 0;
    for (size_t i = start; i < input.size(); ++i) {
        const char c = input[i];
        if (c == '=') break;
        const int v = value_of(c);
        if (v < 0) continue;  // 跳过空白/换行等
        buffer = (buffer << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buffer >> bits) & 0xFF));
        }
    }
    return out;
}
