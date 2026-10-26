#ifndef SENTENCE_TRANSFORMERS_H
#define SENTENCE_TRANSFORMERS_H

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

// 前向声明实现类
class SentenceTransformerImpl;

// SentenceTransformer 类接口
class SentenceTransformer {
public:
    // 构造函数，加载模型
    explicit SentenceTransformer(const std::string& model_path);

    // 析构函数
    ~SentenceTransformer();

    // 生成单个文本的嵌入
    // prompt_name 非空时，使用模型内置的提示词前缀（如 EmbeddingGemma 的 "query" / "document"）
    std::vector<float> encode(const std::string& text, const std::string& prompt_name = "");

    // 批量生成嵌入
    std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts,
                                                 const std::string& prompt_name = "");

    // 获取嵌入维度
    size_t get_embedding_dimension() const;

    // 图像嵌入（需要带视觉塔的模型，如 EmbeddingGemma2）
    std::vector<float> encode_image(const std::string& image_path);
    std::vector<float> encode_image_bytes(const std::vector<uint8_t>& image_bytes);

    // 当前模型是否支持图像编码
    bool supports_image() const;

    // 获取模型内置的提示词名称列表
    std::vector<std::string> get_prompt_names() const;

    // 禁用拷贝
    SentenceTransformer(const SentenceTransformer&) = delete;
    SentenceTransformer& operator=(const SentenceTransformer&) = delete;

    // 允许移动
    SentenceTransformer(SentenceTransformer&&) noexcept;
    SentenceTransformer& operator=(SentenceTransformer&&) noexcept;

private:
    // 使用智能指针管理实现类
    std::unique_ptr<SentenceTransformerImpl> impl;
};

#endif // SENTENCE_TRANSFORMERS_H
