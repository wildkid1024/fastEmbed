#ifndef SENTENCE_TRANSFORMERS_H
#define SENTENCE_TRANSFORMERS_H

#include <vector>
#include <string>
#include <memory>

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
    std::vector<float> encode(const std::string& text);

    // 批量生成嵌入
    std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts);

    // 获取嵌入维度
    size_t get_embedding_dimension() const;

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
