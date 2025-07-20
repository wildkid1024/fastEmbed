#include "sentence_transformers.h"
#include <stdexcept>

// 包含实现类的头文件
#include "sentence_transformers_impl.h"

// 构造函数，加载模型
SentenceTransformer::SentenceTransformer(const std::string& model_path) {
    try {
        impl = std::make_unique<SentenceTransformerImpl>(model_path);
    } catch (const std::exception& e) {
        throw std::runtime_error("加载模型失败: " + std::string(e.what()));
    }
}

// 析构函数
SentenceTransformer::~SentenceTransformer() = default;

// 生成单个文本的嵌入
std::vector<float> SentenceTransformer::encode(const std::string& text) {
    if (!impl) {
        throw std::runtime_error("模型未正确加载");
    }
    return impl->encode(text);
}

// 批量生成嵌入
std::vector<std::vector<float>> SentenceTransformer::encode_batch(const std::vector<std::string>& texts) {
    if (!impl) {
        throw std::runtime_error("模型未正确加载");
    }
    return impl->encode_batch(texts);
}

// 获取嵌入维度
size_t SentenceTransformer::get_embedding_dimension() const {
    if (!impl) {
        throw std::runtime_error("模型未正确加载");
    }
    return impl->get_embedding_dimension();
}

// 移动构造函数
SentenceTransformer::SentenceTransformer(SentenceTransformer&& other) noexcept = default;

// 移动赋值运算符
SentenceTransformer& SentenceTransformer::operator=(SentenceTransformer&& other) noexcept = default;
