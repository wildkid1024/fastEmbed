#include "sentence_transformers.h"
#include "sentence_transformers_impl.h"

// Assume you have a way to get tokenizer_path and config_path from model_path
// For example, by appending fixed strings or using a specific naming convention
SentenceTransformer::SentenceTransformer(const std::string& model_path)
    : impl(std::make_unique<SentenceTransformerImpl>(
        model_path,
        model_path
    )) {}

SentenceTransformer::~SentenceTransformer() = default;

std::vector<float> SentenceTransformer::encode(const std::string& text, const std::string& prompt_name) {
    return impl->encode(text, prompt_name);
}

std::vector<std::vector<float>> SentenceTransformer::encode_batch(const std::vector<std::string>& texts,
                                                                  const std::string& prompt_name) {
    return impl->encode_batch(texts, prompt_name);
}

size_t SentenceTransformer::get_embedding_dimension() const {
    return impl->get_embedding_dimension();
}

std::vector<float> SentenceTransformer::encode_image(const std::string& image_path) {
    return impl->encode_image(image_path);
}

std::vector<float> SentenceTransformer::encode_image_bytes(const std::vector<uint8_t>& image_bytes) {
    return impl->encode_image_bytes(image_bytes);
}

bool SentenceTransformer::supports_image() const {
    return impl->supports_image();
}

std::vector<std::string> SentenceTransformer::get_prompt_names() const {
    return impl->get_prompt_names();
}

SentenceTransformer::SentenceTransformer(SentenceTransformer&&) noexcept = default;

SentenceTransformer& SentenceTransformer::operator=(SentenceTransformer&&) noexcept = default;
