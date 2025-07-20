#include "sentence_transformers.h"
#include "sentence_transformers_impl.h"

// Assume you have a way to get tokenizer_path and config_path from model_path
// For example, by appending fixed strings or using a specific naming convention
SentenceTransformer::SentenceTransformer(const std::string& model_path)
    : impl(std::make_unique<SentenceTransformerImpl>(
        model_path,
        model_path + "/tokenizer.model",  // Replace with the actual way to get tokenizer_path
        model_path + "/tokenizer.json"       // Replace with the actual way to get config_path
    )) {}

SentenceTransformer::~SentenceTransformer() = default;

std::vector<float> SentenceTransformer::encode(const std::string& text) {
    return impl->encode(text);
}

std::vector<std::vector<float>> SentenceTransformer::encode_batch(const std::vector<std::string>& texts) {
    return impl->encode_batch(texts);
}

size_t SentenceTransformer::get_embedding_dimension() const {
    return impl->get_embedding_dimension();
}

SentenceTransformer::SentenceTransformer(SentenceTransformer&&) noexcept = default;

SentenceTransformer& SentenceTransformer::operator=(SentenceTransformer&&) noexcept = default;