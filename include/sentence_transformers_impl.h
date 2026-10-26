#pragma once
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <sentencepiece_processor.h>
#include <nlohmann/json.hpp>
#include "tokenizer.h"  // 引入 tokenizer 头文件
#include "image_processor.h"  // 图像预处理（Gemma4 patchify）

// 移除未使用的CPU头文件
#include "ops/cpu/cpu_attention_ops.h"
#include "ops/cpu/cpu_matrix_ops.h"

#include "ops/cuda/cuda_matrix_ops.h"
#include "ops/cuda/cuda_attention_ops.cuh"

using json = nlohmann::json;

// 模型类型枚举
enum class ModelType {
    BGE,
    BGE_M3,
    QWEN3,  // 添加Qwen3模型类型
    EMBEDDING_GEMMA2  // EmbeddingGemma 2（Gemma 编码器架构 + PLE）
};

class SentenceTransformerImpl {
private:
    // 支持四种分词器
    ModelType model_type;
    std::unique_ptr<BGETokenizer> bge_tokenizer;
    std::unique_ptr<XLMRobertaTokenizer> xlmr_tokenizer;
    std::unique_ptr<Qwen3Tokenizer> qwen3_tokenizer; 
    std::unique_ptr<GemmaTokenizer> gemma_tokenizer;
    
    std::unordered_map<std::string, int64_t> vocab;
    std::unordered_map<std::string, std::vector<float>> weights;
    size_t embedding_dim;        // 隐藏层维度（Transformer hidden_size）
    size_t embedding_output_dim; // 最终输出维度（EmbeddingGemma2 的 embedding_dim）
    nlohmann::json text_config;  // 文本塔配置（多模态模型的 text_config，其它模型等于 config）
    std::vector<std::string> layer_types;   // 逐层注意力类型（full_attention / sliding_attention）
    std::unordered_map<std::string, std::string> prompts;  // 提示词前缀（config_sentence_transformers.json）

    // ===== 视觉塔（图像编码器）相关配置，仅 EmbeddingGemma2 使用 =====
    nlohmann::json vision_config;
    bool has_vision_tower = false;      // 权重中是否存在 vision_tower.*
    size_t vision_hidden_size = 0;
    size_t vision_num_layers = 0;
    size_t vision_num_heads = 0;
    size_t vision_num_kv_heads = 0;
    size_t vision_head_dim = 0;
    size_t vision_intermediate_size = 0;
    size_t vision_patch_size = 0;
    size_t vision_max_patches = 0;      // max_soft_tokens * pooling^2
    size_t vision_pooling_kernel = 0;
    float vision_rope_theta = 100.0f;
    float vision_eps = 1e-6f;
    size_t image_token_id = 0;
    size_t boi_token_id = 0;
    size_t eoi_token_id = 0;
    // 注释掉未使用的CPU实例
    CPUAttentionOps cpu_attention_ops; 
    CPUMatrixOps cpu_matrix_ops; 
    // 添加CUDA操作类实例
    CUDAMatrixOps cuda_matrix_ops;
    CUDAAttentionOps cuda_attention_ops;
    nlohmann::json config;  // 添加 JSON 配置私有成员

    // Private helper functions
    // void load_vocab(const std::string& config_path);
    std::vector<int32_t> tokenize(const std::string& text);
    std::vector<float> embedding_layer(const std::vector<int32_t>& tokens);
    std::vector<float> embedding_layer_batch(const std::vector<std::vector<int32_t>>& tokens_list);
    std::vector<float> add_position_encoding(const std::vector<float>& input, size_t seq_len, size_t embedding_dim);
    std::vector<float> multi_head_attention(
        const std::vector<float>& input,
        const std::vector<float>& weight_q,
        const std::vector<float>& weight_k,
        const std::vector<float>& weight_v,
        const std::vector<float>& query_bias,  // New: query bias
        const std::vector<float>& key_bias,    // New: key bias
        const std::vector<float>& value_bias,  // New: value bias
        size_t num_heads,
        size_t embedding_dim
    );
    // std::vector<float> feed_forward_network(const std::vector<float>& input, const std::vector<float>& weight_ff_1, const std::vector<float>& weight_ff_2, size_t embedding_dim, size_t intermediate_dim);
    std::vector<float> feed_forward_network(const std::vector<float>& input, 
                                                                  const std::vector<float>& weight_ff_1, 
                                                                  const std::vector<float>& bias_ff_1, 
                                                                  const std::vector<float>& weight_ff_2, 
                                                                  const std::vector<float>& bias_ff_2, 
                                                                  const std::vector<float>& ln_gamma,  // 新增LayerNorm参数
                                                                  const std::vector<float>& ln_beta,   // 新增LayerNorm参数
                                                                  size_t embedding_dim, 
                                                                  size_t intermediate_dim,
                                                                  float dropout_prob = 0.1f);
    std::vector<float> bge_forward(const std::vector<int32_t>& tokens);
    std::vector<float> qwen3_forward(const std::vector<int32_t>& tokens);  // 添加Qwen3前向传播函数
    // EmbeddingGemma 2 前向传播（双向注意力编码器 + PLE + 均值池化）
    std::vector<float> embeddinggemma2_forward(const std::vector<int32_t>& tokens);
    std::vector<std::vector<int32_t>> tokenize_batch(const std::vector<std::string>& texts);
    std::vector<std::vector<float>> qwen3_forward_batch(const std::vector<std::vector<int32_t>>& tokens_list);
    std::vector<std::vector<float>> embeddinggemma2_forward_batch(const std::vector<std::vector<int32_t>>& tokens_list);
    // 已经拿到 inputs_embeds 时的语言模型前向（图像 soft token 会先拼进来）
    std::vector<float> embeddinggemma2_embeds_forward(const std::vector<float>& inputs_embeds, size_t seq_len);
    // 视觉塔前向：patch -> 16 层编码 -> 3x3 池化 -> 投影到文本空间，输出 [num_soft_tokens, text_hidden]
    std::vector<float> embeddinggemma2_vision_forward(const GemmaImageInput& image);
    // 图像 -> 与文本同空间的嵌入（视觉 soft token 拼进序列后过语言模型 + 均值池化 + L2）
    std::vector<float> embeddinggemma2_image_embedding(const GemmaImageInput& image);
    // 为文本添加提示词前缀（EmbeddingGemma 系列的 query/document 前缀）
    std::string apply_prompt(const std::string& text, const std::string& prompt_name) const;

public:
    // 支持三种构造函数，一种用于传统BGE模型，一种用于BGE-M3模型，一种用于Qwen3模型    SentenceTransformerImpl(const std::string& model_path, const std::string& tokenizer_path);
    SentenceTransformerImpl(const std::string& model_path, const std::string& tokenizer_path);
     std::vector<float> encode(const std::string& text, const std::string& prompt_name = "");
    std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts, const std::string& prompt_name = "");
    // 图像嵌入（当前支持 EmbeddingGemma2 的视觉塔），输出与文本嵌入同维度同空间
    std::vector<float> encode_image(const std::string& image_path);
    // 从内存字节（PNG/JPEG 等）编码图像，便于 HTTP 直接传 base64
    std::vector<float> encode_image_bytes(const std::vector<uint8_t>& image_bytes);
    // 是否支持图像编码
    bool supports_image() const;
    size_t get_embedding_dimension() const;
    // 模型内置的提示词名称列表（如 query / document / STS ...）
    std::vector<std::string> get_prompt_names() const;
};