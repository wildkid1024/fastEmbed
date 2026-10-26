#include "sentence_transformers_impl.h"
#include <cstdio>
#include <nlohmann/json.hpp>  // 添加缺失的头文件
#include <safetensors.h>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include <vector>  // 添加缺失的vector头文件
#include <filesystem>  // 添加filesystem头文件以使用exists函数
#include <chrono>  // 添加chrono头文件用于计时
#include <algorithm>  // std::sort


// 使用新的 tokenizer 进行分词
std::vector<int32_t> SentenceTransformerImpl::tokenize(const std::string& text) {
    std::vector<int> token_ids;
    if (model_type == ModelType::EMBEDDING_GEMMA2 && gemma_tokenizer) {
        // Gemma 使用 SentencePiece，句首添加 <bos>
        auto ids = gemma_tokenizer->encode_text(text);
        token_ids.assign(ids.begin(), ids.end());
    } else if (model_type == ModelType::BGE_M3 && xlmr_tokenizer) {
        // 使用XLMRobertaTokenizer进行分词
        auto tokens = xlmr_tokenizer->tokenize(text);
        token_ids = xlmr_tokenizer->encode(tokens);
    } else if (model_type == ModelType::QWEN3 && qwen3_tokenizer) {
        // 使用Qwen3Tokenizer进行分词
        auto tokens = qwen3_tokenizer->tokenize(text);
        token_ids = qwen3_tokenizer->encode(tokens);
        // Qwen3模型通常使用特殊token，如<s>和</s>，而不是[CLS]和[SEP]
        // 根据模型配置决定是否添加特殊token
        int bos_token_id = qwen3_tokenizer->get_bos_token_id();
        int eos_token_id = qwen3_tokenizer->get_eos_token_id();
        
        if (bos_token_id != -1) {
            token_ids.insert(token_ids.begin(), bos_token_id);
        }
        if (eos_token_id != -1) {
            token_ids.push_back(eos_token_id);
        }
    } else if (bge_tokenizer) {
        // 使用BGETokenizer进行分词
        auto tokens = bge_tokenizer->tokenize(text);
        token_ids = bge_tokenizer->encode(tokens);
        token_ids.insert(token_ids.begin(), 101);
        token_ids.push_back(102);
    } else {
        throw std::runtime_error("未初始化分词器");
    }
    return std::vector<int32_t>(token_ids.begin(), token_ids.end());
}

// 批量分词
std::vector<std::vector<int32_t>> SentenceTransformerImpl::tokenize_batch(const std::vector<std::string>& texts) {
    std::vector<std::vector<int32_t>> tokens_list;
    tokens_list.reserve(texts.size());
    for (const auto& text : texts) {
        tokens_list.push_back(tokenize(text));
    }
    return tokens_list;
}

// 简单的嵌入层前向传播
std::vector<float> SentenceTransformerImpl::embedding_layer(const std::vector<int32_t>& tokens) {
    // printf("DEBUG: embedding_layer called with %zu tokens\n", tokens.size());
    // 调整输出向量大小为 tokens.size() * embedding_dim
    std::vector<float> output(tokens.size() * embedding_dim, 0.0f);
    
    // 根据模型类型选择正确的词嵌入权重键名
    std::string embed_weight_key;
    if (model_type == ModelType::QWEN3) {
        embed_weight_key = "embed_tokens.weight";
    } else if (model_type == ModelType::EMBEDDING_GEMMA2) {
        // EmbeddingGemma2 的文本塔权重使用 language_model. 前缀
        embed_weight_key = "language_model.embed_tokens.weight";
    } else {
        embed_weight_key = "embeddings.word_embeddings.weight";
    }
    
    const std::vector<float>& embedding_weight = weights[embed_weight_key];
    
    // 调试输出：打印词嵌入权重的前几个值和token IDs
    // if (tokens.size() > 0) {
    //     printf("DEBUG: embed_weight_key=%s, size=%zu\n", embed_weight_key.c_str(), embedding_weight.size());
    //     printf("DEBUG: tokens[0]=%d, tokens[1]=%d\n", tokens[0], tokens.size() > 1 ? tokens[1] : -1);
        
    //     // 打印第一个token对应的权重前5个值
    //     if (tokens[0] >= 0 && static_cast<size_t>(tokens[0]) < embedding_weight.size() / embedding_dim) {
    //         size_t start_idx = tokens[0] * embedding_dim;
    //         printf("DEBUG: embed_weight[token0][:5] = [");
    //         for (size_t j = 0; j < 5; ++j) {
    //             printf("%.6f%s", embedding_weight[start_idx + j], j < 4 ? ", " : "");
    //         }
    //         printf("]\n");
    //     }
    // }

    for (size_t i = 0; i < tokens.size(); ++i) {
        int32_t token = tokens[i];
        if (token < 0 || static_cast<size_t>(token) >= embedding_weight.size() / embedding_dim) {
            // Token超出范围，跳过
            continue;
        }
        size_t start_idx = token * embedding_dim;
        // 复制当前 token 对应的嵌入向量到输出中
        for (size_t j = 0; j < embedding_dim; ++j) {
            output[i * embedding_dim + j] = embedding_weight[start_idx + j];
        }
    }
    return output;
}

// 批量嵌入层前向传播
std::vector<float> SentenceTransformerImpl::embedding_layer_batch(const std::vector<std::vector<int32_t>>& tokens_list) {
    // 找到所有序列的最大长度
    size_t max_seq_len = 0;
    size_t batch_size = tokens_list.size();
    for (const auto& tokens : tokens_list) {
        if (tokens.size() > max_seq_len) {
            max_seq_len = tokens.size();
        }
    }
    
    // 创建填充后的批量tokens
    std::vector<int32_t> padded_tokens(batch_size * max_seq_len, 0);
    std::vector<size_t> seq_lens(batch_size);
    
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& tokens = tokens_list[b];
        seq_lens[b] = tokens.size();
        for (size_t t = 0; t < tokens.size(); ++t) {
            padded_tokens[b * max_seq_len + t] = tokens[t];
        }
    }
    
    // 根据模型类型选择正确的词嵌入权重键名
    std::string embed_weight_key;
    if (model_type == ModelType::QWEN3) {
        embed_weight_key = "embed_tokens.weight";
    } else if (model_type == ModelType::EMBEDDING_GEMMA2) {
        embed_weight_key = "language_model.embed_tokens.weight";
    } else {
        embed_weight_key = "embeddings.word_embeddings.weight";
    }
    
    const std::vector<float>& embedding_weight = weights[embed_weight_key];
    
    // 批量嵌入输出
    std::vector<float> output(batch_size * max_seq_len * embedding_dim, 0.0f);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < max_seq_len; ++t) {
            int32_t token = padded_tokens[b * max_seq_len + t];
            if (token < 0 || static_cast<size_t>(token) >= embedding_weight.size() / embedding_dim) {
                // Token超出范围，保持0
                continue;
            }
            size_t weight_start = token * embedding_dim;
            size_t output_start = b * max_seq_len * embedding_dim + t * embedding_dim;
            for (size_t j = 0; j < embedding_dim; ++j) {
                output[output_start + j] = embedding_weight[weight_start + j];
            }
        }
    }
    
    return output;
}

SentenceTransformerImpl::SentenceTransformerImpl(const std::string& model_path, const std::string& tokenizer_path) {
    std::string config_path = model_path + "/config.json";
    
    // 加载config.json到类成员变量
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("无法打开 config.json 文件: " + config_path);
    }
    config_file >> this->config;
    
    // 文本塔配置：多模态模型（如 EmbeddingGemma 2）把语言模型配置放在 text_config 下
    this->text_config = this->config;
    if (this->config.contains("text_config") && this->config["text_config"].is_object()) {
        this->text_config = this->config["text_config"];
    }
    
    // 从config.json中读取model_type并设置模型类型
    std::string model_type_str = "bge";
    if (this->config.contains("model_type")) {
        model_type_str = this->config["model_type"];
    } else if (this->text_config.contains("model_type")) {
        model_type_str = this->text_config["model_type"];
    } else if (this->config.contains("model")) {
        model_type_str = this->config["model"]["model_type"];
    }
    
    // 根据model_type设置枚举值
    if (model_type_str == "bge" || model_type_str == "bert") {
        this->model_type = ModelType::BGE;
    } else if (model_type_str == "bge-m3" || model_type_str == "xlm-roberta") {
        this->model_type = ModelType::BGE_M3;
    } else if (model_type_str == "qwen2" || model_type_str == "qwen3") {  // 添加Qwen2/Qwen3模型类型检测
        this->model_type = ModelType::QWEN3;
    } else if (model_type_str.find("gemma") != std::string::npos) {
        // embedding_gemma2 / embedding_gemma2_text / gemma3_text / gemma2 ...
        this->model_type = ModelType::EMBEDDING_GEMMA2;
    } else {
        std::cerr << "警告: 未知的模型类型 '" << model_type_str << "'，默认使用BGE_M3" << std::endl;
        this->model_type = ModelType::BGE_M3;
    }
    
    // 根据模型类型初始化不同的分词器
    if (this->model_type == ModelType::BGE_M3) {
        // BGE-M3模型使用XLMRobertaTokenizer
        std::string tokenizer_json_path = tokenizer_path + "/tokenizer.json";
        std::string sentencepiece_model_path = tokenizer_path + "/sentencepiece.bpe.model";
        xlmr_tokenizer = std::make_unique<XLMRobertaTokenizer>(tokenizer_json_path, sentencepiece_model_path);
    } else if (this->model_type == ModelType::QWEN3) {
        // Qwen3模型使用Qwen3Tokenizer
        std::string tokenizer_json_path = tokenizer_path + "/tokenizer.json";
        
        // Qwen3使用Hugging Face格式的tokenizer，通常不使用SentencePiece模型文件
        // 检查是否有SentencePiece模型文件
        std::string sentencepiece_model_path = tokenizer_path + "/tokenizer.model";
        std::string vocab_json_path = tokenizer_path + "/vocab.json";
        if (std::filesystem::exists(sentencepiece_model_path)) {
            // 如果存在tokenizer.model，使用SentencePiece格式构造函数
            qwen3_tokenizer = std::make_unique<Qwen3Tokenizer>(tokenizer_json_path, sentencepiece_model_path);
        }
        if (!std::filesystem::exists(vocab_json_path)){
            throw std::runtime_error("无法找到Qwen3的任何tokenizer");
        }
        // 如果存在vocab.json，使用Hugging Face格式构造函数
        qwen3_tokenizer = std::make_unique<Qwen3Tokenizer>(tokenizer_json_path, vocab_json_path);
    } else if (this->model_type == ModelType::EMBEDDING_GEMMA2) {
        // Gemma / EmbeddingGemma 使用 SentencePiece（tokenizer.model）
        std::string sentencepiece_model_path = tokenizer_path + "/tokenizer.model";
        if (!std::filesystem::exists(sentencepiece_model_path)) {
            throw std::runtime_error("无法找到 Gemma 的 SentencePiece 模型: " + sentencepiece_model_path);
        }
        std::string tokenizer_config_path = tokenizer_path + "/tokenizer_config.json";
        gemma_tokenizer = std::make_unique<GemmaTokenizer>(
            sentencepiece_model_path,
            std::filesystem::exists(tokenizer_config_path) ? tokenizer_config_path : ""
        );
    } else {
        // 传统BGE模型使用BGETokenizer
        bge_tokenizer = std::make_unique<BGETokenizer>(tokenizer_path + "/vocab.txt");
        std::string tokenizer_config_path = tokenizer_path + "/tokenizer.json";
    }
    
    // 加载模型内置的提示词（config_sentence_transformers.json 中的 prompts 字段）
    {
        std::string st_config_path = model_path + "/config_sentence_transformers.json";
        if (std::filesystem::exists(st_config_path)) {
            std::ifstream st_file(st_config_path);
            if (st_file.is_open()) {
                try {
                    nlohmann::json st_config;
                    st_file >> st_config;
                    if (st_config.contains("prompts") && st_config["prompts"].is_object()) {
                        for (auto it = st_config["prompts"].begin(); it != st_config["prompts"].end(); ++it) {
                            if (it.value().is_string()) {
                                prompts[it.key()] = it.value().get<std::string>();
                            }
                        }
                        std::cout << "已加载 " << prompts.size() << " 个内置提示词" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "解析 config_sentence_transformers.json 失败: " << e.what() << std::endl;
                }
            }
        }
    }
    
    std::string tensor_path = model_path + "/model.safetensors";
    // 多模态模型保留文本塔与视觉塔（图像编码器），跳过音频塔以省内存
    std::vector<std::string> skip_prefixes;
    if (this->model_type == ModelType::EMBEDDING_GEMMA2) {
        skip_prefixes = {"audio_tower.", "embed_audio."};
    }
    if (!load_single_safetensors_file(tensor_path, weights, skip_prefixes)) {
        throw std::runtime_error("加载模型权重失败");
    }
    
    // 设置嵌入维度
    const size_t vocab_size = this->text_config.contains("vocab_size")
        ? this->text_config["vocab_size"].get<size_t>()
        : this->config["vocab_size"].get<size_t>();
    
    // 根据模型类型查找嵌入层权重
    std::string embedding_weight_name = "embeddings.word_embeddings.weight";
    if (model_type == ModelType::QWEN3) {
        // Qwen3模型使用不同的嵌入层权重名称
        embedding_weight_name = "embed_tokens.weight";
    } else if (model_type == ModelType::EMBEDDING_GEMMA2) {
        // EmbeddingGemma2 的文本塔权重统一使用 language_model. 前缀
        embedding_weight_name = "language_model.embed_tokens.weight";
    }
    if (weights.find(embedding_weight_name) == weights.end()) {
        throw std::runtime_error("未找到嵌入层权重: " + embedding_weight_name);
    }

    embedding_dim = weights[embedding_weight_name].size() / vocab_size;
    embedding_output_dim = embedding_dim;
    
    if (model_type == ModelType::EMBEDDING_GEMMA2) {
        // EmbeddingGemma2 在最后一层后接一个线性投影，把 hidden_size 映射到 embedding_dim
        auto proj_it = weights.find("language_model.embedding_projection.weight");
        if (proj_it != weights.end() && embedding_dim > 0) {
            embedding_output_dim = proj_it->second.size() / embedding_dim;
        }
        
        // 逐层注意力类型（决定 RoPE base 与是否使用滑动窗口）
        if (this->text_config.contains("layer_types") && this->text_config["layer_types"].is_array()) {
            for (const auto& item : this->text_config["layer_types"]) {
                layer_types.push_back(item.get<std::string>());
            }
        }
        std::cout << "EmbeddingGemma2: hidden_size=" << embedding_dim
                  << ", embedding_dim=" << embedding_output_dim
                  << ", layers=" << this->text_config.value("num_hidden_layers", 0)
                  << ", heads=" << this->text_config.value("num_attention_heads", 0)
                  << std::endl;
        
        // ===== 视觉塔（图像编码器）初始化 =====
        has_vision_tower = weights.find("vision_tower.patch_embedder.input_proj.weight") != weights.end() &&
                           weights.find("embed_vision.embedding_projection.weight") != weights.end();
        if (has_vision_tower) {
            if (this->config.contains("vision_config") && this->config["vision_config"].is_object()) {
                vision_config = this->config["vision_config"];
            }
            vision_hidden_size = vision_config.value("hidden_size", 768);
            vision_num_layers = vision_config.value("num_hidden_layers", 0);
            vision_num_heads = vision_config.value("num_attention_heads", 0);
            vision_num_kv_heads = vision_config.value("num_key_value_heads", vision_num_heads);
            vision_intermediate_size = vision_config.value("intermediate_size", 0);
            vision_patch_size = vision_config.value("patch_size", 16);
            vision_pooling_kernel = vision_config.value("pooling_kernel_size", 3);
            vision_head_dim = vision_config.contains("head_dim")
                ? vision_config["head_dim"].get<size_t>()
                : (vision_num_heads > 0 ? vision_hidden_size / vision_num_heads : 0);
            vision_eps = vision_config.value("rms_norm_eps", 1e-6f);
            if (vision_config.contains("rope_parameters") && vision_config["rope_parameters"].is_object()) {
                vision_rope_theta = vision_config["rope_parameters"].value("rope_theta", 100.0f);
            }
            
            // 每张图的 soft token 上限：processor_config.json 的 image_seq_length
            std::string processor_config_path = model_path + "/processor_config.json";
            size_t max_soft_tokens = 280;
            if (std::filesystem::exists(processor_config_path)) {
                std::ifstream pc_file(processor_config_path);
                if (pc_file.is_open()) {
                    try {
                        nlohmann::json pc;
                        pc_file >> pc;
                        if (pc.contains("image_seq_length") && pc["image_seq_length"].is_number_integer()) {
                            max_soft_tokens = pc["image_seq_length"].get<size_t>();
                        }
                        if (pc.contains("image_processor") && pc["image_processor"].is_object()) {
                            const auto& ip = pc["image_processor"];
                            vision_patch_size = ip.value("patch_size", vision_patch_size);
                            vision_pooling_kernel = ip.value("pooling_kernel_size", vision_pooling_kernel);
                            if (ip.contains("max_soft_tokens")) {
                                max_soft_tokens = ip["max_soft_tokens"].get<size_t>();
                            }
                        }
                    } catch (const std::exception&) {
                        // 保持默认值
                    }
                }
            }
            vision_max_patches = max_soft_tokens * vision_pooling_kernel * vision_pooling_kernel;
            
            // 图像占位 token
            image_token_id = this->config.value("image_token_id", 0);
            boi_token_id = this->config.value("boi_token_id", 0);
            eoi_token_id = this->config.value("eoi_token_id", 0);
            
            std::cout << "视觉塔: hidden=" << vision_hidden_size
                      << ", layers=" << vision_num_layers
                      << ", heads=" << vision_num_heads << "/" << vision_num_kv_heads
                      << ", patch=" << vision_patch_size
                      << ", max_patches=" << vision_max_patches
                      << ", rope_theta=" << vision_rope_theta
                      << ", image_token=" << image_token_id
                      << ", boi=" << boi_token_id << ", eoi=" << eoi_token_id
                      << std::endl;
        } else {
            std::cout << "未找到视觉塔权重，仅支持文本嵌入" << std::endl;
        }
    }
}


// 辅助函数：根据输入ID动态生成位置ID
std::vector<std::vector<int>> create_position_ids_from_input_ids(
    const std::vector<std::vector<int>>& input_ids, 
    int padding_idx = 0) {
    std::vector<std::vector<int>> position_ids;
    position_ids.reserve(input_ids.size());
    
    for (const auto& seq : input_ids) {
        std::vector<int> pos_ids(seq.size(), 0);
        int pos = 2; // 从1开始编号（假设0是padding_idx）
        
        for (size_t i = 0; i < seq.size(); ++i) {
            if (seq[i] != padding_idx) {
                pos_ids[i] = pos++; // 有效token连续编号
            }
            // padding token保持为0
        }
        
        position_ids.push_back(pos_ids);
    }
    
    return position_ids;
}

// 完整的 BGE 前向传播
std::vector<float> SentenceTransformerImpl::bge_forward(const std::vector<int32_t>& tokens) {
    // 获取词嵌入
    std::vector<float> embedded = embedding_layer(tokens);
    size_t seq_len = tokens.size();
    
    // 获取位置嵌入权重
    const std::vector<float>& pos_embeddings = weights["embeddings.position_embeddings.weight"];
    size_t max_seq_len = pos_embeddings.size() / embedding_dim;
    if (seq_len > max_seq_len) {
        throw std::runtime_error("序列长度超过位置嵌入的最大长度");
    }
    
    // 根据模型类型选择位置编码方式
    if (model_type == ModelType::BGE_M3) {
        // BGE-M3模型：动态生成位置ID，忽略padding
        int padding_idx = this->config.contains("pad_token_id") ? int(this->config["pad_token_id"]) : 0;
        std::vector<std::vector<int>> position_ids = create_position_ids_from_input_ids({tokens}, padding_idx);
        
        for (size_t i = 0; i < seq_len; ++i) {
            size_t pos_start = i * embedding_dim;
            int pos_id = position_ids[0][i];
            // 确保位置ID不越界
            if (pos_id < 0 || pos_id >= static_cast<int>(max_seq_len)) {
                pos_id = 0;
            }
            for (size_t j = 0; j < embedding_dim; ++j) {
                embedded[pos_start + j] += pos_embeddings[pos_id * embedding_dim + j];
            }
        }
    } else {
        // 传统BGE模型：直接使用序列索引作为位置ID
        for (size_t i = 0; i < seq_len; ++i) {
            size_t pos_start = i * embedding_dim;
            for (size_t j = 0; j < embedding_dim; ++j) {
                embedded[pos_start + j] += pos_embeddings[pos_start + j];
            }
        }
    }
    
    
    // {{ 新增：添加 token type 嵌入（段嵌入）}}
    // 1. 检查段嵌入权重是否存在
    if (weights.find("embeddings.token_type_embeddings.weight") == weights.end()) {
        throw std::runtime_error("未找到段嵌入权重: embeddings.token_type_embeddings.weight");
    }
    const std::vector<float>& token_type_embeddings = weights["embeddings.token_type_embeddings.weight"];

    // 2. 单句场景：所有 token 的段 ID 为 0（句子对需区分 0/1，此处简化）
    size_t segment_id = 0;  // 段 ID（0 表示第一句，1 表示第二句）
    size_t token_type_dim = token_type_embeddings.size() / embedding_dim;  // 段类型数量（通常为 2）
    if (segment_id >= token_type_dim) {
        throw std::runtime_error("段 ID 超出范围（最大段类型数量: " + std::to_string(token_type_dim) + "）");
    }

    // 3. 叠加段嵌入（每个 token 叠加对应段 ID 的嵌入向量）
    size_t token_type_start = segment_id * embedding_dim;  // 段 ID 对应嵌入的起始索引
    for (size_t i = 0; i < seq_len; ++i) {
        size_t token_start = i * embedding_dim;  // 当前 token 的嵌入起始索引
        for (size_t j = 0; j < embedding_dim; ++j) {
            embedded[token_start + j] += token_type_embeddings[token_type_start + j];
        }
    }
    // {{ 段嵌入添加结束 }}
    const float epsilon = this->config["layer_norm_eps"];
    
    // 应用嵌入层 LayerNorm（词嵌入 + 位置嵌入 + 段嵌入 后的归一化）
    const std::vector<float>& ln_weight = weights["embeddings.LayerNorm.weight"];
    const std::vector<float>& ln_bias = weights["embeddings.LayerNorm.bias"];
    embedded = cuda_matrix_ops.layer_norm(embedded, ln_weight, ln_bias, embedding_dim, epsilon);

    // {{ 预训练位置嵌入处理结束 }}
    
    // 假设 BGE 模型有 4 层 Transformer 编码器
    const int num_layers = this->config["num_hidden_layers"];
    for (int layer = 0; layer < num_layers; ++layer) {
        std::string q_weight_name = "encoder.layer." + std::to_string(layer) + ".attention.self.query.weight";
        std::string k_weight_name = "encoder.layer." + std::to_string(layer) + ".attention.self.key.weight";
        std::string v_weight_name = "encoder.layer." + std::to_string(layer) + ".attention.self.value.weight";
        // 添加QKV偏置名称
        std::string q_bias_name = "encoder.layer." + std::to_string(layer) + ".attention.self.query.bias";
        std::string k_bias_name = "encoder.layer." + std::to_string(layer) + ".attention.self.key.bias";
        std::string v_bias_name = "encoder.layer." + std::to_string(layer) + ".attention.self.value.bias";
        std::string ln_gamma_name = "encoder.layer." + std::to_string(layer) + ".attention.output.LayerNorm.weight";
        std::string ln_beta_name = "encoder.layer." + std::to_string(layer) + ".attention.output.LayerNorm.bias";
        std::string attn_dense_weight_name = "encoder.layer." + std::to_string(layer) + ".attention.output.dense.weight";
        std::string attn_dense_bias_name = "encoder.layer." + std::to_string(layer) + ".attention.output.dense.bias";

        std::string ff_weight_1_name = "encoder.layer." + std::to_string(layer) + ".intermediate.dense.weight";
        std::string ff_bias_1_name = "encoder.layer." + std::to_string(layer) + ".intermediate.dense.bias";
        std::string ff_weight_2_name = "encoder.layer." + std::to_string(layer) + ".output.dense.weight";
        std::string ff_bias_2_name = "encoder.layer." + std::to_string(layer) + ".output.dense.bias";
        std::string ln_ff_gamma_name = "encoder.layer." + std::to_string(layer) + ".output.LayerNorm.weight";
        std::string ln_ff_beta_name = "encoder.layer." + std::to_string(layer) + ".output.LayerNorm.bias";

        // 获取对应层的权重
        const std::vector<float>& weight_q = weights[q_weight_name];
        const std::vector<float>& weight_k = weights[k_weight_name];
        const std::vector<float>& weight_v = weights[v_weight_name];
        // 添加QKV偏置权重
        const std::vector<float>& query_bias = weights[q_bias_name];
        const std::vector<float>& key_bias = weights[k_bias_name];
        const std::vector<float>& value_bias = weights[v_bias_name];
        const std::vector<float>& attn_dense_weight = weights[attn_dense_weight_name];
        const std::vector<float>& attn_dense_bias = weights[attn_dense_bias_name];
        const std::vector<float>& ln_gamma = weights[ln_gamma_name];
        const std::vector<float>& ln_beta = weights[ln_beta_name];
        // {{ 添加前馈网络偏置项 }}
        const std::vector<float>& weight_ff_1 = weights[ff_weight_1_name];
        const std::vector<float>& bias_ff_1 = weights[ff_bias_1_name];
        const std::vector<float>& weight_ff_2 = weights[ff_weight_2_name];
        const std::vector<float>& bias_ff_2 = weights[ff_bias_2_name];
        const std::vector<float>& ln_ff_gamma = weights[ln_ff_gamma_name];
        const std::vector<float>& ln_ff_beta = weights[ln_ff_beta_name];

        int num_attention_heads = this->config["num_attention_heads"];
        // 多头注意力机制
        // 修改多头注意力调用
        std::vector<float> attn_output = multi_head_attention(
        embedded, weight_q, weight_k, weight_v, query_bias, key_bias, value_bias,
        num_attention_heads, embedding_dim
        );
        // {{ 新增：注意力输出 Dense 层（修复未使用 weight 的问题）}}
        // 3. 矩阵乘法：attn_output（[seq_len, embedding_dim]） × 权重（[embedding_dim, embedding_dim]）
        size_t batch_seq_len = attn_output.size() / embedding_dim;
        attn_output = cuda_matrix_ops.matrix_multiply_transpose(
        attn_output, attn_dense_weight, batch_seq_len, embedding_dim, embedding_dim
        );
        
        // 4. 叠加偏置
        for (size_t i = 0; i < attn_output.size(); ++i) {
            attn_output[i] += attn_dense_bias[i % embedding_dim];
        }
        // {{ Dense 层处理结束 }}

        // 残差连接（当前层输入 + Dense 层输出）
        for (size_t i = 0; i < embedded.size(); ++i) {
            attn_output[i] += embedded[i];
        }

        // 层归一化
        attn_output = cuda_matrix_ops.layer_norm(attn_output, ln_gamma, ln_beta, embedding_dim, epsilon);

        const int intermediate_dim = this->config["intermediate_size"];

        // 前馈神经网络（修改调用参数）
        std::vector<float> ff_output = feed_forward_network(
            attn_output, 
            weight_ff_1, bias_ff_1, 
            weight_ff_2, bias_ff_2,
            ln_ff_gamma, ln_ff_beta, 
            embedding_dim, intermediate_dim,
            0.1f  // dropout概率
        );
            
        embedded = ff_output;
    }

    
    // {{ 添加池化层处理（使用 pooler.dense.weight 和 pooler.dense.bias ）}}
    const std::vector<float>& pooler_weight = weights["pooler.dense.weight"];
    const std::vector<float>& pooler_bias = weights["pooler.dense.bias"];

    // 池化层处理 - 使用cpu_matrix_ops矩阵乘加算法
    std::vector<float> cls_output(embedding_dim);
    std::copy_n(embedded.begin(), embedding_dim, cls_output.begin());

    /*
    // pooler.dense 池化，用于分类任务
    // 使用matrix_multiply_transpose替换手动矩阵乘法
    std::vector<float> pooled = cuda_matrix_ops.matrix_multiply_transpose(
        cls_output,  // 输入向量 [1, embedding_dim]
        pooler_weight,  // 权重矩阵 [embedding_dim, embedding_dim]
        1,  // batch_size
        embedding_dim,  // input_dim
        embedding_dim   // output_dim
    );
    
    // 添加偏置
    for (size_t i = 0; i < embedding_dim; ++i) {
        pooled[i] += pooler_bias[i];
    }
    
    // 应用tanh激活
    for (float& val : pooled) {
        val = std::tanh(val);
    }
    // {{ 池化层处理结束 }}
    */

    /*
    //  平均池化
    // std::vector<float> pooled;
    pooled.resize(embedding_dim, 0.0f);
    for (size_t i = 0; i < embedded.size(); i += embedding_dim) {
        // if (i == 0) continue;
        for (size_t j = 0; j < embedding_dim; ++j) {
            pooled[j] += embedded[i + j];
        }
    }

    size_t token_count = (embedded.size() / embedding_dim) - 1; // 减去CLS token
    for (float& val : pooled) {
        val /= token_count;
    }
    */
    std::vector<float> pooled = cls_output;

    // 归一化
    float norm = 0.0f;
    for (float val : pooled) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    for (float& val : pooled) {
        val /= norm;
    }

    return pooled;
}

// RoPE (Rotary Position Embedding) 实现 - 匹配transformers库的Qwen2/Qwen3实现
static void apply_rope(std::vector<float>& q_states, std::vector<float>& k_states,
                       size_t seq_len, size_t q_dim, size_t num_q_heads,
                       size_t k_dim, size_t num_kv_heads,
                       float rope_theta = 1000000.0f) {
    size_t q_head_dim = q_dim / num_q_heads;
    size_t k_head_dim = k_dim / num_kv_heads;
    size_t q_half_dim = q_head_dim / 2;
    size_t k_half_dim = k_head_dim / 2;
    
    // 对每个位置应用RoPE
    for (size_t pos = 0; pos < seq_len; ++pos) {
        // 对Q应用RoPE
        for (size_t h = 0; h < num_q_heads; ++h) {
            // 计算该位置的cos/sin值（每个维度独立）
            std::vector<float> cos_vals(q_head_dim);
            std::vector<float> sin_vals(q_head_dim);
            for (size_t d = 0; d < q_head_dim; ++d) {
                // transformers库的公式：inv_freq[i] = 1.0 / (base^(2*i / head_dim))
                // 然后cos[d] = cos(pos * inv_freq[d % half_dim])
                size_t idx = d < q_half_dim ? d : d - q_half_dim;
                float inv_freq = 1.0f / std::pow(rope_theta, 2.0f * static_cast<float>(idx) / static_cast<float>(q_head_dim));
                float theta = static_cast<float>(pos) * inv_freq;
                cos_vals[d] = std::cos(theta);
                sin_vals[d] = std::sin(theta);
            }
            
            // 应用RoPE：q' = q * cos + rotate_half(q) * sin
            // rotate_half(q) = [-q[half:], q[:half]]
            // 所以：q'[d] = q[d] * cos[d] - q[d + half] * sin[d]  (d < half)
            //      q'[d + half] = q[d] * sin[d] + q[d + half] * cos[d]
            for (size_t d = 0; d < q_half_dim; ++d) {
                size_t base_idx = pos * q_dim + h * q_head_dim;
                float q0 = q_states[base_idx + d];
                float q1 = q_states[base_idx + d + q_half_dim];
                float cos_d = cos_vals[d];
                float sin_d = sin_vals[d];
                
                q_states[base_idx + d] = q0 * cos_d - q1 * sin_d;
                q_states[base_idx + d + q_half_dim] = q0 * sin_d + q1 * cos_d;
            }
        }
        
        // 对K应用RoPE
        for (size_t h = 0; h < num_kv_heads; ++h) {
            // 计算该位置的cos/sin值
            std::vector<float> cos_vals(k_head_dim);
            std::vector<float> sin_vals(k_head_dim);
            for (size_t d = 0; d < k_head_dim; ++d) {
                // transformers库的公式：inv_freq[i] = 1.0 / (base^(2*i / head_dim))
                size_t idx = d < k_half_dim ? d : d - k_half_dim;
                float inv_freq = 1.0f / std::pow(rope_theta, 2.0f * static_cast<float>(idx) / static_cast<float>(k_head_dim));
                float theta = static_cast<float>(pos) * inv_freq;
                cos_vals[d] = std::cos(theta);
                sin_vals[d] = std::sin(theta);
            }
            
            // 应用RoPE
            for (size_t d = 0; d < k_half_dim; ++d) {
                size_t base_idx = pos * k_dim + h * k_head_dim;
                float k0 = k_states[base_idx + d];
                float k1 = k_states[base_idx + d + k_half_dim];
                float cos_d = cos_vals[d];
                float sin_d = sin_vals[d];
                
                k_states[base_idx + d] = k0 * cos_d - k1 * sin_d;
                k_states[base_idx + d + k_half_dim] = k0 * sin_d + k1 * cos_d;
            }
        }
    }
}

// Qwen3模型的前向传播
std::vector<float> SentenceTransformerImpl::qwen3_forward(const std::vector<int32_t>& tokens) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 获取词嵌入
    std::vector<float> embedded = embedding_layer(tokens);
    size_t seq_len = tokens.size();
    size_t hidden_size = embedding_dim;
    
    size_t num_layers = this->config["num_hidden_layers"];
    size_t num_attention_heads = this->config["num_attention_heads"];
    size_t num_key_value_heads = this->config.contains("num_key_value_heads") ? 
                                this->config["num_key_value_heads"].get<size_t>() : num_attention_heads;
    
    // Qwen3使用GQA，Q的维度可能不同于hidden_size
    std::string q_proj_key_check = "layers.0.self_attn.q_proj.weight";
    size_t q_output_dim = hidden_size;  // 默认值
    if (weights.find(q_proj_key_check) != weights.end()) {
        const auto& q_weight = weights[q_proj_key_check];
        q_output_dim = q_weight.size() / hidden_size;  // [q_output_dim, hidden_size]
    }
    size_t head_dim = q_output_dim / num_attention_heads;
    size_t kv_output_dim = num_key_value_heads * head_dim;
    
    // 通过每一层Transformer
    for (size_t layer = 0; layer < num_layers; ++layer) {
        std::string layer_prefix = "layers." + std::to_string(layer) + ".";
        
        // 保存残差连接的输入
        std::vector<float> residual = embedded;
        
        // 1. 输入层RMS归一化（Pre-Norm架构）
        std::string input_layernorm_weight_key = layer_prefix + "input_layernorm.weight";
        if (weights.find(input_layernorm_weight_key) != weights.end()) {
            const std::vector<float>& ln_weight = weights[input_layernorm_weight_key];
            
            // RMS归一化
            float eps = this->config.value("rms_norm_eps", 1e-6f);
            
            
            embedded = cuda_matrix_ops.rms_norm(embedded, ln_weight, hidden_size, eps);
        } 

        
        
        // 用于保存注意力残差后的值，供MLP残差使用
        std::vector<float> hidden_after_attn;
        
        // 2. 自注意力机制
        std::string q_proj_key = layer_prefix + "self_attn.q_proj.weight";
        std::string k_proj_key = layer_prefix + "self_attn.k_proj.weight";
        std::string v_proj_key = layer_prefix + "self_attn.v_proj.weight";
        std::string o_proj_key = layer_prefix + "self_attn.o_proj.weight";
        
        // 检查是否存在注意力权重
        if (weights.find(q_proj_key) != weights.end() && 
            weights.find(k_proj_key) != weights.end() && 
            weights.find(v_proj_key) != weights.end() && 
            weights.find(o_proj_key) != weights.end()) {
            
            // 获取投影权重
            const std::vector<float>& q_proj_weight = weights[q_proj_key];
            const std::vector<float>& k_proj_weight = weights[k_proj_key];
            const std::vector<float>& v_proj_weight = weights[v_proj_key];
            const std::vector<float>& o_proj_weight = weights[o_proj_key];
            
           
            
            // 获取Qwen3特有的查询和键归一化权重
            std::string q_norm_key = layer_prefix + "self_attn.q_norm.weight";
            std::string k_norm_key = layer_prefix + "self_attn.k_norm.weight";
            
            if (weights.find(q_norm_key) != weights.end() && weights.find(k_norm_key) != weights.end()) {
                const std::vector<float>& q_norm_weight = weights[q_norm_key];
                const std::vector<float>& k_norm_weight = weights[k_norm_key];
                
                // 执行Q、K、V投影（使用正确的维度）
                // Q: [seq_len, hidden_size] x [hidden_size, q_output_dim]
                std::vector<float> q_states = cuda_matrix_ops.matrix_multiply_transpose(
                        embedded, q_proj_weight, seq_len, hidden_size, q_output_dim
                    );
                
                // K/V: [seq_len, hidden_size] x [hidden_size, kv_output_dim]
                std::vector<float> k_states = cuda_matrix_ops.matrix_multiply_transpose(
                        embedded, k_proj_weight, seq_len, hidden_size, kv_output_dim
                    );
                
                std::vector<float> v_states = cuda_matrix_ops.matrix_multiply_transpose(
                        embedded, v_proj_weight, seq_len, hidden_size, kv_output_dim
                    );
                
                // GQA注意力计算 - 使用CUDA加速
                // q_norm、RoPE和reshape/permute都在GPU内部完成
                std::vector<float> attn_output = cuda_attention_ops.grouped_query_attention(
                    q_states, k_states, v_states,
                    q_norm_weight, k_norm_weight,
                    seq_len, num_attention_heads, num_key_value_heads, head_dim
                );
                
                // 应用输出投影 O: [seq_len, q_output_dim] × [hidden_size, q_output_dim]^T
                std::vector<float> projected_output = cuda_matrix_ops.matrix_multiply_transpose(
                        attn_output, o_proj_weight, seq_len, q_output_dim, hidden_size
                    );
                
                // 将注意力输出加到残差输入上（CPU实现更高效）
                for (size_t i = 0; i < embedded.size(); ++i) {
                    embedded[i] = residual[i] + projected_output[i];
                }
                
                // 保存注意力残差后的值，用于MLP残差连接
                hidden_after_attn = embedded;
            }
        }
        
        // 3. 后注意力层RMS归一化（在MLP之前应用）
        std::string post_attn_layernorm_key = layer_prefix + "post_attention_layernorm.weight";
        if (weights.find(post_attn_layernorm_key) != weights.end()) {
            const std::vector<float>& ln_weight = weights[post_attn_layernorm_key];
            float eps = this->config.value("rms_norm_eps", 1e-6f);
            
            embedded = cuda_matrix_ops.rms_norm(embedded, ln_weight, hidden_size, eps);
        }
            
        // 4. MLP层（在post-attention LN之后）
        std::string gate_proj_key = layer_prefix + "mlp.gate_proj.weight";
        std::string up_proj_key = layer_prefix + "mlp.up_proj.weight";
        std::string down_proj_key = layer_prefix + "mlp.down_proj.weight";
        
        if (weights.find(gate_proj_key) != weights.end() &&
            weights.find(up_proj_key) != weights.end() &&
            weights.find(down_proj_key) != weights.end()) {
            
            const std::vector<float>& gate_proj_weight = weights[gate_proj_key];
            const std::vector<float>& up_proj_weight = weights[up_proj_key];
            const std::vector<float>& down_proj_weight = weights[down_proj_key];
            
            // 从权重中获取intermediate_size
            size_t intermediate_size = gate_proj_weight.size() / hidden_size;
            
            // 执行完整的MLP前馈网络（GPU融合：gate_proj + up_proj + SiLU_multiply + down_proj）
            std::vector<float> mlp_output = cuda_matrix_ops.mlp_fused(
                embedded, gate_proj_weight, up_proj_weight, down_proj_weight,
                seq_len, hidden_size, intermediate_size
            );
            
            
            // 将MLP输出加到注意力残差后的值上（不是LN后的值）
            for (size_t i = 0; i < embedded.size(); ++i) {
                embedded[i] = hidden_after_attn[i] + mlp_output[i];
            }
        }
    }
    
    // 最后的RMS层归一化
    if (weights.find("norm.weight") != weights.end()) {
        const std::vector<float>& final_norm_weight = weights["norm.weight"];
        float eps = this->config.value("rms_norm_eps", 1e-6f); // 使用配置中的rms_norm_eps值
        
        embedded = cuda_matrix_ops.rms_norm(embedded, final_norm_weight, hidden_size, eps);
    }
    
    // Qwen3使用last token pooling策略
    // 取最后一个token的隐藏状态作为句子表示
    std::vector<float> pooled_output(hidden_size);
    for (size_t j = 0; j < hidden_size; ++j) {
        pooled_output[j] = embedded[(seq_len - 1) * embedding_dim + j];  // 最后一个token
    }
    // 归一化输出向量
    float norm = 0.0f;
    for (float val : pooled_output) {
        norm += val * val;
            }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& val : pooled_output) {
            val /= norm;
            }
        }
        
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("[C++ Inference Time] %lld ms\n", static_cast<long long>(duration.count()));
        
    return pooled_output;
}

// Qwen3 批量前向传播 - 使用真正的批量 CUDA 操作
std::vector<std::vector<float>> SentenceTransformerImpl::qwen3_forward_batch(const std::vector<std::vector<int32_t>>& tokens_list) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    size_t batch_size = tokens_list.size();
    if (batch_size == 0) {
        return {};
    }
    
    // 找到最大序列长度
    size_t max_seq_len = 0;
    for (const auto& tokens : tokens_list) {
        if (tokens.size() > max_seq_len) {
            max_seq_len = tokens.size();
        }
    }
    
    size_t hidden_size = embedding_dim;
    size_t num_layers = this->config["num_hidden_layers"];
    size_t num_attention_heads = this->config["num_attention_heads"];
    size_t num_key_value_heads = this->config.contains("num_key_value_heads") ? 
                                this->config["num_key_value_heads"].get<size_t>() : num_attention_heads;
    
    // Qwen3使用GQA，Q的维度可能不同于hidden_size
    std::string q_proj_key_check = "layers.0.self_attn.q_proj.weight";
    size_t q_output_dim = hidden_size;  // 默认值
    if (weights.find(q_proj_key_check) != weights.end()) {
        const auto& q_weight = weights[q_proj_key_check];
        q_output_dim = q_weight.size() / hidden_size;  // [q_output_dim, hidden_size]
    }
    size_t head_dim = q_output_dim / num_attention_heads;
    size_t kv_output_dim = num_key_value_heads * head_dim;
    
    // 获取批量嵌入
    auto inference_start = std::chrono::high_resolution_clock::now();
    std::vector<float> embedded = embedding_layer_batch(tokens_list);
    
    // 通过每一层Transformer
    for (size_t layer = 0; layer < num_layers; ++layer) {
        std::string layer_prefix = "layers." + std::to_string(layer) + ".";
        
        // 保存残差连接的输入
        std::vector<float> residual = embedded;
        
        // 1. 输入层RMS归一化（Pre-Norm架构）
        std::string input_layernorm_weight_key = layer_prefix + "input_layernorm.weight";
        if (weights.find(input_layernorm_weight_key) != weights.end()) {
            const std::vector<float>& ln_weight = weights[input_layernorm_weight_key];
            float eps = this->config.value("rms_norm_eps", 1e-6f);
            embedded = cuda_matrix_ops.rms_norm_batch(embedded, ln_weight, batch_size, max_seq_len, hidden_size, eps);
        }
        
        // 用于保存注意力残差后的值，供MLP残差使用
        std::vector<float> hidden_after_attn;
        
        // 2. 自注意力机制 - 这里我们对每个样本单独处理，但可以优化
        std::string q_proj_key = layer_prefix + "self_attn.q_proj.weight";
        std::string k_proj_key = layer_prefix + "self_attn.k_proj.weight";
        std::string v_proj_key = layer_prefix + "self_attn.v_proj.weight";
        std::string o_proj_key = layer_prefix + "self_attn.o_proj.weight";
        
        if (weights.find(q_proj_key) != weights.end() && 
            weights.find(k_proj_key) != weights.end() && 
            weights.find(v_proj_key) != weights.end() && 
            weights.find(o_proj_key) != weights.end()) {
            
            const std::vector<float>& q_proj_weight = weights[q_proj_key];
            const std::vector<float>& k_proj_weight = weights[k_proj_key];
            const std::vector<float>& v_proj_weight = weights[v_proj_key];
            const std::vector<float>& o_proj_weight = weights[o_proj_key];
            
            std::string q_norm_key = layer_prefix + "self_attn.q_norm.weight";
            std::string k_norm_key = layer_prefix + "self_attn.k_norm.weight";
            
            if (weights.find(q_norm_key) != weights.end() && weights.find(k_norm_key) != weights.end()) {
                const std::vector<float>& q_norm_weight = weights[q_norm_key];
                const std::vector<float>& k_norm_weight = weights[k_norm_key];
                
                // 批量投影 Q, K, V
                std::vector<float> q_states_batch = cuda_matrix_ops.matrix_multiply_transpose_batch(
                    embedded, q_proj_weight, batch_size, max_seq_len, hidden_size, q_output_dim
                );
                
                std::vector<float> k_states_batch = cuda_matrix_ops.matrix_multiply_transpose_batch(
                    embedded, k_proj_weight, batch_size, max_seq_len, hidden_size, kv_output_dim
                );
                
                std::vector<float> v_states_batch = cuda_matrix_ops.matrix_multiply_transpose_batch(
                    embedded, v_proj_weight, batch_size, max_seq_len, hidden_size, kv_output_dim
                );
                
                // 对每个样本单独处理注意力
                std::vector<float> attn_output_batch(batch_size * max_seq_len * hidden_size, 0.0f);
                for (size_t b = 0; b < batch_size; ++b) {
                    size_t seq_len = tokens_list[b].size();
                    
                    // 提取单个样本的 Q, K, V
                    std::vector<float> q_single(q_output_dim * seq_len);
                    std::vector<float> k_single(kv_output_dim * seq_len);
                    std::vector<float> v_single(kv_output_dim * seq_len);
                    
                    for (size_t t = 0; t < seq_len; ++t) {
                        for (size_t d = 0; d < q_output_dim; ++d) {
                            q_single[t * q_output_dim + d] = q_states_batch[(b * max_seq_len + t) * q_output_dim + d];
                        }
                        for (size_t d = 0; d < kv_output_dim; ++d) {
                            k_single[t * kv_output_dim + d] = k_states_batch[(b * max_seq_len + t) * kv_output_dim + d];
                            v_single[t * kv_output_dim + d] = v_states_batch[(b * max_seq_len + t) * kv_output_dim + d];
                        }
                    }
                    
                    // 计算单个样本的注意力
                    std::vector<float> attn_single = cuda_attention_ops.grouped_query_attention(
                        q_single, k_single, v_single, q_norm_weight, k_norm_weight,
                        seq_len, num_attention_heads, num_key_value_heads, head_dim
                    );
                    
                    // 投影输出
                    std::vector<float> projected_single = cuda_matrix_ops.matrix_multiply_transpose(
                        attn_single, o_proj_weight, seq_len, q_output_dim, hidden_size
                    );
                    
                    // 残差连接 - 只对有效部分
                    for (size_t t = 0; t < seq_len; ++t) {
                        for (size_t d = 0; d < hidden_size; ++d) {
                            size_t output_idx = (b * max_seq_len + t) * hidden_size + d;
                            attn_output_batch[output_idx] = residual[(b * max_seq_len + t) * hidden_size + d] + projected_single[t * hidden_size + d];
                        }
                    }
                }
                
                embedded = std::move(attn_output_batch);
                hidden_after_attn = embedded;
            }
        }
        
        // 3. 后注意力层RMS归一化（在MLP之前应用）
        std::string post_attn_layernorm_key = layer_prefix + "post_attention_layernorm.weight";
        if (weights.find(post_attn_layernorm_key) != weights.end()) {
            const std::vector<float>& ln_weight = weights[post_attn_layernorm_key];
            float eps = this->config.value("rms_norm_eps", 1e-6f);
            embedded = cuda_matrix_ops.rms_norm_batch(embedded, ln_weight, batch_size, max_seq_len, hidden_size, eps);
        }
        
        // 4. MLP层（在post-attention LN之后）
        std::string gate_proj_key = layer_prefix + "mlp.gate_proj.weight";
        std::string up_proj_key = layer_prefix + "mlp.up_proj.weight";
        std::string down_proj_key = layer_prefix + "mlp.down_proj.weight";
        
        if (weights.find(gate_proj_key) != weights.end() &&
            weights.find(up_proj_key) != weights.end() &&
            weights.find(down_proj_key) != weights.end()) {
            
            const std::vector<float>& gate_proj_weight = weights[gate_proj_key];
            const std::vector<float>& up_proj_weight = weights[up_proj_key];
            const std::vector<float>& down_proj_weight = weights[down_proj_key];
            
            size_t intermediate_size = gate_proj_weight.size() / hidden_size;
            
            // 执行批量MLP
            std::vector<float> mlp_output = cuda_matrix_ops.mlp_fused_batch(
                embedded, gate_proj_weight, up_proj_weight, down_proj_weight,
                batch_size, max_seq_len, hidden_size, intermediate_size
            );
            
            // 残差连接
            embedded = cuda_matrix_ops.vector_add_batch(hidden_after_attn, mlp_output);
        }
    }
    
    // 最后的RMS层归一化
    if (weights.find("norm.weight") != weights.end()) {
        const std::vector<float>& final_norm_weight = weights["norm.weight"];
        float eps = this->config.value("rms_norm_eps", 1e-6f);
        embedded = cuda_matrix_ops.rms_norm_batch(embedded, final_norm_weight, batch_size, max_seq_len, hidden_size, eps);
    }
    
    // 提取每个样本的最后一个token并归一化
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(batch_size);
    
    for (size_t b = 0; b < batch_size; ++b) {
        size_t seq_len = tokens_list[b].size();
        std::vector<float> pooled_output(hidden_size);
        
        // 取最后一个token的隐藏状态
        for (size_t j = 0; j < hidden_size; ++j) {
            pooled_output[j] = embedded[(b * max_seq_len + seq_len - 1) * hidden_size + j];
        }
        
        // 归一化
        float norm = 0.0f;
        for (float val : pooled_output) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float& val : pooled_output) {
                val /= norm;
            }
        }
        
        embeddings.push_back(std::move(pooled_output));
    }
    
    auto inference_end = std::chrono::high_resolution_clock::now();
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start);
    
    printf("[Qwen3 Batch] Total: %lld ms | Inference: %lld ms | Batch size: %zu\n",
           static_cast<long long>(total_duration.count()),
           static_cast<long long>(inference_duration.count()),
           batch_size);
    
    return embeddings;
}

// ================== EmbeddingGemma 2 前向传播 ==================
// 参考实现: transformers/models/embedding_gemma2/modeling_embedding_gemma2.py
// 结构要点：
//   1. 词嵌入乘 sqrt(hidden_size)
//   2. 全局 PLE 投影 -> 每层一份 per-layer 输入（RMSNorm）
//   3. 每层: Pre-Norm 双向注意力 -> Pre-Norm MLP(gelu_pytorch_tanh) -> PLE block -> layer_scalar
//   4. 最终 RMSNorm -> embedding_projection -> 均值池化 -> L2 归一化
namespace {
inline float gelu_tanh_scalar(float x) {
    constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float kBeta = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + kBeta * x * x * x)));
}
}  // namespace

std::vector<float> SentenceTransformerImpl::embeddinggemma2_forward(const std::vector<int32_t>& tokens) {
    const size_t seq_len = tokens.size();
    if (seq_len == 0) {
        throw std::runtime_error("EmbeddingGemma2: 输入序列为空");
    }
    
    // 1) 词嵌入，Gemma 系列需要乘 sqrt(hidden_size)
    std::vector<float> inputs_embeds = embedding_layer(tokens);
    const float embed_scale = std::sqrt(static_cast<float>(embedding_dim));
    for (float& val : inputs_embeds) {
        val *= embed_scale;
    }
    return embeddinggemma2_embeds_forward(inputs_embeds, seq_len);
}

std::vector<float> SentenceTransformerImpl::embeddinggemma2_embeds_forward(
    const std::vector<float>& inputs_embeds, size_t seq_len) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const size_t hidden_size = embedding_dim;
    if (seq_len == 0) {
        throw std::runtime_error("EmbeddingGemma2: 输入序列为空");
    }
    if (inputs_embeds.size() != seq_len * hidden_size) {
        throw std::runtime_error("EmbeddingGemma2: inputs_embeds 尺寸不匹配");
    }
    if (seq_len > 1000000) {
        throw std::runtime_error("EmbeddingGemma2: 序列过长");
    }
    std::vector<float> hidden = inputs_embeds;
    
    const size_t num_layers = this->text_config.value("num_hidden_layers", 0);
    const size_t num_attention_heads = this->text_config.value("num_attention_heads", 0);
    const float eps = this->text_config.value("rms_norm_eps", 1e-6f);
    const int sliding_window = this->text_config.value("sliding_window", 0);
    if (num_layers == 0 || num_attention_heads == 0) {
        throw std::runtime_error("EmbeddingGemma2: config 缺少 num_hidden_layers / num_attention_heads");
    }
    
    // 2) 全局 PLE：per_layer_model_projection(无偏置) -> 缩放 -> 每层 RMSNorm
    const std::string ple_proj_key = "language_model.ple.per_layer_model_projection.weight";
    const std::string ple_norm_key = "language_model.ple.per_layer_projection_norm.weight";
    const auto ple_proj_it = weights.find(ple_proj_key);
    const auto ple_norm_it = weights.find(ple_norm_key);
    const bool use_ple = (ple_proj_it != weights.end() && ple_norm_it != weights.end());
    size_t per_layer_dim = hidden_size;
    std::vector<float> per_layer_input;  // [seq_len, num_layers, per_layer_dim]
    if (use_ple) {
        per_layer_dim = ple_norm_it->second.size();
        per_layer_input = cuda_matrix_ops.matrix_multiply_transpose(
            hidden, ple_proj_it->second, seq_len, hidden_size, num_layers * per_layer_dim);
        const float ple_scale = 1.0f / std::sqrt(static_cast<float>(hidden_size));
        for (float& val : per_layer_input) {
            val *= ple_scale;
        }
        per_layer_input = cuda_matrix_ops.rms_norm(per_layer_input, ple_norm_it->second, per_layer_dim, eps);
    }
    
    const float default_rope_theta = this->text_config.value("rope_theta", 1000000.0f);
    
    // 3) 逐层编码
    for (size_t layer = 0; layer < num_layers; ++layer) {
        const std::string prefix = "language_model.layers." + std::to_string(layer) + ".";
        
        // 该层的注意力类型决定 RoPE base 与滑动窗口
        const std::string layer_type = (layer < layer_types.size()) ? layer_types[layer] : std::string("full_attention");
        float rope_theta = default_rope_theta;
        int window = 0;
        if (layer_type == "sliding_attention" && sliding_window > 0) {
            window = sliding_window;
        }
        if (this->text_config.contains("rope_parameters") &&
            this->text_config["rope_parameters"].is_object()) {
            const auto& rope_cfg = this->text_config["rope_parameters"];
            if (rope_cfg.contains(layer_type) && rope_cfg[layer_type].is_object() &&
                rope_cfg[layer_type].contains("rope_theta")) {
                rope_theta = rope_cfg[layer_type]["rope_theta"].get<float>();
            }
        }
        
        // 3.1 Pre-Norm + 双向自注意力
        std::vector<float> residual = hidden;
        std::vector<float> normed = cuda_matrix_ops.rms_norm(
            hidden, weights.at(prefix + "input_layernorm.weight"), hidden_size, eps);
        
        const std::vector<float>& q_weight = weights.at(prefix + "self_attn.q_proj.weight");
        const std::vector<float>& k_weight = weights.at(prefix + "self_attn.k_proj.weight");
        const std::vector<float>& v_weight = weights.at(prefix + "self_attn.v_proj.weight");
        const std::vector<float>& o_weight = weights.at(prefix + "self_attn.o_proj.weight");
        
        const size_t q_out_dim = q_weight.size() / hidden_size;
        const size_t head_dim = q_out_dim / num_attention_heads;
        const size_t kv_out_dim = k_weight.size() / hidden_size;
        const size_t num_kv_heads = kv_out_dim / head_dim;
        if (head_dim == 0 || num_kv_heads == 0) {
            throw std::runtime_error("EmbeddingGemma2: 注意力头维度解析失败");
        }
        
        std::vector<float> q = cuda_matrix_ops.matrix_multiply_transpose(
            normed, q_weight, seq_len, hidden_size, q_out_dim);
        std::vector<float> k = cuda_matrix_ops.matrix_multiply_transpose(
            normed, k_weight, seq_len, hidden_size, kv_out_dim);
        std::vector<float> v = cuda_matrix_ops.matrix_multiply_transpose(
            normed, v_weight, seq_len, hidden_size, kv_out_dim);
        
        std::vector<float> attn = cuda_attention_ops.encoder_attention(
            q, k, v,
            weights.at(prefix + "self_attn.q_norm.weight"),
            weights.at(prefix + "self_attn.k_norm.weight"),
            seq_len, num_attention_heads, num_kv_heads, head_dim,
            rope_theta,
            1.0f,   // EmbeddingGemma2 的注意力缩放固定为 1.0（Q/K 已做 RMSNorm）
            true,   // V 做无权重 RMSNorm
            window,
            eps
        );
        
        std::vector<float> attn_out = cuda_matrix_ops.matrix_multiply_transpose(
            attn, o_weight, seq_len, q_out_dim, hidden_size);
        attn_out = cuda_matrix_ops.rms_norm(
            attn_out, weights.at(prefix + "post_attention_layernorm.weight"), hidden_size, eps);
        for (size_t i = 0; i < hidden.size(); ++i) {
            hidden[i] = residual[i] + attn_out[i];
        }
        
        // 3.2 Pre-Norm + MLP（gelu_pytorch_tanh 门控）
        residual = hidden;
        normed = cuda_matrix_ops.rms_norm(
            hidden, weights.at(prefix + "pre_feedforward_layernorm.weight"), hidden_size, eps);
        
        const std::vector<float>& gate_weight = weights.at(prefix + "mlp.gate_proj.weight");
        const std::vector<float>& up_weight = weights.at(prefix + "mlp.up_proj.weight");
        const std::vector<float>& down_weight = weights.at(prefix + "mlp.down_proj.weight");
        const size_t intermediate_size = gate_weight.size() / hidden_size;
        
        std::vector<float> mlp = cuda_matrix_ops.mlp_fused_gelu(
            normed, gate_weight, up_weight, down_weight,
            seq_len, hidden_size, intermediate_size);
        mlp = cuda_matrix_ops.rms_norm(
            mlp, weights.at(prefix + "post_feedforward_layernorm.weight"), hidden_size, eps);
        for (size_t i = 0; i < hidden.size(); ++i) {
            hidden[i] = residual[i] + mlp[i];
        }
        
        // 3.3 PLE block：把该层的 per-layer 输入门控进残差流
        if (use_ple) {
            residual = hidden;
            std::vector<float> gate = cuda_matrix_ops.matrix_multiply_transpose(
                hidden, weights.at(prefix + "ple_block.per_layer_input_gate.weight"),
                seq_len, hidden_size, per_layer_dim);
            for (size_t t = 0; t < seq_len; ++t) {
                const size_t pl_offset = (t * num_layers + layer) * per_layer_dim;
                for (size_t d = 0; d < per_layer_dim; ++d) {
                    gate[t * per_layer_dim + d] =
                        gelu_tanh_scalar(gate[t * per_layer_dim + d]) * per_layer_input[pl_offset + d];
                }
            }
            gate = cuda_matrix_ops.matrix_multiply_transpose(
                gate, weights.at(prefix + "ple_block.per_layer_projection.weight"),
                seq_len, per_layer_dim, hidden_size);
            gate = cuda_matrix_ops.rms_norm(
                gate, weights.at(prefix + "ple_block.post_per_layer_input_norm.weight"), hidden_size, eps);
            for (size_t i = 0; i < hidden.size(); ++i) {
                hidden[i] = residual[i] + gate[i];
            }
        }
        
        // 3.4 逐层标量缩放（buffer，权重文件中为 [1]）
        const auto layer_scalar_it = weights.find(prefix + "layer_scalar");
        if (layer_scalar_it != weights.end() && !layer_scalar_it->second.empty()) {
            const float layer_scalar = layer_scalar_it->second[0];
            for (float& val : hidden) {
                val *= layer_scalar;
            }
        }
    }
    
    // 4) 最终 RMSNorm
    hidden = cuda_matrix_ops.rms_norm(hidden, weights.at("language_model.norm.weight"), hidden_size, eps);
    
    // 5) 输出投影 hidden_size -> embedding_dim
    std::vector<float> projected;
    const auto proj_it = weights.find("language_model.embedding_projection.weight");
    if (proj_it != weights.end()) {
        projected = cuda_matrix_ops.matrix_multiply_transpose(
            hidden, proj_it->second, seq_len, hidden_size, embedding_output_dim);
    } else {
        projected = hidden;
    }
    
    // 6) 均值池化（单条文本无 padding，所有 token 参与） + L2 归一化
    std::vector<float> pooled(embedding_output_dim, 0.0f);
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t d = 0; d < embedding_output_dim; ++d) {
            pooled[d] += projected[t * embedding_output_dim + d];
        }
    }
    const float inv_count = 1.0f / static_cast<float>(seq_len);
    float norm = 0.0f;
    for (float& val : pooled) {
        val *= inv_count;
        norm += val * val;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& val : pooled) {
            val /= norm;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("[EmbeddingGemma2 Inference Time] %lld ms\n", static_cast<long long>(duration.count()));
    
    return pooled;
}

std::vector<std::vector<float>> SentenceTransformerImpl::embeddinggemma2_forward_batch(
    const std::vector<std::vector<int32_t>>& tokens_list) {
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(tokens_list.size());
    for (const auto& tokens : tokens_list) {
        embeddings.push_back(embeddinggemma2_forward(tokens));
    }
    return embeddings;
}

// ===== 视觉塔（图像编码器）前向 =====
// 对齐 transformers 的 Gemma4VisionModel：
//   patch_embedder(2*(x-0.5) -> Linear -> +2D位置嵌入) -> 16 层双向编码 -> 3x3 位置均值池化
//   -> ×sqrt(hidden) -> embed_vision（无权重 RMSNorm + 投影到文本 hidden）
std::vector<float> SentenceTransformerImpl::embeddinggemma2_vision_forward(const GemmaImageInput& image) {
    if (!has_vision_tower) {
        throw std::runtime_error("当前模型没有可用的视觉塔权重");
    }
    const size_t num_patches = image.num_patches;
    const size_t hidden = vision_hidden_size;
    const size_t head_dim = vision_head_dim;
    const size_t q_out_dim = vision_num_heads * head_dim;
    const size_t kv_out_dim = vision_num_kv_heads * head_dim;
    // 每个 patch 的展开长度 = 3 * patch^2（本模型为 768，与 hidden 恰好相等）
    const size_t patch_pixels = 3 * vision_patch_size * vision_patch_size;
    if (num_patches == 0 || hidden == 0 || head_dim == 0 || patch_pixels == 0) {
        throw std::runtime_error("视觉塔: 维度参数非法");
    }
    if (image.pixel_values.size() != num_patches * patch_pixels) {
        throw std::runtime_error("视觉塔: patch 维度与 patch_size 不匹配");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 1) patch 归一化：2*(x-0.5)，然后 input_proj（patch_pixels -> hidden，无偏置）
    std::vector<float> patches = image.pixel_values;
    for (float& val : patches) {
        val = 2.0f * (val - 0.5f);
    }
    std::vector<float> hidden_states = cuda_matrix_ops.matrix_multiply_transpose(
        patches, weights.at("vision_tower.patch_embedder.input_proj.weight"),
        num_patches, patch_pixels, hidden);

    // 2) 2D 位置嵌入：table [2, position_embedding_size, hidden]，x/y 分别查表后相加
    {
        const std::vector<float>& pos_table = weights.at("vision_tower.patch_embedder.position_embedding_table");
        const size_t pos_size = pos_table.size() / (2 * hidden);
        if (pos_size == 0) {
            throw std::runtime_error("视觉塔: 位置嵌入表尺寸异常");
        }
        for (size_t p = 0; p < num_patches; ++p) {
            if (image.valid[p] == 0) {
                continue;  // padding patch 的位置嵌入置 0
            }
            int x = image.position_ids[p * 2];
            int y = image.position_ids[p * 2 + 1];
            x = std::max(0, std::min(x, static_cast<int>(pos_size) - 1));
            y = std::max(0, std::min(y, static_cast<int>(pos_size) - 1));
            const float* emb_x = pos_table.data() + static_cast<size_t>(x) * hidden;
            const float* emb_y = pos_table.data() + (pos_size + static_cast<size_t>(y)) * hidden;
            float* dst = hidden_states.data() + p * hidden;
            for (size_t d = 0; d < hidden; ++d) {
                dst[d] += emb_x[d] + emb_y[d];
            }
        }
    }

    // 3) 逐层编码（Pre-Norm + 双向注意力 + GELU 门控 MLP）
    for (size_t layer = 0; layer < vision_num_layers; ++layer) {
        const std::string prefix = "vision_tower.encoder.layers." + std::to_string(layer) + ".";

        // 3.1 自注意力
        std::vector<float> residual = hidden_states;
        std::vector<float> normed = cuda_matrix_ops.rms_norm(
            hidden_states, weights.at(prefix + "input_layernorm.weight"), hidden, vision_eps);

        std::vector<float> q = cuda_matrix_ops.matrix_multiply_transpose(
            normed, weights.at(prefix + "self_attn.q_proj.linear.weight"),
            num_patches, hidden, q_out_dim);
        std::vector<float> k = cuda_matrix_ops.matrix_multiply_transpose(
            normed, weights.at(prefix + "self_attn.k_proj.linear.weight"),
            num_patches, hidden, kv_out_dim);
        std::vector<float> v = cuda_matrix_ops.matrix_multiply_transpose(
            normed, weights.at(prefix + "self_attn.v_proj.linear.weight"),
            num_patches, hidden, kv_out_dim);

        std::vector<float> attn = cuda_attention_ops.vision_attention(
            q, k, v,
            weights.at(prefix + "self_attn.q_norm.weight"),
            weights.at(prefix + "self_attn.k_norm.weight"),
            image.position_ids, image.valid,
            num_patches, vision_num_heads, vision_num_kv_heads, head_dim,
            vision_rope_theta,
            1.0f,        // Gemma4 视觉塔的注意力缩放固定为 1.0
            vision_eps);

        std::vector<float> attn_out = cuda_matrix_ops.matrix_multiply_transpose(
            attn, weights.at(prefix + "self_attn.o_proj.linear.weight"),
            num_patches, q_out_dim, hidden);
        attn_out = cuda_matrix_ops.rms_norm(
            attn_out, weights.at(prefix + "post_attention_layernorm.weight"), hidden, vision_eps);
        for (size_t i = 0; i < hidden_states.size(); ++i) {
            hidden_states[i] = residual[i] + attn_out[i];
        }

        // 3.2 前馈网络（gelu_pytorch_tanh 门控）
        residual = hidden_states;
        normed = cuda_matrix_ops.rms_norm(
            hidden_states, weights.at(prefix + "pre_feedforward_layernorm.weight"), hidden, vision_eps);
        std::vector<float> mlp = cuda_matrix_ops.mlp_fused_gelu(
            normed,
            weights.at(prefix + "mlp.gate_proj.linear.weight"),
            weights.at(prefix + "mlp.up_proj.linear.weight"),
            weights.at(prefix + "mlp.down_proj.linear.weight"),
            num_patches, hidden, vision_intermediate_size);
        mlp = cuda_matrix_ops.rms_norm(
            mlp, weights.at(prefix + "post_feedforward_layernorm.weight"), hidden, vision_eps);
        for (size_t i = 0; i < hidden_states.size(); ++i) {
            hidden_states[i] = residual[i] + mlp[i];
        }
    }

    // 4) 3x3 位置均值池化：每个块 = 9 个 patch 的平均（padding patch 置 0 后不贡献）
    const size_t k = vision_pooling_kernel;
    const size_t grid_h = image.patch_grid_h;
    const size_t grid_w = image.patch_grid_w;
    const size_t blocks_per_row = grid_w / k;
    const size_t num_soft_tokens = image.num_real_patches / (k * k);
    if (num_soft_tokens == 0 || blocks_per_row == 0) {
        throw std::runtime_error("视觉塔: 池化后 soft token 数为 0");
    }
    std::vector<float> pooled(num_soft_tokens * hidden, 0.0f);
    const float inv_k2 = 1.0f / static_cast<float>(k * k);
    for (size_t py = 0; py < grid_h; ++py) {
        for (size_t px = 0; px < grid_w; ++px) {
            const size_t patch_idx = py * grid_w + px;
            const size_t block = (py / k) * blocks_per_row + (px / k);
            if (block >= num_soft_tokens || image.valid[patch_idx] == 0) {
                continue;
            }
            const float* src = hidden_states.data() + patch_idx * hidden;
            float* dst = pooled.data() + block * hidden;
            for (size_t d = 0; d < hidden; ++d) {
                dst[d] += src[d] * inv_k2;
            }
        }
    }

    // 5) ×sqrt(hidden_size)
    const float root_scale = std::sqrt(static_cast<float>(hidden));
    for (float& val : pooled) {
        val *= root_scale;
    }

    // 6) embed_vision：无权重 RMSNorm（权重全 1） + 投影到文本 hidden
    std::vector<float> unit_weight(hidden, 1.0f);
    std::vector<float> normed_pooled = cuda_matrix_ops.rms_norm(pooled, unit_weight, hidden, vision_eps);
    std::vector<float> projected = cuda_matrix_ops.matrix_multiply_transpose(
        normed_pooled, weights.at("embed_vision.embedding_projection.weight"),
        num_soft_tokens, hidden, embedding_dim);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("[Vision Inference Time] %lld ms (%zu patches -> %zu soft tokens)\n",
           static_cast<long long>(duration.count()), num_patches, num_soft_tokens);

    return projected;
}

// 图像嵌入：视觉 soft token 替换文本序列中的 <image> 占位，再过语言模型
std::vector<float> SentenceTransformerImpl::embeddinggemma2_image_embedding(const GemmaImageInput& image) {
    if (!has_vision_tower) {
        throw std::runtime_error("当前模型没有可用的视觉塔权重");
    }
    if (image_token_id == 0 || boi_token_id == 0 || eoi_token_id == 0) {
        throw std::runtime_error("配置缺少 image_token_id / boi_token_id / eoi_token_id，无法拼接图像序列");
    }
    const size_t k = vision_pooling_kernel;
    const size_t num_soft_tokens = image.num_real_patches / (k * k);
    if (num_soft_tokens == 0) {
        throw std::runtime_error("图像编码失败: soft token 数为 0");
    }

    // 1) 视觉塔 -> [num_soft_tokens, text_hidden]
    std::vector<float> soft_tokens = embeddinggemma2_vision_forward(image);
    const size_t text_hidden = embedding_dim;
    if (soft_tokens.size() != num_soft_tokens * text_hidden) {
        throw std::runtime_error("图像编码失败: soft token 维度异常");
    }

    // 2) 构造序列：[bos] + [boi] + [image] * n + [eoi] + [eos]
    std::vector<int32_t> tokens;
    tokens.reserve(num_soft_tokens + 4);
    if (gemma_tokenizer) {
        tokens.push_back(static_cast<int32_t>(gemma_tokenizer->get_bos_token_id()));
    }
    if (boi_token_id > 0) {
        tokens.push_back(static_cast<int32_t>(boi_token_id));
    }
    const size_t image_start = tokens.size();
    for (size_t i = 0; i < num_soft_tokens; ++i) {
        tokens.push_back(static_cast<int32_t>(image_token_id));
    }
    if (eoi_token_id > 0) {
        tokens.push_back(static_cast<int32_t>(eoi_token_id));
    }
    if (gemma_tokenizer && gemma_tokenizer->get_eos_token_id() >= 0) {
        tokens.push_back(static_cast<int32_t>(gemma_tokenizer->get_eos_token_id()));
    }

    // 3) 文本部分取词嵌入并乘 sqrt(hidden)，图像占位替换为视觉 soft token
    const size_t seq_len = tokens.size();
    std::vector<float> inputs_embeds = embedding_layer(tokens);
    const float embed_scale = std::sqrt(static_cast<float>(text_hidden));
    for (float& val : inputs_embeds) {
        val *= embed_scale;
    }
    for (size_t i = 0; i < num_soft_tokens; ++i) {
        const float* src = soft_tokens.data() + i * text_hidden;
        float* dst = inputs_embeds.data() + (image_start + i) * text_hidden;
        std::copy(src, src + text_hidden, dst);
    }

    // 4) 语言模型 + 均值池化 + L2
    return embeddinggemma2_embeds_forward(inputs_embeds, seq_len);
}

std::vector<float> SentenceTransformerImpl::encode_image(const std::string& image_path) {
    if (model_type != ModelType::EMBEDDING_GEMMA2 || !has_vision_tower) {
        throw std::runtime_error("当前模型不支持图像编码（需要带视觉塔的 EmbeddingGemma2）");
    }
    GemmaImageInput image = load_and_preprocess_image(
        image_path, static_cast<int>(vision_patch_size),
        static_cast<int>(vision_max_patches), static_cast<int>(vision_pooling_kernel));
    return embeddinggemma2_image_embedding(image);
}

std::vector<float> SentenceTransformerImpl::encode_image_bytes(const std::vector<uint8_t>& image_bytes) {
    if (model_type != ModelType::EMBEDDING_GEMMA2 || !has_vision_tower) {
        throw std::runtime_error("当前模型不支持图像编码（需要带视觉塔的 EmbeddingGemma2）");
    }
    if (image_bytes.empty()) {
        throw std::runtime_error("图像数据为空");
    }
    int h = 0, w = 0;
    std::vector<uint8_t> rgb = decode_image_rgb_from_memory(image_bytes.data(), image_bytes.size(), &h, &w);
    GemmaImageInput image = preprocess_gemma_image(
        rgb, h, w, static_cast<int>(vision_patch_size),
        static_cast<int>(vision_max_patches), static_cast<int>(vision_pooling_kernel));
    return embeddinggemma2_image_embedding(image);
}

bool SentenceTransformerImpl::supports_image() const {
    return model_type == ModelType::EMBEDDING_GEMMA2 && has_vision_tower;
}

std::vector<float> SentenceTransformerImpl::encode(const std::string& text, const std::string& prompt_name) {
    // 分词（如需提示词前缀，先拼接）
    std::vector<int32_t> tokens = tokenize(apply_prompt(text, prompt_name));
    // 打印tokens
    std::cout << "tokens: " << std::endl;
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    // 根据模型类型选择前向传播方法
    if (model_type == ModelType::QWEN3) {
        return qwen3_forward(tokens);
    } else if (model_type == ModelType::EMBEDDING_GEMMA2) {
        return embeddinggemma2_forward(tokens);
    } else {
        return bge_forward(tokens);
    }
}

std::vector<std::vector<float>> SentenceTransformerImpl::encode_batch(const std::vector<std::string>& texts, const std::string& prompt_name) {
    if (model_type == ModelType::QWEN3) {
        // Qwen3 模型使用批量推理
        std::vector<std::string> prompted;
        prompted.reserve(texts.size());
        for (const auto& text : texts) {
            prompted.push_back(apply_prompt(text, prompt_name));
        }
        auto tokens_list = tokenize_batch(prompted);
        return qwen3_forward_batch(tokens_list);
    } else if (model_type == ModelType::EMBEDDING_GEMMA2) {
        // EmbeddingGemma2 的注意力是双向的，逐条处理避免 padding 干扰池化
        std::vector<std::vector<float>> result;
        result.reserve(texts.size());
        for (const auto& text : texts) {
            result.push_back(encode(text, prompt_name));
        }
        return result;
    } else {
        // 其他模型逐个处理
        std::vector<std::vector<float>> result;
        for (const auto& text : texts) {
            result.push_back(encode(text, prompt_name));
        }
        return result;
    }
}

size_t SentenceTransformerImpl::get_embedding_dimension() const {
    return embedding_output_dim;
}

std::vector<std::string> SentenceTransformerImpl::get_prompt_names() const {
    std::vector<std::string> names;
    names.reserve(prompts.size());
    for (const auto& item : prompts) {
        names.push_back(item.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::string SentenceTransformerImpl::apply_prompt(const std::string& text, const std::string& prompt_name) const {
    if (prompt_name.empty()) {
        return text;
    }
    auto it = prompts.find(prompt_name);
    if (it == prompts.end()) {
        // 找不到提示词时退化为原始文本，保证服务可用
        return text;
    }
    return it->second + text;
}

// 实现 SentenceTransformerImpl 的 multi_head_attention 函数
std::vector<float> SentenceTransformerImpl::multi_head_attention(
    const std::vector<float>& input,
    const std::vector<float>& weight_q,
    const std::vector<float>& weight_k,
    const std::vector<float>& weight_v,
    const std::vector<float>& query_bias,  // New: query bias
    const std::vector<float>& key_bias,    // New: key bias
    const std::vector<float>& value_bias,  // New: value bias
    size_t num_heads,
    size_t embedding_dim
) {
    // 调用 CPUAttentionOps 的 multi_head_attention 方法
    return cuda_attention_ops.multi_head_attention(input, weight_q, weight_k, weight_v, query_bias, key_bias, value_bias, num_heads, embedding_dim);
}

// 实现 SentenceTransformerImpl 的 feed_forward_network 函数
// {{ 修改函数签名，添加偏置参数 }}
std::vector<float> SentenceTransformerImpl::feed_forward_network(const std::vector<float>& input, 
                                                                  const std::vector<float>& weight_ff_1, 
                                                                  const std::vector<float>& bias_ff_1, 
                                                                  const std::vector<float>& weight_ff_2, 
                                                                  const std::vector<float>& bias_ff_2, 
                                                                  const std::vector<float>& ln_gamma,  
                                                                  const std::vector<float>& ln_beta,   
                                                                  size_t embedding_dim, 
                                                                  size_t intermediate_dim,
                                                                  float dropout_prob) {
    size_t batch_seq_len = input.size() / embedding_dim;
    const float epsilon = this->config["layer_norm_eps"];

    // 第一层线性变换 - 替换为CUDA矩阵乘
    std::vector<float> hidden = cuda_matrix_ops.matrix_multiply_transpose(
        input, weight_ff_1, batch_seq_len, embedding_dim, intermediate_dim
    );
    
    // 添加第一层偏置
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] += bias_ff_1[i % intermediate_dim];
    }

    // 激活函数 - 替换为CUDA版本
    if (this->config["hidden_act"] == "gelu") {
        hidden = cuda_matrix_ops.gelu(hidden);
    } else if (this->config["hidden_act"] == "relu") {
        hidden = cuda_matrix_ops.relu(hidden);
    } else if (this->config["hidden_act"] == "swish") {
        hidden = cuda_matrix_ops.swish(hidden);
    } else {
        throw std::runtime_error("Unsupported activation function");
    }

    // 第二层线性变换 - 替换为CUDA矩阵乘
    std::vector<float> output = cuda_matrix_ops.matrix_multiply_transpose(
        hidden, weight_ff_2, batch_seq_len, intermediate_dim, embedding_dim
    );
    
    // 添加第二层偏置
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] += bias_ff_2[i % embedding_dim];
    }

    // 残差连接 + LayerNorm - 替换为CUDA层归一化
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] += input[i];  // 残差连接
    }
    output = cuda_matrix_ops.layer_norm(output, ln_gamma, ln_beta, embedding_dim, epsilon);

    return output;
}

std::vector<float> SentenceTransformerImpl::add_position_encoding(const std::vector<float>& input, 
                                                                  size_t seq_len, 
                                                                  size_t embedding_dim) {
    if (input.size() != seq_len * embedding_dim) {
        std::cout << "input.size(): " << input.size() << std::endl;
        std::cout << "seq_len * embedding_dim: " << seq_len * embedding_dim << std::endl;
        throw std::runtime_error("Input size does not match seq_len * embedding_dim");
    }
    std::vector<float> output = input;
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < embedding_dim; ++i) {
            float angle = static_cast<float>(pos) / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(embedding_dim));
            if (i % 2 == 0) {
                output[pos * embedding_dim + i] += std::sin(angle);
            } else {
                output[pos * embedding_dim + i] += std::cos(angle);
            }
        }
    }
    return output;
}