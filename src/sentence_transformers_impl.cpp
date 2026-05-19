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


// 使用新的 tokenizer 进行分词
std::vector<int32_t> SentenceTransformerImpl::tokenize(const std::string& text) {
    std::vector<int> token_ids;
    if (model_type == ModelType::BGE_M3 && xlmr_tokenizer) {
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

// 简单的嵌入层前向传播
std::vector<float> SentenceTransformerImpl::embedding_layer(const std::vector<int32_t>& tokens) {
    printf("DEBUG: embedding_layer called with %zu tokens\n", tokens.size());
    // 调整输出向量大小为 tokens.size() * embedding_dim
    std::vector<float> output(tokens.size() * embedding_dim, 0.0f);
    
    // 根据模型类型选择正确的词嵌入权重键名
    std::string embed_weight_key;
    if (model_type == ModelType::QWEN3) {
        embed_weight_key = "embed_tokens.weight";
    } else {
        embed_weight_key = "embeddings.word_embeddings.weight";
    }
    
    const std::vector<float>& embedding_weight = weights[embed_weight_key];
    
    // 调试输出：打印词嵌入权重的前几个值和token IDs
    if (tokens.size() > 0) {
        printf("DEBUG: embed_weight_key=%s, size=%zu\n", embed_weight_key.c_str(), embedding_weight.size());
        printf("DEBUG: tokens[0]=%d, tokens[1]=%d\n", tokens[0], tokens.size() > 1 ? tokens[1] : -1);
        
        // 打印第一个token对应的权重前5个值
        if (tokens[0] >= 0 && static_cast<size_t>(tokens[0]) < embedding_weight.size() / embedding_dim) {
            size_t start_idx = tokens[0] * embedding_dim;
            printf("DEBUG: embed_weight[token0][:5] = [");
            for (size_t j = 0; j < 5; ++j) {
                printf("%.6f%s", embedding_weight[start_idx + j], j < 4 ? ", " : "");
            }
            printf("]\n");
        }
    }

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

SentenceTransformerImpl::SentenceTransformerImpl(const std::string& model_path, const std::string& tokenizer_path) {
    std::string config_path = model_path + "/config.json";
    
    // 加载config.json到类成员变量
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("无法打开 config.json 文件: " + config_path);
    }
    config_file >> this->config;
    
    // 从config.json中读取model_type并设置模型类型
    std::string model_type_str = "bge";
    if (this->config.contains("model_type")) {
        model_type_str = this->config["model_type"];
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
    } else {
        // 传统BGE模型使用BGETokenizer
        bge_tokenizer = std::make_unique<BGETokenizer>(tokenizer_path + "/vocab.txt");
        std::string tokenizer_config_path = tokenizer_path + "/tokenizer.json";
    }
    
    std::string tensor_path = model_path + "/model.safetensors";
    if (!load_single_safetensors_file(tensor_path, weights)) {
        throw std::runtime_error("加载模型权重失败");
    }
    
    // 设置嵌入维度
    const int vocab_size = this->config["vocab_size"];
    
    // 根据模型类型查找嵌入层权重
    std::string embedding_weight_name = "embeddings.word_embeddings.weight";
    if (model_type == ModelType::QWEN3) {
        // Qwen3模型使用不同的嵌入层权重名称
        embedding_weight_name = "embed_tokens.weight";
    } else if (weights.find(embedding_weight_name) == weights.end()) {
        throw std::runtime_error("未找到嵌入层权重");
    }

    embedding_dim = weights[embedding_weight_name].size() / vocab_size;
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
    embedded = cpu_matrix_ops.layer_norm(embedded, ln_weight, ln_bias, embedding_dim, epsilon);

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
        attn_output = cpu_matrix_ops.matrix_multiply_transpose(
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
        attn_output = cpu_matrix_ops.layer_norm(attn_output, ln_gamma, ln_beta, embedding_dim, epsilon);

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
    std::vector<float> pooled = cpu_matrix_ops.matrix_multiply_transpose(
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
    printf("DEBUG: qwen3_forward start, seq_len=%zu\n", tokens.size());
    // 获取词嵌入
    std::vector<float> embedded = embedding_layer(tokens);
    size_t seq_len = tokens.size();
    size_t hidden_size = embedding_dim;
    
    // 调试输出：打印embedding_layer的输出
    if (seq_len > 0) {
        size_t last_tok = (seq_len - 1) * hidden_size;
        printf("DEBUG: after embedding_layer[last][:5] = [");
        for (size_t j = 0; j < 5 && j < hidden_size; ++j) {
            printf("%.6f%s", embedded[last_tok + j], j < 4 ? ", " : "");
        }
        printf("]\n");
    }
    
    printf("DEBUG: after embedding, hidden_size=%zu\n", hidden_size);
    
    // 获取模型配置
    size_t num_layers = this->config["num_hidden_layers"];
    size_t num_attention_heads = this->config["num_attention_heads"];
    size_t num_key_value_heads = this->config.contains("num_key_value_heads") ? 
                                this->config["num_key_value_heads"].get<size_t>() : num_attention_heads;
    
    printf("DEBUG: num_layers=%zu, num_attn_heads=%zu, num_kv_heads=%zu\n", num_layers, num_attention_heads, num_key_value_heads);
    
    // Qwen3使用GQA，Q的维度可能不同于hidden_size
    std::string q_proj_key_check = "layers.0.self_attn.q_proj.weight";
    size_t q_output_dim = hidden_size;  // 默认值
    if (weights.find(q_proj_key_check) != weights.end()) {
        const auto& q_weight = weights[q_proj_key_check];
        q_output_dim = q_weight.size() / hidden_size;  // [q_output_dim, hidden_size]
        printf("DEBUG: q_weight.size()=%zu, q_output_dim=%zu\n", q_weight.size(), q_output_dim);
    }
    size_t head_dim = q_output_dim / num_attention_heads;
    size_t kv_output_dim = num_key_value_heads * head_dim;
    
    printf("DEBUG: head_dim=%zu, kv_output_dim=%zu\n", head_dim, kv_output_dim);
    
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
            
            // 调试输出：Layer 0的LN权重和输入
            if (layer == 0) {
                size_t last_embedded_before = (seq_len - 1) * hidden_size;
                printf("DEBUG Layer 0 before Input LN[last][:5] = [");
                for (size_t j = 0; j < 5 && j < hidden_size; ++j) {
                    printf("%.6f%s", embedded[last_embedded_before + j], j < 4 ? ", " : "");
                }
                printf("]\n");
                
                // 计算embedded的统计信息（仅最后一个token）
                float sum_sq = 0.0f;
                float min_val = embedded[last_embedded_before], max_val = embedded[last_embedded_before];
                for (size_t j = 0; j < hidden_size; ++j) {
                    float val = embedded[last_embedded_before + j];
                    sum_sq += val * val;
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
                float rms = std::sqrt(sum_sq / hidden_size);
                printf("DEBUG Layer 0 last token stats: RMS=%.6f, min=%.6f, max=%.6f\n", rms, min_val, max_val);
                
                printf("DEBUG Layer 0 found input_layernorm.weight, size=%zu\n", ln_weight.size());
                printf("DEBUG Layer 0 input_layernorm.weight[:5] = [");
                for (size_t j = 0; j < 5 && j < ln_weight.size(); ++j) {
                    printf("%.6f%s", ln_weight[j], j < 4 ? ", " : "");
                }
                printf("]\n");
            }
            
            embedded = cpu_matrix_ops.rms_norm(embedded, ln_weight, hidden_size, eps);
        } else {
            if (layer == 0) {
                printf("DEBUG Layer 0 ERROR: input_layernorm.weight NOT FOUND! Key: %s\n", input_layernorm_weight_key.c_str());
            }
        }

        // 调试输出：Layer 0 input LN后的值
        if (layer == 0) {
            size_t last_embedded = (seq_len - 1) * hidden_size;
            printf("DEBUG Layer 0 after Input LN[last][:5] = [");
            for (size_t j = 0; j < 5 && j < hidden_size; ++j) {
                printf("%.6f%s", embedded[last_embedded + j], j < 4 ? ", " : "");
            }
            printf("]\n");
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
            
            // 调试输出：打印权重形状和前几个值
            if (layer == 0) {
                printf("DEBUG Layer 0 q_proj_weight.size()=%zu (expected %zu)\n", q_proj_weight.size(), q_output_dim * hidden_size);
                printf("DEBUG Layer 0 k_proj_weight.size()=%zu (expected %zu)\n", k_proj_weight.size(), kv_output_dim * hidden_size);
                printf("DEBUG Layer 0 q_proj_weight[:5] = [");
                for (size_t i = 0; i < 5 && i < q_proj_weight.size(); ++i) {
                    printf("%.6f%s", q_proj_weight[i], i < 4 ? ", " : "");
                }
                printf("]\n");
            }
            
            // 获取Qwen3特有的查询和键归一化权重
            std::string q_norm_key = layer_prefix + "self_attn.q_norm.weight";
            std::string k_norm_key = layer_prefix + "self_attn.k_norm.weight";
            
            if (weights.find(q_norm_key) != weights.end() && weights.find(k_norm_key) != weights.end()) {
                const std::vector<float>& q_norm_weight = weights[q_norm_key];
                const std::vector<float>& k_norm_weight = weights[k_norm_key];
                
                // 执行Q、K、V投影（使用正确的维度）
                // Q: [seq_len, hidden_size] x [hidden_size, q_output_dim]
                std::vector<float> q_states = cpu_matrix_ops.matrix_multiply_transpose(
                    embedded, q_proj_weight, seq_len, hidden_size, q_output_dim
                );
                
                // K/V: [seq_len, hidden_size] x [hidden_size, kv_output_dim]
                std::vector<float> k_states = cpu_matrix_ops.matrix_multiply_transpose(
                    embedded, k_proj_weight, seq_len, hidden_size, kv_output_dim
                );
                
                std::vector<float> v_states = cpu_matrix_ops.matrix_multiply_transpose(
                    embedded, v_proj_weight, seq_len, hidden_size, kv_output_dim
                );
                
                // 调试输出：打印embedded最后一个token的前5个值
                if (layer == 0) {
                    size_t last_embedded = (seq_len - 1) * hidden_size;
                    printf("DEBUG Layer 0 embedded[last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", embedded[last_embedded + j], j < 4 ? ", " : "");
                    printf("]\n");
                    
                    printf("DEBUG Layer 0 Q[last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", q_states[(seq_len - 1) * q_output_dim + j], j < 4 ? ", " : "");
                    printf("]\n");
                    printf("DEBUG Layer 0 K[last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", k_states[(seq_len - 1) * kv_output_dim + j], j < 4 ? ", " : "");
                    printf("]\n");
                    printf("DEBUG Layer 0 V[last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", v_states[(seq_len - 1) * kv_output_dim + j], j < 4 ? ", " : "");
                    printf("]\n");
                }
                
                // Reshape Q/K/V为多头格式: [seq_len, num_heads, head_dim]
                // 存储为 [seq_len * num_heads * head_dim]
                std::vector<float> q_reshaped(seq_len * num_attention_heads * head_dim);
                std::vector<float> k_reshaped(seq_len * num_key_value_heads * head_dim);
                std::vector<float> v_reshaped(seq_len * num_key_value_heads * head_dim);
                
                printf("DEBUG: q_output_dim=%zu, kv_output_dim=%zu, head_dim=%zu\n", q_output_dim, kv_output_dim, head_dim);
                printf("DEBUG: q_reshaped size=%zu, expected=%zu\n", q_reshaped.size(), seq_len * num_attention_heads * head_dim);
                
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t h = 0; h < num_attention_heads; ++h) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t src_idx = i * q_output_dim + h * head_dim + d;
                            size_t dst_idx = i * num_attention_heads * head_dim + h * head_dim + d;
                            if (src_idx < q_states.size() && dst_idx < q_reshaped.size()) {
                                q_reshaped[dst_idx] = q_states[src_idx];
                            }
                        }
                    }
                    for (size_t h = 0; h < num_key_value_heads; ++h) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t src_idx_k = i * kv_output_dim + h * head_dim + d;
                            size_t dst_idx_k = i * num_key_value_heads * head_dim + h * head_dim + d;
                            if (src_idx_k < k_states.size() && dst_idx_k < k_reshaped.size()) {
                                k_reshaped[dst_idx_k] = k_states[src_idx_k];
                            }
                            
                            size_t src_idx_v = i * kv_output_dim + h * head_dim + d;
                            size_t dst_idx_v = i * num_key_value_heads * head_dim + h * head_dim + d;
                            if (src_idx_v < v_states.size() && dst_idx_v < v_reshaped.size()) {
                                v_reshaped[dst_idx_v] = v_states[src_idx_v];
                            }
                        }
                    }
                }
                
                // 应用q_norm/k_norm（在RoPE之前）
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t h = 0; h < num_attention_heads; ++h) {
                        float norm_factor = 0.0f;
                        size_t start = i * num_attention_heads * head_dim + h * head_dim;
                        
                        for (size_t d = 0; d < head_dim; ++d) {
                            norm_factor += q_reshaped[start + d] * q_reshaped[start + d];
                        }
                        norm_factor = 1.0f / std::sqrt(norm_factor / head_dim + 1e-6f);
                        
                        for (size_t d = 0; d < head_dim; ++d) {
                            q_reshaped[start + d] *= norm_factor * q_norm_weight[d];
                        }
                    }
                    
                    for (size_t h = 0; h < num_key_value_heads; ++h) {
                        float norm_factor = 0.0f;
                        size_t start = i * num_key_value_heads * head_dim + h * head_dim;
                        
                        for (size_t d = 0; d < head_dim; ++d) {
                            norm_factor += k_reshaped[start + d] * k_reshaped[start + d];
                        }
                        norm_factor = 1.0f / std::sqrt(norm_factor / head_dim + 1e-6f);
                        
                        for (size_t d = 0; d < head_dim; ++d) {
                            k_reshaped[start + d] *= norm_factor * k_norm_weight[d];
                        }
                    }
                }
                
                // 调试输出：q_norm后
                if (layer == 0) {
                    size_t last_q = (seq_len - 1) * num_attention_heads * head_dim;
                    size_t last_k = (seq_len - 1) * num_key_value_heads * head_dim;
                    printf("DEBUG Layer 0 after q_norm Q[head0][last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", q_reshaped[last_q + j], j < 4 ? ", " : "");
                    printf("]\n");
                    printf("DEBUG Layer 0 after q_norm Q[head0][last][64:69] = [");
                    for (size_t j = 64; j < 69; ++j) printf("%.6f%s", q_reshaped[last_q + j], j < 68 ? ", " : "");
                    printf("]\n");
                    printf("DEBUG Layer 0 after k_norm K[head0][last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", k_reshaped[last_k + j], j < 4 ? ", " : "");
                    printf("]\n");
                }
                
                // 应用RoPE
                apply_rope(q_reshaped, k_reshaped, seq_len, 
                          num_attention_heads * head_dim, num_attention_heads,
                          num_key_value_heads * head_dim, num_key_value_heads);
                
                // 调试输出：RoPE后
                if (layer == 0) {
                    size_t last_q = (seq_len - 1) * num_attention_heads * head_dim;
                    size_t last_k = (seq_len - 1) * num_key_value_heads * head_dim;
                    printf("DEBUG Layer 0 after RoPE Q[head0][last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", q_reshaped[last_q + j], j < 4 ? ", " : "");
                    printf("]\n");
                    printf("DEBUG Layer 0 after RoPE Q[head0][last][64:69] = [");
                    for (size_t j = 64; j < 69; ++j) printf("%.6f%s", q_reshaped[last_q + j], j < 68 ? ", " : "");
                    printf("]\n");
                    printf("DEBUG Layer 0 after RoPE K[head0][last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", k_reshaped[last_k + j], j < 4 ? ", " : "");
                    printf("]\n");
                }
                
                // GQA注意力计算
                std::vector<float> attn_output(seq_len * q_output_dim, 0.0f);
                
                // 对每个Q头计算attention
                for (size_t h = 0; h < num_attention_heads; ++h) {
                    // GQA: 确定对应的K/V头索引
                    size_t kv_head_idx = h / (num_attention_heads / num_key_value_heads);
                    
                    // 提取当前Q头 [seq_len, head_dim]
                    std::vector<float> q_head(seq_len * head_dim);
                    for (size_t i = 0; i < seq_len; ++i) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            q_head[i * head_dim + d] = q_reshaped[i * num_attention_heads * head_dim + h * head_dim + d];
                        }
                    }
                    
                    // 提取对应的K头 [seq_len, head_dim]
                    std::vector<float> k_head(seq_len * head_dim);
                    for (size_t i = 0; i < seq_len; ++i) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            k_head[i * head_dim + d] = k_reshaped[i * num_key_value_heads * head_dim + kv_head_idx * head_dim + d];
                        }
                    }
                    
                    // 提取对应的V头 [seq_len, head_dim]
                    std::vector<float> v_head(seq_len * head_dim);
                    for (size_t i = 0; i < seq_len; ++i) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            v_head[i * head_dim + d] = v_reshaped[i * num_key_value_heads * head_dim + kv_head_idx * head_dim + d];
                        }
                    }
                    
                    // 计算attention scores: Q @ K^T / sqrt(head_dim)
                    std::vector<float> scores(seq_len * seq_len, 0.0f);
                    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                    
                    for (size_t i = 0; i < seq_len; ++i) {
                        for (size_t j = 0; j < seq_len; ++j) {
                            float score = 0.0f;
                            for (size_t d = 0; d < head_dim; ++d) {
                                score += q_head[i * head_dim + d] * k_head[j * head_dim + d];
                            }
                            scores[i * seq_len + j] = score * scale;
                        }
                    }
                    
                    // Softmax
                    for (size_t i = 0; i < seq_len; ++i) {
                        float max_score = scores[i * seq_len];
                        for (size_t j = 1; j < seq_len; ++j) {
                            if (scores[i * seq_len + j] > max_score) {
                                max_score = scores[i * seq_len + j];
                            }
                        }
                        
                        float sum = 0.0f;
                        for (size_t j = 0; j < seq_len; ++j) {
                            scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_score);
                            sum += scores[i * seq_len + j];
                        }
                        
                        for (size_t j = 0; j < seq_len; ++j) {
                            scores[i * seq_len + j] /= sum;
                        }
                    }
                    
                    // Attention output: scores @ V
                    std::vector<float> attn_head(seq_len * head_dim, 0.0f);
                    for (size_t i = 0; i < seq_len; ++i) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            float val = 0.0f;
                            for (size_t j = 0; j < seq_len; ++j) {
                                val += scores[i * seq_len + j] * v_head[j * head_dim + d];
                            }
                            attn_head[i * head_dim + d] = val;
                        }
                    }
                    
                    // 将结果放入attn_output
                    for (size_t i = 0; i < seq_len; ++i) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            attn_output[i * q_output_dim + h * head_dim + d] = attn_head[i * head_dim + d];
                        }
                    }
                }
                
                // Reshape attn_output back to [seq_len, q_output_dim]
                // 已经是正确的格式了
                
                // 应用输出投影 O: [seq_len, q_output_dim] × [hidden_size, q_output_dim]^T
                std::vector<float> projected_output = cpu_matrix_ops.matrix_multiply_transpose(
                    attn_output, o_proj_weight, seq_len, q_output_dim, hidden_size
                );
                
                // 调试输出：打印Layer 0的attn_output和projected_output
                if (layer == 0) {
                    printf("DEBUG Layer 0 attn_output stats: min=%.6f, max=%.6f, mean=%.6f\n",
                           *std::min_element(attn_output.begin(), attn_output.end()),
                           *std::max_element(attn_output.begin(), attn_output.end()),
                           std::accumulate(attn_output.begin(), attn_output.end(), 0.0f) / attn_output.size());
                    
                    printf("DEBUG Layer 0 attn_output[first][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", attn_output[j], j < 4 ? ", " : "");
                    printf("]\n");
                    
                    size_t last_attn = (seq_len - 1) * q_output_dim;
                    printf("DEBUG Layer 0 attn_output[last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", attn_output[last_attn + j], j < 4 ? ", " : "");
                    printf("]\n");

                    // projected_output[first][:5]
                    printf("DEBUG Layer %zu projected_output[first][:5] = [", layer);
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", projected_output[j], j < 4 ? ", " : "");
                    printf("]\n");
                    
                    size_t last_proj = (seq_len - 1) * hidden_size;
                    printf("DEBUG Layer 0 projected_output[last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", projected_output[last_proj + j], j < 4 ? ", " : "");
                    printf("]\n");
                }
                
                // 将注意力输出加到残差输入上（不是LN后的值）
                for (size_t i = 0; i < embedded.size(); ++i) {
                    embedded[i] = residual[i] + projected_output[i];
                }
                
                // 保存注意力残差后的值，用于MLP残差连接
                hidden_after_attn = embedded;
                
                // 调试输出：残差连接后的值
                if (layer == 0) {
                    size_t last_embedded = (seq_len - 1) * hidden_size;
                    printf("DEBUG Layer 0 After residual[last][:5] = [");
                    for (size_t j = 0; j < 5; ++j) printf("%.6f%s", embedded[last_embedded + j], j < 4 ? ", " : "");
                    printf("]\n");
                }
            }
        }
        
        // 3. 后注意力层RMS归一化（在MLP之前应用）
        std::string post_attn_layernorm_key = layer_prefix + "post_attention_layernorm.weight";
        if (weights.find(post_attn_layernorm_key) != weights.end()) {
            const std::vector<float>& ln_weight = weights[post_attn_layernorm_key];
            float eps = this->config.value("rms_norm_eps", 1e-6f);
            
            embedded = cpu_matrix_ops.rms_norm(embedded, ln_weight, hidden_size, eps);
            
            // 调试输出：post-attention LN后的值
            if (layer == 0) {
                size_t last_embedded = (seq_len - 1) * hidden_size;
                printf("DEBUG Layer 0 After post-attn LN[last][:5] = [");
                for (size_t j = 0; j < 5; ++j) printf("%.6f%s", embedded[last_embedded + j], j < 4 ? ", " : "");
                printf("]\n");
            }
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
            
            // 执行MLP前馈网络
            // gate_proj: [seq_len, hidden_size] × [hidden_size, intermediate_size]
            std::vector<float> gate_output = cpu_matrix_ops.matrix_multiply_transpose(
                embedded, gate_proj_weight, seq_len, hidden_size, intermediate_size
            );
            
            // up_proj: [seq_len, hidden_size] × [hidden_size, intermediate_size]
            std::vector<float> up_output = cpu_matrix_ops.matrix_multiply_transpose(
                embedded, up_proj_weight, seq_len, hidden_size, intermediate_size
            );
            
            // 调试输出：Gate和Up投影后的值
            if (layer == 0) {
                size_t last_gate = (seq_len - 1) * intermediate_size;
                printf("DEBUG Layer 0 Gate output[last][:5] = [");
                for (size_t j = 0; j < 5; ++j) printf("%.6f%s", gate_output[last_gate + j], j < 4 ? ", " : "");
                printf("]\n");
                
                printf("DEBUG Layer 0 Up output[last][:5] = [");
                for (size_t j = 0; j < 5; ++j) printf("%.6f%s", up_output[last_gate + j], j < 4 ? ", " : "");
                printf("]\n");
            }
            
            // 应用SiLU激活函数: x * sigmoid(x)
            for (size_t i = 0; i < gate_output.size(); ++i) {
                gate_output[i] = gate_output[i] * (1.0f / (1.0f + std::exp(-gate_output[i])));
            }
            
            // 调试输出：SiLU激活后的值
            if (layer == 0) {
                size_t last_gate = (seq_len - 1) * intermediate_size;
                printf("DEBUG Layer 0 Gate after SiLU[last][:5] = [");
                for (size_t j = 0; j < 5; ++j) printf("%.6f%s", gate_output[last_gate + j], j < 4 ? ", " : "");
                printf("]\n");
            }
            
            // 逐元素相乘
            for (size_t i = 0; i < gate_output.size(); ++i) {
                gate_output[i] *= up_output[i];
            }
            
            // 调试输出：gate*up后的值
            if (layer == 0) {
                size_t last_intermediate = (seq_len - 1) * intermediate_size;
                printf("DEBUG Layer 0 After gate*up[last][:5] = [");
                for (size_t j = 0; j < 5; ++j) printf("%.6f%s", gate_output[last_intermediate + j], j < 4 ? ", " : "");
                printf("]\n");
            }
            
            // down_proj: [seq_len, intermediate_size] × [intermediate_size, hidden_size]
            std::vector<float> mlp_output = cpu_matrix_ops.matrix_multiply_transpose(
                gate_output, down_proj_weight, seq_len, intermediate_size, hidden_size
            );
            
            // 调试输出：down_proj权重形状和前几个值
            if (layer == 0) {
                printf("DEBUG Layer 0 down_proj_weight.size()=%zu (expected %zu)\n", 
                       down_proj_weight.size(), intermediate_size * hidden_size);
                printf("DEBUG Layer 0 down_proj_weight[:5] = [");
                for (size_t i = 0; i < 5 && i < down_proj_weight.size(); ++i) {
                    printf("%.6f%s", down_proj_weight[i], i < 4 ? ", " : "");
                }
                printf("]\n");
            }
            
            // 调试输出：MLP输出
            if (layer == 0) {
                size_t last_mlp = (seq_len - 1) * hidden_size;
                printf("DEBUG Layer 0 MLP output[last][:5] = [");
                for (size_t j = 0; j < 5; ++j) printf("%.6f%s", mlp_output[last_mlp + j], j < 4 ? ", " : "");
                printf("]\n");
            }
            
            // 将MLP输出加到注意力残差后的值上（不是LN后的值）
            for (size_t i = 0; i < embedded.size(); ++i) {
                embedded[i] = hidden_after_attn[i] + mlp_output[i];
            }
            
            // 调试输出：MLP残差连接后
            if (layer == 0) {
                size_t last_embedded = (seq_len - 1) * hidden_size;
                printf("DEBUG Layer 0 After MLP residual[last][:5] = [");
                for (size_t j = 0; j < 5; ++j) printf("%.6f%s", embedded[last_embedded + j], j < 4 ? ", " : "");
                printf("]\n");
            }
        }
        
        // 调试输出：每层结束后的值（作为下一层的输入）
        if (layer == 0 || layer == 1) {
            size_t last_embedded = (seq_len - 1) * hidden_size;
            printf("DEBUG Layer %zu output (Layer %zu input)[last][:5] = [",
                   layer, layer + 1);
            for (size_t j = 0; j < 5; ++j) printf("%.6f%s", embedded[last_embedded + j], j < 4 ? ", " : "");
            printf("]\n");
        }
    }
    
    // 最后的RMS层归一化
    if (weights.find("norm.weight") != weights.end()) {
        const std::vector<float>& final_norm_weight = weights["norm.weight"];
        float eps = this->config.value("rms_norm_eps", 1e-6f); // 使用配置中的rms_norm_eps值
        
        embedded = cpu_matrix_ops.rms_norm(embedded, final_norm_weight, hidden_size, eps);
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
    
    return pooled_output;
}

std::vector<float> SentenceTransformerImpl::encode(const std::string& text) {
    // 分词
    std::vector<int32_t> tokens = tokenize(text);
    // 打印tokens
    std::cout << "tokens: " << std::endl;
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    // 根据模型类型选择前向传播方法
    if (model_type == ModelType::QWEN3) {
        return qwen3_forward(tokens);
    } else {
        return bge_forward(tokens);
    }
}

std::vector<std::vector<float>> SentenceTransformerImpl::encode_batch(const std::vector<std::string>& texts) {
    std::vector<std::vector<float>> result;
    for (const auto& text : texts) {
        result.push_back(encode(text));
    }
    return result;
}

size_t SentenceTransformerImpl::get_embedding_dimension() const {
    return embedding_dim;
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
    return cpu_attention_ops.multi_head_attention(input, weight_q, weight_k, weight_v, query_bias, key_bias, value_bias, num_heads, embedding_dim);
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
    std::vector<float> hidden = cpu_matrix_ops.matrix_multiply_transpose(
        input, weight_ff_1, batch_seq_len, embedding_dim, intermediate_dim
    );
    
    // 添加第一层偏置
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] += bias_ff_1[i % intermediate_dim];
    }

    // 激活函数 - 替换为CUDA版本
    if (this->config["hidden_act"] == "gelu") {
        hidden = cpu_matrix_ops.gelu(hidden);
    } else if (this->config["hidden_act"] == "relu") {
        hidden = cpu_matrix_ops.relu(hidden);
    } else if (this->config["hidden_act"] == "swish") {
        hidden = cpu_matrix_ops.swish(hidden);
    } else {
        throw std::runtime_error("Unsupported activation function");
    }

    // 第二层线性变换 - 替换为CUDA矩阵乘
    std::vector<float> output = cpu_matrix_ops.matrix_multiply_transpose(
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
    output = cpu_matrix_ops.layer_norm(output, ln_gamma, ln_beta, embedding_dim, epsilon);

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