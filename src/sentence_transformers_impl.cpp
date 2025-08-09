#include "sentence_transformers_impl.h"
#include <safetensors.h>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>

// 从 config.json 加载词汇表
void SentenceTransformerImpl::load_vocab(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("无法打开 config.json 文件: " + config_path);
    }

    json config;
    config_file >> config;

    // 取['model']['vocab']中的数据表
    // json vocab_json = config["model"]["vocab"];
    config = config["model"];
    
    if (config.contains("vocab") && config["vocab"].is_object()) {
        for (auto& [key, value] : config["vocab"].items()) {
            if (value.is_number_integer()) {
                vocab[key] = value.get<int64_t>();
            }
        }
    } else {
        throw std::runtime_error("config.json 中未找到有效的词汇表字段");
    }
}

// 使用新的 tokenizer 进行分词
std::vector<int32_t> SentenceTransformerImpl::tokenize(const std::string& text) {
    auto tokens = tokenizer.tokenize(text);
    auto token_ids = tokenizer.encode(tokens);
    token_ids.insert(token_ids.begin(), 101);
    token_ids.push_back(102);
    return std::vector<int32_t>(token_ids.begin(), token_ids.end());
}

// 简单的嵌入层前向传播
std::vector<float> SentenceTransformerImpl::embedding_layer(const std::vector<int32_t>& tokens) {
    // 调整输出向量大小为 tokens.size() * embedding_dim
    std::vector<float> output(tokens.size() * embedding_dim, 0.0f);
    const std::vector<float>& embedding_weight = weights["embeddings.word_embeddings.weight"];

    for (size_t i = 0; i < tokens.size(); ++i) {
        int32_t token = tokens[i];
        size_t start_idx = token * embedding_dim;
        // 复制当前 token 对应的嵌入向量到输出中
        for (size_t j = 0; j < embedding_dim; ++j) {
            output[i * embedding_dim + j] = embedding_weight[start_idx + j];
        }
    }
    return output;
}

SentenceTransformerImpl::SentenceTransformerImpl(const std::string& model_path, const std::string& tokenizer_path, const std::string& config_path)
    : tokenizer(model_path + "/vocab.txt") { // 假设词汇表文件在 model_path 下
    // 加载词汇表
    load_vocab(config_path);

    std::string tensor_path = model_path + "/model.safetensors";

    if (!load_single_safetensors_file(tensor_path, weights)) {  // 使用函数
        throw std::runtime_error("加载模型权重失败");
    }

    // 打印safetensor unordered_map 权重的key
    for (const auto& key : weights) {
        std::cout << key.first << std::endl;
    }

    // 假设嵌入层权重名为 "embedding.weight"
    if (weights.find("embeddings.word_embeddings.weight") != weights.end()) {
        // 假设嵌入维度是权重矩阵的列数
        embedding_dim = weights["embeddings.word_embeddings.weight"].size() / vocab.size();
        // 打印vocab size 和 embedding_dim
        std::cout << "vocab size: " << vocab.size() << std::endl;
        std::cout << "embedding_dim: " << embedding_dim << std::endl;
    } else {
        throw std::runtime_error("未找到嵌入层权重");
    }
}

// 完整的 BGE 前向传播
std::vector<float> SentenceTransformerImpl::bge_forward(const std::vector<int32_t>& tokens) {
    // 先通过嵌入层（词嵌入）
    std::vector<float> embedded = embedding_layer(tokens);
    size_t seq_len = tokens.size();

    // {{ 修改前：使用生成式位置编码 }}
    // embedded = add_position_encoding(embedded, seq_len, embedding_dim);
    // {{ 修改后：使用预训练位置嵌入 }}
    // 获取位置嵌入 (shape: [max_seq_len, embedding_dim])
    const std::vector<float>& pos_embeddings = weights["embeddings.position_embeddings.weight"];
    size_t max_seq_len = pos_embeddings.size() / embedding_dim;
    if (seq_len > max_seq_len) {
        throw std::runtime_error("序列长度超过位置嵌入的最大长度");
    }
    
    // 词嵌入 + 位置嵌入
    for (size_t i = 0; i < seq_len; ++i) {
        size_t pos_start = i * embedding_dim;
        for (size_t j = 0; j < embedding_dim; ++j) {
            embedded[pos_start + j] += pos_embeddings[pos_start + j];
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

    // 应用嵌入层 LayerNorm（词嵌入 + 位置嵌入 + 段嵌入 后的归一化）
    const std::vector<float>& ln_weight = weights["embeddings.LayerNorm.weight"];
    const std::vector<float>& ln_bias = weights["embeddings.LayerNorm.bias"];
    embedded = cpu_matrix_ops.layer_norm(embedded, ln_weight, ln_bias, embedding_dim);

    // {{ 预训练位置嵌入处理结束 }}
    
    // 假设 BGE 模型有 4 层 Transformer 编码器
    const int num_layers = 4;
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

        // 多头注意力机制
        // 修改多头注意力调用，传递偏置参数
        std::vector<float> attn_output = multi_head_attention(embedded, weight_q, weight_k, weight_v, query_bias, key_bias, value_bias, 8, embedding_dim);
        // {{ 新增：注意力输出 Dense 层（修复未使用 weight 的问题）}}
        // 1. 构造当前层的 attention.output.dense 权重名称
        
        // 3. 矩阵乘法：attn_output（[seq_len, embedding_dim]） × 权重（[embedding_dim, embedding_dim]）
        size_t batch_seq_len = attn_output.size() / embedding_dim;
        attn_output = cpu_matrix_ops.matrix_multiply_transpose(attn_output, attn_dense_weight, batch_seq_len, embedding_dim, embedding_dim);
        
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
        attn_output = cpu_matrix_ops.layer_norm(attn_output, ln_gamma, ln_beta, embedding_dim);

        const int intermediate_dim = 2048;

        // 前馈神经网络（修改调用参数）
        std::vector<float> ff_output = feed_forward_network(
            attn_output, 
            weight_ff_1, bias_ff_1, 
            weight_ff_2, bias_ff_2,
            ln_ff_gamma, ln_ff_beta,  // 传递LayerNorm参数
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

std::vector<float> SentenceTransformerImpl::encode(const std::string& text) {
    // 分词
    std::vector<int32_t> tokens = tokenize(text);
    // 打印tokens
    std::cout << "tokens: " << std::endl;
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    // BGE 模型前向传播
    return bge_forward(tokens);
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
                                                                  const std::vector<float>& ln_gamma,  // 新增LayerNorm参数
                                                                  const std::vector<float>& ln_beta,   // 新增LayerNorm参数
                                                                  size_t embedding_dim, 
                                                                  size_t intermediate_dim,
                                                                  float dropout_prob) {  // 新增dropout参数
    size_t batch_seq_len = input.size() / embedding_dim;

    // 第一层线性变换
    std::vector<float> hidden = cpu_matrix_ops.matrix_multiply_transpose(
        input, weight_ff_1, batch_seq_len, embedding_dim, intermediate_dim
    );
    
    // 添加第一层偏置
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] += bias_ff_1[i % intermediate_dim];
    }

    // 激活函数
    hidden = cpu_matrix_ops.gelu(hidden);

    // 第二层线性变换
    std::vector<float> output = cpu_matrix_ops.matrix_multiply_transpose(
        hidden, weight_ff_2, batch_seq_len, intermediate_dim, embedding_dim
    );
    
    // 添加第二层偏置
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] += bias_ff_2[i % embedding_dim];
    }

    // 添加Dropout层（需要在cpu_matrix_ops中实现）
    // output = cpu_matrix_ops.dropout(output, dropout_prob);

    // 残差连接 + LayerNorm（与PyTorch实现一致）
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] += input[i];  // 残差连接
    }
    output = cpu_matrix_ops.layer_norm(output, ln_gamma, ln_beta, embedding_dim);  // 层归一化

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