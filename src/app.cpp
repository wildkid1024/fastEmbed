#include "sentence_transformers.h"
#include "safetensors.h"
#include "flashinfer/flashinfer.h"  // FlashInfer库
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <vector>

// 前向声明内部结构
struct ModelParams;
struct Tokenizer;

// 实现类，隐藏具体细节
class SentenceTransformerImpl {
public:
    // 构造函数，加载模型
    explicit SentenceTransformerImpl(const std::string& model_path) {
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("模型路径不存在: " + model_path);
        }

        // 加载模型参数
        params = load_model_params(model_path);
        if (!params) {
            throw std::runtime_error("无法加载模型参数");
        }

        // 加载分词器
        tokenizer = load_tokenizer(model_path);
        if (!tokenizer) {
            throw std::runtime_error("无法加载分词器");
        }

        // 加载safetensors格式的模型权重
        if (!load_safetensors_weights(model_path)) {
            throw std::runtime_error("无法加载模型权重");
        }

        embedding_dim = params->embedding_dim;

        // 初始化FlashInfer注意力引擎
        init_flashinfer_engine();
    }

    // 析构函数，释放资源
    ~SentenceTransformerImpl() {
        if (tokenizer) {
            free_tokenizer(tokenizer);
            tokenizer = nullptr;
        }

        if (params) {
            delete params;
            params = nullptr;
        }

        // 释放safetensors权重
        weights.clear();
    }

    // 生成单个文本的嵌入
    std::vector<float> encode(const std::string& text) {
        // 1. 文本分词
        std::vector<int> tokens = tokenize(text);
        if (tokens.empty()) {
            throw std::runtime_error("文本分词失败");
        }

        // 2. 张量准备（词嵌入 + 位置嵌入）
        std::vector<float> hidden_states = prepare_input_embeddings(tokens);
        if (hidden_states.empty()) {
            throw std::runtime_error("无法准备输入嵌入");
        }

        // 3. 模型推理（Transformer层）
        for (int i = 0; i < params->num_layers; ++i) {
            hidden_states = transformer_layer(hidden_states, tokens.size(), i);
        }

        // 4. 池化层（获取句子嵌入）
        std::vector<float> embedding = pooling_layer(hidden_states, tokens.size());
        
        return embedding;
    }

    // 批量生成嵌入
    std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts) {
        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(texts.size());

        for (const auto& text : texts) {
            embeddings.push_back(encode(text));
        }

        return embeddings;
    }

    // 获取嵌入维度
    size_t get_embedding_dimension() const {
        return embedding_dim;
    }

    // 禁用拷贝
    SentenceTransformerImpl(const SentenceTransformerImpl&) = delete;
    SentenceTransformerImpl& operator=(const SentenceTransformerImpl&) = delete;

    // 允许移动
    SentenceTransformerImpl(SentenceTransformerImpl&& other) noexcept
        : params(other.params),
          tokenizer(other.tokenizer),
          weights(std::move(other.weights)),
          embedding_dim(other.embedding_dim),
          flashinfer_engine(std::move(other.flashinfer_engine)) {
        other.params = nullptr;
        other.tokenizer = nullptr;
        other.embedding_dim = 0;
    }

    SentenceTransformerImpl& operator=(SentenceTransformerImpl&& other) noexcept {
        if (this != &other) {
            // 释放当前资源
            cleanup();

            // 移动资源
            params = other.params;
            tokenizer = other.tokenizer;
            weights = std::move(other.weights);
            embedding_dim = other.embedding_dim;
            flashinfer_engine = std::move(other.flashinfer_engine);

            // 清空源对象
            other.params = nullptr;
            other.tokenizer = nullptr;
            other.embedding_dim = 0;
        }
        return *this;
    }

private:
    // 模型参数结构
    struct ModelParams {
        int embedding_dim;
        int hidden_size;
        int num_layers;
        int num_attention_heads;
        int max_position_embeddings;
        std::string model_type;
    };

    // 分词器结构
    struct Tokenizer {
        std::unordered_map<std::string, int> vocab;
        std::unordered_map<int, std::string> vocab_inv;
    };

    // 内部资源
    ModelParams* params = nullptr;
    Tokenizer* tokenizer = nullptr;
    std::unordered_map<std::string, std::vector<float>> weights;  // 存储safetensors权重
    size_t embedding_dim = 0;
    flashinfer::FlashInferEngine flashinfer_engine;  // FlashInfer引擎

    // 清理资源
    void cleanup() {
        if (tokenizer) {
            free_tokenizer(tokenizer);
            tokenizer = nullptr;
        }
        if (params) {
            delete params;
            params = nullptr;
        }
        weights.clear();
        embedding_dim = 0;
    }

    // 初始化FlashInfer引擎
    void init_flashinfer_engine() {
        // 配置FlashInfer引擎
        flashinfer::FlashInferConfig config;
        config.num_heads = params->num_attention_heads;
        config.head_dim = params->hidden_size / params->num_attention_heads;
        config.max_seq_len = params->max_position_embeddings;
        config.dtype = flashinfer::DataType::kFloat32;  // 或kFloat16根据模型类型
        config.device = flashinfer::Device::kGPU;  // 使用GPU加速
        
        // 初始化引擎
        if (!flashinfer_engine.init(config)) {
            throw std::runtime_error("FlashInfer引擎初始化失败");
        }
    }

    // 加载模型参数
    ModelParams* load_model_params(const std::string& model_path) {
        std::string config_path = model_path + "/config.json";
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            return nullptr;
        }

        // 实际实现中应解析JSON配置文件
        auto params = new ModelParams();
        // 从config.json读取实际参数（示例值）
        params->embedding_dim = 384;
        params->hidden_size = 384;
        params->num_layers = 6;
        params->num_attention_heads = 12;
        params->max_position_embeddings = 512;
        params->model_type = "bert";

        return params;
    }

    // 加载分词器
    Tokenizer* load_tokenizer(const std::string& model_path) {
        Tokenizer* tokenizer = new Tokenizer();
        std::string vocab_path = model_path + "/vocab.txt";
        
        std::ifstream vocab_file(vocab_path);
        if (vocab_file.is_open()) {
            std::string line;
            int id = 0;
            while (std::getline(vocab_file, line)) {
                tokenizer->vocab[line] = id;
                tokenizer->vocab_inv[id] = line;
                id++;
            }
        }

        return tokenizer;
    }

    // 释放分词器
    void free_tokenizer(Tokenizer* tokenizer) {
        if (tokenizer) {
            tokenizer->vocab.clear();
            tokenizer->vocab_inv.clear();
            delete tokenizer;
        }
    }

    // 加载safetensors格式的模型权重
    bool load_safetensors_weights(const std::string& model_path) {
        // 遍历所有safetensors文件
        for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
            if (entry.path().extension() == ".safetensors") {
                if (!load_single_safetensors_file(entry.path().string())) {
                    return false;
                }
            }
        }
        
        return !weights.empty();
    }

    // 加载单个safetensors文件
    bool load_single_safetensors_file(const std::string& file_path) {
        try {
            // 使用safetensors-cpp库加载文件
            safetensors::SafeTensors st = safetensors::load(file_path);
            
            // 遍历所有张量并存储
            for (const auto& [name, tensor] : st.tensors()) {
                // 将张量数据复制到我们的存储结构中
                std::vector<float> data(tensor.num_elements());
                std::memcpy(data.data(), tensor.data<float>(), 
                           tensor.num_elements() * sizeof(float));
                weights[name] = std::move(data);
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "加载safetensors文件失败: " << e.what() << std::endl;
            return false;
        }
    }

    // 文本分词
    std::vector<int> tokenize(const std::string& text) {
        std::vector<int> tokens;
        
        // 添加特殊标记
        tokens.push_back(101);  // [CLS]
        
        // 简单分词示例（实际应使用WordPiece或BPE分词）
        std::stringstream ss(text);
        std::string word;
        while (ss >> word) {
            if (tokenizer->vocab.find(word) != tokenizer->vocab.end()) {
                tokens.push_back(tokenizer->vocab[word]);
            } else {
                tokens.push_back(100);  // [UNK]
            }
        }
        
        // 添加特殊标记
        tokens.push_back(102);  // [SEP]
        
        return tokens;
    }

    // 准备输入嵌入（词嵌入 + 位置嵌入）
    std::vector<float> prepare_input_embeddings(const std::vector<int>& tokens) {
        const int seq_len = tokens.size();
        std::vector<float> embeddings(seq_len * params->hidden_size, 0.0f);
        
        // 获取权重
        const auto& word_emb = weights.at("bert.embeddings.word_embeddings.weight");
        const auto& position_emb = weights.at("bert.embeddings.position_embeddings.weight");
        const auto& token_type_emb = weights.at("bert.embeddings.token_type_embeddings.weight");
        
        // 词嵌入 + 位置嵌入 + 类型嵌入
        for (int i = 0; i < seq_len; ++i) {
            const int token_id = tokens[i];
            const int pos = i;
            
            // 词嵌入
            const float* word_ptr = &word_emb[token_id * params->hidden_size];
            // 位置嵌入
            const float* pos_ptr = &position_emb[pos * params->hidden_size];
            // 类型嵌入（默认0类型）
            const float* type_ptr = &token_type_emb[0 * params->hidden_size];
            
            // 合并嵌入
            for (int j = 0; j < params->hidden_size; ++j) {
                embeddings[i * params->hidden_size + j] = 
                    word_ptr[j] + pos_ptr[j] + type_ptr[j];
            }
        }
        
        // 应用LayerNorm
        return layer_norm(embeddings, seq_len, "bert.embeddings.LayerNorm");
    }

    // Transformer层实现
    std::vector<float> transformer_layer(const std::vector<float>& input, int seq_len, int layer_idx) {
        // 残差连接 + 多头注意力（使用FlashInfer加速）
        std::vector<float> attention_output = multi_head_attention(input, seq_len, layer_idx);
        std::vector<float> residual1 = add(input, attention_output);
        std::vector<float> norm1 = layer_norm(residual1, seq_len, 
            "bert.encoder.layer." + std::to_string(layer_idx) + ".attention.output.LayerNorm");
        
        // 残差连接 + 前馈网络
        std::vector<float> ff_output = feed_forward(norm1, seq_len, layer_idx);
        std::vector<float> residual2 = add(norm1, ff_output);
        std::vector<float> output = layer_norm(residual2, seq_len,
            "bert.encoder.layer." + std::to_string(layer_idx) + ".output.LayerNorm");
        
        return output;
    }

    // 多头注意力实现（使用FlashInfer）
    std::vector<float> multi_head_attention(const std::vector<float>& input, int seq_len, int layer_idx) {
        const int hidden_size = params->hidden_size;
        const int num_heads = params->num_attention_heads;
        const int head_size = hidden_size / num_heads;
        
        // 1. 线性投影得到Q, K, V
        std::vector<float> q = linear_projection(input, seq_len, 
            "bert.encoder.layer." + std::to_string(layer_idx) + ".attention.self.query");
        std::vector<float> k = linear_projection(input, seq_len, 
            "bert.encoder.layer." + std::to_string(layer_idx) + ".attention.self.key");
        std::vector<float> v = linear_projection(input, seq_len, 
            "bert.encoder.layer." + std::to_string(layer_idx) + ".attention.self.value");
        
        // 2. 使用FlashInfer计算注意力
        std::vector<float> attention_output(seq_len * hidden_size);
        
        // 准备FlashInfer输入
        flashinfer::AttentionInput attn_input;
        attn_input.q = q.data();
        attn_input.k = k.data();
        attn_input.v = v.data();
        attn_input.output = attention_output.data();
        attn_input.seq_len = seq_len;
        attn_input.batch_size = 1;
        
        // 对于自注意力，使用全1掩码
        std::vector<int> cu_seqlens = {0, seq_len};
        attn_input.cu_seqlens_q = cu_seqlens.data();
        attn_input.cu_seqlens_kv = cu_seqlens.data();
        attn_input.max_seq_len_q = seq_len;
        attn_input.max_seq_len_kv = seq_len;
        
        // 执行FlashInfer注意力计算
        if (!flashinfer_engine.forward(attn_input)) {
            throw std::runtime_error("FlashInfer注意力计算失败");
        }
        
        // 3. 输出投影
        return linear_projection(attention_output, seq_len, 
            "bert.encoder.layer." + std::to_string(layer_idx) + ".attention.output.dense");
    }

    // 前馈网络
    std::vector<float> feed_forward(const std::vector<float>& input, int seq_len, int layer_idx) {
        // 第一层线性变换 + Gelu激活
        std::vector<float> intermediate = linear_projection(input, seq_len,
            "bert.encoder.layer." + std::to_string(layer_idx) + ".intermediate.dense");
        intermediate = gelu(intermediate);
        
        // 第二层线性变换
        return linear_projection(intermediate, seq_len,
            "bert.encoder.layer." + std::to_string(layer_idx) + ".output.dense");
    }

    // 池化层（均值池化）
    std::vector<float> pooling_layer(const std::vector<float>& hidden_states, int seq_len) {
        std::vector<float> embedding(params->embedding_dim, 0.0f);
        const int hidden_size = params->hidden_size;
        
        // 对所有token的隐藏状态进行平均
        for (int d = 0; d < hidden_size; ++d) {
            float sum = 0.0f;
            for (int i = 0; i < seq_len; ++i) {
                sum += hidden_states[i * hidden_size + d];
            }
            embedding[d % params->embedding_dim] = sum / seq_len;
        }
        
        // 最终投影到嵌入维度
        if (params->hidden_size != params->embedding_dim) {
            // 使用输出投影层
            std::vector<float> projected(params->embedding_dim, 0.0f);
            const auto& weight = weights.at("sentence_transformers.model.SentenceTransformer.1.weight");
            const auto& bias = weights.at("sentence_transformers.model.SentenceTransformer.1.bias");
            
            for (int i = 0; i < params->embedding_dim; ++i) {
                float val = bias[i];
                for (int j = 0; j < hidden_size; ++j) {
                    val += embedding[j % params->embedding_dim] * weight[i * hidden_size + j];
                }
                projected[i] = val;
            }
            embedding = projected;
        }
        
        // L2归一化
        return l2_normalize(embedding);
    }

    // 辅助函数：线性投影
    std::vector<float> linear_projection(const std::vector<float>& input, int seq_len, const std::string& weight_name) {
        const int in_features = params->hidden_size;
        const int out_features = params->hidden_size;  // 对于BERT，大多数线性层保持维度不变
        
        const auto& weight = weights.at(weight_name + ".weight");
        const auto& bias = weights.at(weight_name + ".bias");
        
        std::vector<float> output(seq_len * out_features, 0.0f);
        
        // 计算 y = x * W^T + b
        for (int i = 0; i < seq_len; ++i) {
            for (int o = 0; o < out_features; ++o) {
                float val = bias[o];
                for (int i_feat = 0; i_feat < in_features; ++i_feat) {
                    val += input[i * in_features + i_feat] * weight[o * in_features + i_feat];
                }
                output[i * out_features + o] = val;
            }
        }
        
        return output;
    }

    // 辅助函数：LayerNorm
    std::vector<float> layer_norm(const std::vector<float>& input, int seq_len, const std::string& param_name) {
        const int hidden_size = params->hidden_size;
        const float eps = 1e-12f;
        
        const auto& gamma = weights.at(param_name + ".weight");
        const auto& beta = weights.at(param_name + ".bias");
        
        std::vector<float> output(input.size());
        
        for (int i = 0; i < seq_len; ++i) {
            // 计算均值
            float mean = 0.0f;
            for (int d = 0; d < hidden_size; ++d) {
                mean += input[i * hidden_size + d];
            }
            mean /= hidden_size;
            
            // 计算方差
            float var = 0.0f;
            for (int d = 0; d < hidden_size; ++d) {
                const float diff = input[i * hidden_size + d] - mean;
                var += diff * diff;
            }
            var /= hidden_size;
            
            // 归一化并应用缩放和偏移
            for (int d = 0; d < hidden_size; ++d) {
                output[i * hidden_size + d] = 
                    gamma[d] * (input[i * hidden_size + d] - mean) / std::sqrt(var + eps) + beta[d];
            }
        }
        
        return output;
    }

    // 辅助函数：GELU激活函数
    std::vector<float> gelu(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        
        for (size_t i = 0; i < input.size(); ++i) {
            const float x = input[i];
            // GELU近似: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            output[i] = 0.5f * x * (1.0f + std::tanh(
                sqrt_2_over_pi * (x + 0.044715f * x * x * x)
            ));
        }
        
        return output;
    }

    // 辅助函数：向量加法（残差连接）
    std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("向量长度不匹配，无法相加");
        }
        
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        
        return result;
    }

    // 辅助函数：L2归一化
    std::vector<float> l2_normalize(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        
        // 计算L2范数
        float norm = 0.0f;
        for (float val : input) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        // 防止除零
        if (norm < 1e-12f) {
            return output;
        }
        
        // 归一化
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = input[i] / norm;
        }
        
        return output;
    }
};

// SentenceTransformer 类的实现
SentenceTransformer::SentenceTransformer(const std::string& model_path)
    : impl(std::make_unique<SentenceTransformerImpl>(model_path)) {}

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

// 移动构造和赋值
SentenceTransformer::SentenceTransformer(SentenceTransformer&&) noexcept = default;
SentenceTransformer& SentenceTransformer::operator=(SentenceTransformer&&) noexcept = default;
    