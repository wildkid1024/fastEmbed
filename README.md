# fastEmbed

fastEmbed 是一个高性能文本嵌入（Text Embedding）推理框架，采用纯 C++ 和 CUDA 编写，提供轻量级、低依赖的向量生成解决方案。

## 核心特性

- **原生模型加载**：直接读取 HuggingFace 原始模型文件（safetensors/bin 格式），无需任何格式转换
- **一键服务部署**：单命令启动 HTTP 服务，提供 OpenAI 兼容的 API 接口
- **极致轻量化**：编译后二进制文件仅约 2MB，无 Python 依赖
- **纯 C++/CUDA 实现**：直接对接底层硬件加速，支持 GPU 批量推理
- **隐私保护**：支持完全离线运行，数据无需上传云端
- **多模型支持**：内置 BGE 系列、Qwen3-Embedding 系列和 EmbeddingGemma 2

## 快速开始

### 三步即可用

```bash
# 1. 下载模型
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./models/qwen3

# 2. 编译项目
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DENABLE_SERVER=ON
make -j

# 3. 启动服务
./server --model_path ../models/qwen3 --serve_model_name qwen3
```

服务启动后，立即提供 OpenAI 兼容的 `/v1/embeddings` 接口。

## 技术规格

### 支持模型

| 模型系列 | 模型名称 | 语言支持 | 输出维度 | 最大序列长度 | 适用场景 |
|---------|---------|---------|---------|------------|---------|
| BGE | bge-small-zh-v1.5 | 中文 | 512 | 512 | 轻量级中文检索 |
| BGE | bge-base-zh-v1.5 | 中文 | 768 | 512 | 平衡性能与精度 |
| BGE | bge-large-zh-v1.5 | 中文 | 1024 | 512 | 高精度中文理解 |
| BGE | bge-m3 | 多语言 | 1024 | 512 | 多语言、多粒度检索 |
| Qwen3 | Qwen3-Embedding-0.6B | 中英双语 | 1024 | 8192 | 长文档理解、复杂推理 |
| EmbeddingGemma | embeddinggemma-2 | 多语言（文本塔） | 768 | 2048 | 高精度多语言检索、可截断维度 |

> EmbeddingGemma 2 说明：
> - 纯 C++/CUDA 复刻了官方 `EmbeddingGemma2Model` 的**文本塔**（24 层双向编码器 + PLE 每层输入 + `gelu_pytorch_tanh` MLP），输出 768 维，均值池化后 L2 归一化；
> - 支持官方内置的 20 个任务提示词（`query` / `Document` / `SearchQuery` / `STS` …），通过 `prompt_name` 参数启用；
> - 支持 Matryoshka 维度截断，客户端取前 768/512/256/128 维后**需重新归一化**（本服务默认返回完整 768 维并已归一化）；
> - 视觉/音频塔尚未实现，本服务仅支持文本输入。

### 系统要求

- **操作系统**：Linux（推荐 Ubuntu 18.04+）
- **编译器**：支持 C++17 的编译器（GCC 7+ 或 Clang 5+）
- **GPU 加速**：CUDA Toolkit 11.0+（可选，CPU 模式无需 CUDA）
- **构建工具**：CMake 3.15+
- **依赖库**：Asio（异步网络库）、nlohmann/json、sentencepiece

### 依赖安装

```bash
# 使用 conda 安装 Asio（推荐）
conda install -c conda-forge asio

# 或使用系统包管理器（Ubuntu/Debian）
sudo apt-get install libasio-dev

# 其他依赖通过 CMake 自动获取
```

## 使用指南

### 模型准备

从 HuggingFace 下载模型文件：

```bash
# BGE 系列模型
huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3

# Qwen3-Embedding 系列模型
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./models/qwen3

# EmbeddingGemma 2（需要 tokenizer.model + model.safetensors 等完整目录）
huggingface-cli download google/embeddinggemma-2 --local-dir ./models/embeddinggemma-2
```

### 启动 HTTP 服务

```bash
# 基本用法（默认端口 8080）
./build/server \
  --model_path ./models/bge-m3 \
  --serve_model_name bge-m3

# 自定义端口和主机
./build/server \
  --model_path ./models/qwen3 \
  --serve_model_name qwen3 \
  --host 127.0.0.1 \
  --port 5001
```

### API 调用示例

#### cURL

```bash
# 单文本嵌入
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "这是一个测试句子", "model": "bge-m3"}'

# 批量文本嵌入
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["句子1", "句子2", "句子3"], "model": "bge-m3"}'
```

#### Python

```python
import requests

API_URL = "http://localhost:8080/v1/embeddings"

# 单文本嵌入
response = requests.post(API_URL, json={
    "input": "这是一个测试句子",
    "model": "bge-m3"
})
embedding = response.json()["data"][0]["embedding"]
print(f"嵌入维度: {len(embedding)}")

# 批量文本嵌入
response = requests.post(API_URL, json={
    "input": ["句子1", "句子2", "句子3"],
    "model": "bge-m3"
})
embeddings = [item["embedding"] for item in response.json()["data"]]
print(f"批量生成 {len(embeddings)} 个嵌入向量")
```

#### 使用内置提示词（EmbeddingGemma 2）

EmbeddingGemma 2 训练时给文本加了任务指令前缀，检索场景下建议 query 和 document 分别使用不同前缀：

```bash
# query 侧
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "北极光是怎么形成的？", "prompt_name": "SearchQuery"}'

# document 侧（无标题时用 Document，前缀为 "title: none | text: "）
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "北极光由太阳带电粒子与高层大气相互作用产生。", "prompt_name": "Document"}'
```

`prompt_name` 可选值来自模型的 `config_sentence_transformers.json`，例如
`SearchQuery`、`Document`、`STS`、`Classification`、`Clustering`、`QuestionAnswering`、`CodeRetrieval` 等。
不传该字段则直接对原文编码（效果略降但可用）。C++ API 用法相同：

```cpp
SentenceTransformer model("./models/embeddinggemma-2");
auto query_emb = model.encode("北极光是怎么形成的？", "SearchQuery");
auto doc_emb   = model.encode("北极光由太阳带电粒子与高层大气相互作用产生。", "Document");
```

### 命令行工具
fastEmbed 还提供命令行工具用于快速测试：

```bash
# 交互式输入
echo "Hello World" | ./build/cli ./models/bge-m3

# 查看帮助
./build/cli --help
```

## 应用场景

### 1. 离线隐私保护

在移动设备或边缘设备上本地运行 embedding 推理，数据不出设备，完全保护用户隐私。

**典型应用**：
- 📱 **智能手机**：本地短信分类、联系人智能搜索、输入法预测
- 🏥 **医疗设备**：病历文本分析、医疗记录检索（符合 HIPAA 要求）
- 🚗 **车载系统**：语音助手本地理解、导航指令解析（保护位置隐私）
- 🏠 **智能家居**：家庭对话分析、隐私日程管理

**核心优势**：
- 数据零泄露：所有计算在本地完成
- 离线可用：无网络环境下仍可正常工作
- 低延迟：本地推理，毫秒级响应
- 合规性：满足 GDPR、HIPAA 等数据保护法规要求

### 2. 端侧多模态应用

结合视觉模型和文本 embedding 能力，在端侧构建完整的多模态 AI 应用。

**典型架构**：
```
输入（图片 + 文本）
    ↓
┌──────────────┐    ┌──────────────────┐
│ 视觉编码器    │    │ 文本编码器        │
│ (图像特征)    │    │ (文本向量)        │
└──────────────┘    └──────────────────┘
    ↓                    ↓
    └────→ 本地融合 ←────┘
         多模态检索/匹配
```

**典型场景**：
- **个人相册管理**：图像识别 + 文本描述 → 本地语义搜索
- **医疗影像辅助**：X光片分析 + 医生笔记理解 → 本地病例匹配
- **车载智能助手**：路况识别 + 语音指令理解 → 本地决策

### 3. 资源受限设备部署

fastEmbed 的轻量化设计完美适配存储和内存受限的嵌入式设备。

**资源占用**：
- 二进制大小：约 2MB
- 内存占用：取决于模型大小和输入文本长度
- 启动时间：毫秒级
- 依赖数量：仅需 CUDA 运行时（GPU 模式）或无额外依赖（CPU 模式）

**支持设备**：
- NVIDIA Jetson 系列（Nano/TX2/Xavier/Orin）
- 树莓派 + GPU 加速器
- 工业嵌入式 PC（x86_64 + NVIDIA GPU）
- 边缘服务器（低功耗 x86/ARM + GPU）


## 开发指南

### 从源码构建

```bash
# 克隆仓库
git clone https://github.com/your-repo/fastEmbed.git
cd fastEmbed

# 初始化子模块
git submodule update --init --recursive

# 创建构建目录
mkdir build && cd build

# 配置构建选项
cmake .. -DUSE_CUDA=ON -DENABLE_SERVER=ON

# 编译
make -j

# 运行测试
ctest
```

### 构建选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `USE_CUDA` | 启用 CUDA GPU 加速 | OFF |
| `ENABLE_SERVER` | 启用 HTTP 服务器 | ON |
| `CMAKE_BUILD_TYPE` | 构建类型（Release/Debug） | Release |

### Docker 部署

fastEmbed 提供 Docker 镜像，方便快速部署和管理。

#### 构建镜像

```bash
# 克隆仓库并进入项目目录
git clone https://github.com/your-repo/fastEmbed.git
cd fastEmbed

# 构建 Docker 镜像
docker build -t fastembed:latest -f Dockerfile .
```

#### 运行容器

```bash
# 基本用法（GPU 加速）
docker run -d --gpus all -p 8080:8080 \
  -v /path/to/models:/app/models \
  fastembed:latest \
  ./server --model_path /app/models/bge-m3 --serve_model_name bge-m3

# CPU 模式（无需 GPU）
docker run -d -p 8080:8080 \
  -v /path/to/models:/app/models \
  fastembed:latest \
  ./server --model_path /app/models/bge-m3 --serve_model_name bge-m3

# 自定义端口
docker run -d --gpus all -p 5001:5001 \
  -v /path/to/models:/app/models \
  fastembed:latest \
  ./server --model_path /app/models/qwen3 --serve_model_name qwen3 --port 5001
```

#### 使用 Docker Compose

项目根目录提供了 `docker-compose.yml` 配置文件：

```yaml
version: '3.8'
services:
  fastembed:
    image: fastembed:latest
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      ./server
      --model_path /app/models/bge-m3
      --serve_model_name bge-m3
```

启动服务：

```bash
docker-compose up -d
```

查看日志：

```bash
docker-compose logs -f
```

停止服务：

```bash
docker-compose down
```

## 常见问题

### Q: 是否需要将模型转换为特定格式？

A: 不需要。fastEmbed 直接读取 HuggingFace 原始模型文件（safetensors 或 bin 格式），无需任何转换步骤。

### Q: 是否必须使用 GPU？

A: 不是必须的。fastEmbed 支持 CPU 和 GPU 两种模式。使用 `-DUSE_CUDA=OFF` 编译即可启用 CPU 模式。

### Q: 如何确保数据隐私？

A: fastEmbed 支持完全离线运行，所有计算在本地完成，数据无需上传到任何外部服务。您可以将服务部署在本地服务器或边缘设备上，确保数据不出本地网络。

### Q: 如何选择合适的模型？

A: 
- 如果主要处理中文短文本，推荐 BGE 系列
- 如果需要处理中英双语或长文档，推荐 Qwen3-Embedding
- 如果资源受限，可选择 BGE-small 或 BGE-base
- 如果需要最高精度，可选择 BGE-large 或 Qwen3-Embedding

### Q: 如何优化内存占用？

A: 可以通过以下方式优化：
1. 选择较小的模型（如 BGE-small）
2. 使用 CPU 模式（无需加载 CUDA 相关库）
3. 缩短输入文本长度
4. 启用模型量化（未来版本支持）

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 LICENSE 文件。

## 致谢

本项目开发过程中使用了以下开源工具和库：
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 模型参考实现
- [Crow](https://github.com/CrowCpp/Crow) - HTTP 服务器框架
- [nlohmann/json](https://github.com/nlohmann/json) - JSON 处理库
- [sentencepiece](https://github.com/google/sentencepiece) - 分词库
- [Asio](https://think-async.com/Asio/) - 异步网络库

感谢所有贡献者和社区成员的支持！
