# fastEmbed 🚀

fastEmbed是一个**高性能词嵌入推理框架**，采用纯C++和CUDA编写，旨在提供最小依赖和跨平台兼容性的词嵌入生成解决方案。

## 核心优势 ⚡
- **纯C++/CUDA实现** 🔧: 无Python依赖，直接对接底层硬件加速
- **低依赖设计** 📦: 仅需基础系统库和CUDA运行时
- **跨平台支持** 🌐: 兼容Linux系统
- **高性能推理** ⚡: 针对GPU优化的计算内核，支持批量处理

## 技术规格 🛠️

### 支持模型 📋
| 模型名称 | 语言 | 特点 |
|----------|------|------|
| bge-small-zh-v1.5 | 中文 | 轻量级，平衡性能与资源消耗 |
| bge-base-zh-v1.5 | 中文 | 中等规模，更高精度 |
| bge-large-zh-v1.5 | 中文 | 大规模，最高精度 |

### 核心参数
| 参数                | 规格                  |
|---------------------|-----------------------|
| 最大序列长度        | 512                   |
| 输出维度            | 512, 768, 1024        |
| 精度支持            | FP32                  |
| 编译要求            | C++17+, CUDA Toolkit 11+ |

## ⚡ 快速开始

### Docker 部署

#### 构建 Docker 镜像
```bash
docker build -t fastembed:v0.1.0 -f Dockerfile .
```

#### 使用 Docker Compose 启动服务
```bash
docker-compose up -d
```

单个 Docker 命令启动：

```bash
# 启动嵌入服务
docker run -d --gpus all -p 5000:8080 \
  -v /path/to/your/model:/app/models \
  fastembed:v0.1.0 \
  ./embedding_server --model_path /app/models/bge-small-zh-v1.5 --serve_model_name bge-small-zh
```

### 源码编译

#### 依赖

-   C++17 或更高版本
-   CUDA Toolkit 11.0 或更高版本
-   CMake 3.15 或更高版本
-   nlohmann/json 库
-   sentencepiece 库

#### 构建

```bash
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DENABLE_SERVER=ON
make -j
```


### OpenAI HTTP Server 使用示例

#### 服务端示例

```bash

# 启动服务
./build/embedding_server --model_path ./bge-small-zh-v1.5 --serve_model_name bge-small-zh

# 查看帮助
./build/embedding_server --help
```

#### 命令行参数

| 参数                | 描述                  | 默认值 |
|---------------------|-----------------------|--------|
| `--model_path`      | 模型文件路径          | 无     |
| `--serve_model_name`| 服务模型名称          | 无     |
| `--host`            | 服务绑定主机          | `0.0.0.0` |
| `--port`            | 服务绑定端口          | `8080`  |
| `--help`            | 显示帮助信息          | 无     |

#### 客户端示例

```bash


# 测试单文本嵌入
curl -X POST http://localhost:5000/v1/embeddings -H "Content-Type: application/json" -d '{"input": "这是一个测试句子", "model": "bge-small-zh"}'

# 测试批量文本嵌入
curl -X POST http://localhost:5000/v1/embeddings -H "Content-Type: application/json" -d '{"input": ["句子1", "句子2", "句子3"], "model": "bge-small-zh"}'
```


## 📄 许可证
本项目采用Apache License 2.0许可证 - 详见LICENSE文件。

## 🙏 致谢
本项目开发过程中使用了Trae编程辅助工具。