# 使用基础 CUDA 镜像
FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive

# 更新系统并安装必要的依赖，包含 SentencePiece 开发库
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libsentencepiece-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 nlohmann/json-3-dev 库
RUN apt-get update && apt-get install -y \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace/Embedding_app
