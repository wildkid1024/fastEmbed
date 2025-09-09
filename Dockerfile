# 使用基础 CUDA 镜像
FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装必要的工具和依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 nlohmann/json-3-dev 库
RUN apt-get update && apt-get install -y \
    nlohmann-json3-dev \
    libsentencepiece-dev \
    libasio-dev      \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace/Embedding_app

# 复制项目代码
COPY . .

# 编译项目
RUN mkdir -p build && cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON  && cmake --build build

# 定义默认命令
# CMD ["./build/embedding_sever", "--model_path", "/workspace/Embedding_app/models/bge-m3.onnx", ]