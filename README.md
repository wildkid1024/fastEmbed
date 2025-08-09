
# fastEmbed 🚀

fastEmbed是一个**高性能词嵌入推理框架**，采用纯C++和CUDA编写，旨在提供最小依赖和跨平台兼容性的词嵌入生成解决方案。

## 核心优势 ⚡
- **纯C++/CUDA实现** 🔧: 无Python依赖，直接对接底层硬件加速
- **低依赖设计** 📦: 仅需基础系统库和CUDA运行时
- **跨平台支持** 🌐: 兼容Linux、Windows和macOS系统
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

### 依赖

-   C++17 或更高版本
-   CUDA Toolkit 11.0 或更高版本
-   CMake 3.15 或更高版本

### 构建

```bash
mkdir build
cd build
cmake ..
make -j
```

### 使用示例

```cpp
#include "sentence_transformers.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    try {
        // 指定模型路径
        std::string model_path = "/path/to/bge-small-zh-v1.5";

        // 初始化 SentenceTransformer 模型
        SentenceTransformer model(model_path);

        // 中文文本示例
        std::vector<std::string> texts = {
            "This is a test sentence.", "Another example sentence.",
        };

        // 批量编码文本获取嵌入向量
        std::vector<std::vector<float>> embeddings = model.encode_batch(texts);

        // 打印嵌入向量信息
        for (size_t i = 0; i < embeddings.size(); ++i) {
            std::cout << "文本: " << texts[i] << std::endl;
            std::cout << "嵌入向量维度: " << embeddings[i].size() << std::endl;
            std::cout << "嵌入向量前5个值: ";
            for (size_t j = 0; j < 5 && j < embeddings[i].size(); ++j) {
                std::cout << embeddings[i][j] << " ";
            }
            std::cout << "...\n\n";
        }
    } catch (const std::exception& e) {
        // 错误处理
        std::cerr << "发生错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

## 🌍 跨平台支持
- **Linux** 🐧: Ubuntu 18.04+, CentOS 7+
- **Windows** 🪟: Windows 10+, Visual Studio 2019+
- **macOS** 🍎: macOS 10.15+, Xcode 11+

## 许可证 📄
本项目采用Apache License 2.0许可证 - 详见LICENSE文件。

## 致谢 🙏
本项目开发过程中使用了Trae编程辅助工具。