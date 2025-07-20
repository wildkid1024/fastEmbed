cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(Embedding_APP)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义编译选项
option(USE_CUDA "Compile with CUDA support" OFF)

# 查找 tokenizers 库和 nlohmann/json 库
find_package(tokenizers-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

# 设置头文件包含路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/FlashInfer/include
)

# 设置库文件链接路径
link_directories(
    ${CMAKE_SOURCE_DIR}/thirdparty/FlashInfer/lib
)

# 根据选项收集源文件
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    file(GLOB_RECURSE SOURCES 
        "src/*.cpp" 
        "src/ops/cuda/*.cpp" 
        "src/ops/cuda/*.cu"
    )
    cuda_add_executable(Embedding_APP
        app.cpp
        ${SOURCES}
    )
    target_link_libraries(Embedding_APP
        flashinfer  # 实际使用时替换为正确的库名
        tokenizers-cpp::tokenizers-cpp
        nlohmann_json::nlohmann_json
        ${CUDA_LIBRARIES}
    )
else()
    file(GLOB_RECURSE SOURCES 
        "src/*.cpp" 
        "src/ops/cpu/*.cpp"
    )
    add_executable(Embedding_APP
        app.cpp
        ${SOURCES}
    )
    target_link_libraries(Embedding_APP
        flashinfer  # 实际使用时替换为正确的库名
        tokenizers-cpp::tokenizers-cpp
        nlohmann_json::nlohmann_json
    )
endif()

# 可选：设置编译选项
if(MSVC)
    target_compile_options(Embedding_APP PRIVATE /W4 /WX)
else()
    target_compile_options(Embedding_APP PRIVATE -Wall -Wextra -Werror)
endif()
