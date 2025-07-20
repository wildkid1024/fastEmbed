cmake_minimum_required(VERSION 3.10)
project(Embedding_APP)

# 添加选项，默认不使用 CUDA
option(USE_CUDA "Use CUDA for acceleration" OFF)

# find_package(SentencePiece REQUIRED)
find_path(SentencePiece_INCLUDE_DIRS NAMES sentencepiece_processor.h
          PATHS /usr/include /usr/local/include /usr/local/Cellar/sentencepiece/*/include)
find_library(SentencePiece_LIBRARIES NAMES sentencepiece
             PATHS /usr/lib /usr/local/lib /usr/local/Cellar/sentencepiece/*/lib)


if(USE_CUDA)
    # Enable CUDA support
    enable_language(CUDA)

    # Find CUDA
    find_package(CUDA REQUIRED)

    # Set CUDA architecture
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch=sm_75)

    # Source files
    file(GLOB_RECURSE CUDA_SOURCES "src/ops/cuda/*.cu")
    file(GLOB_RECURSE CPP_SOURCES "src/*.cpp" "src/ops/cuda/*.cpp")

    # Include directories
    include_directories(${CUDA_INCLUDE_DIRS} include include/ops)

    # Link libraries
    set(ADDITIONAL_LIBS ${CUDA_LIBRARIES})
else()
    # 仅查找非 CUDA 相关的 C++ 源文件
    file(GLOB_RECURSE CPP_SOURCES "src/*.cpp" "src/ops/cpu/*.cpp")
    # list(FILTER CPP_SOURCES EXCLUDE REGEX "src/ops/cpu/.*\\.cpp")

    # Include directories
    include_directories(include include/ops)

    # 不添加额外库
    set(ADDITIONAL_LIBS "")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

include_directories(${SentencePiece_INCLUDE_DIRS})


# Executable
add_executable(Embedding_APP app.cpp ${CPP_SOURCES})


target_link_libraries(Embedding_APP 
    ${ADDITIONAL_LIBS}
    ${SentencePiece_LIBRARIES}
)
