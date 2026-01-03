cmake_minimum_required(VERSION 3.10)
project(Embedding_APP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加选项，默认不使用 CUDA
option(USE_CUDA "Use CUDA for acceleration" OFF)

# 添加选项，默认使用 cuBLAS 进行矩阵计算
option(USE_CUBLAS "Use cuBLAS for matrix operations" OFF)

# 添加选项，默认不启用测试
option(ENABLE_TESTS "Enable unit tests" OFF)

# 添加选项，默认不启用嵌入服务
option(ENABLE_SERVER "Enable embedding service" OFF)

# 查找 sentencepiece 库 - 简化版本，只链接必要的库
find_library(SENTENCEPIECE_LIBRARY NAMES sentencepiece libsentencepiece)
find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h)

if(NOT SENTENCEPIECE_LIBRARY OR NOT SENTENCEPIECE_INCLUDE_DIR)
    message(FATAL_ERROR "sentencepiece 库未找到，请安装 libsentencepiece-dev")
endif()

# 只使用 sentencepiece 库，不需要 sentencepiece_train
set(SENTENCEPIECE_LIBRARIES ${SENTENCEPIECE_LIBRARY})


if(USE_CUDA)
    cmake_policy(SET CMP0146 OLD) 

    # Enable CUDA support
    enable_language(CUDA)

    # Find CUDA
    find_package(CUDA REQUIRED)

    # Set policy to suppress CMP0146 warning
    # cmake_policy(SET CMP0146 NEW)

    # 设置 CUDA 架构
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)

    # Find CUDA Toolkit (modern CMake)
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA architecture and compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75")

    # 根据是否使用 cuBLAS 分类源文件
    if(USE_CUBLAS)
        message(STATUS "Building with cuBLAS support")
        # 使用 cuBLAS 的 CUDA 源文件
        file(GLOB CUDA_BASE_SOURCES "src/ops/cuda/cuda_matrix_ops.cu")
        file(GLOB CUDA_CUBLAS_SOURCES "src/ops/cuda/cuda_attention_cublas_ops.cu")
        set(CUDA_SOURCES ${CUDA_BASE_SOURCES} ${CUDA_CUBLAS_SOURCES})
        # 添加 cuBLAS 库
        set(ADDITIONAL_LIBS ${CUDA_LIBRARIES} CUDA::cublas ${SENTENCEPIECE_LIBRARIES})
    else()
        message(STATUS "Building without cuBLAS support")
        # 不使用 cuBLAS 的 CUDA 源文件
        file(GLOB CUDA_BASE_SOURCES "src/ops/cuda/cuda_matrix_ops.cu")
        file(GLOB CUDA_NON_CUBLAS_SOURCES "src/ops/cuda/cuda_attention_ops.cu")
        set(CUDA_SOURCES ${CUDA_BASE_SOURCES} ${CUDA_NON_CUBLAS_SOURCES})
        # 不添加 cuBLAS 库
        set(ADDITIONAL_LIBS ${CUDA_LIBRARIES} ${SENTENCEPIECE_LIBRARIES})
    endif()
    
    # C++ 源文件
    file(GLOB_RECURSE CPP_SOURCES "src/*.cpp" "src/ops/cuda/*.cpp")

    # Include directories
    include_directories(${CUDA_INCLUDE_DIRS} include include/ops ${SENTENCEPIECE_INCLUDE_DIR})
else()
    # 仅查找非 CUDA 相关的 C++ 源文件
    file(GLOB_RECURSE CPP_SOURCES "src/*.cpp" "src/ops/cpu/*.cpp")
    # list(FILTER CPP_SOURCES EXCLUDE REGEX "src/ops/cpu/.*\\.cpp")

    # Include directories
    include_directories(include include/ops ${SENTENCEPIECE_INCLUDE_DIR})

    # 添加 sentencepiece 库
    set(ADDITIONAL_LIBS ${SENTENCEPIECE_LIBRARIES})
endif()

# Executable
if(USE_CUDA)
    add_executable(app app.cpp ${CPP_SOURCES} ${CUDA_SOURCES})
    add_executable(cli cli_app.cpp ${CPP_SOURCES} ${CUDA_SOURCES})
else()
    add_executable(app app.cpp ${CPP_SOURCES})
    add_executable(cli cli_app.cpp ${CPP_SOURCES})
endif()

# 查找并链接 nlohmann_json 库
find_package(nlohmann_json REQUIRED)

# 链接库
target_link_libraries(app 
    ${ADDITIONAL_LIBS}
    nlohmann_json::nlohmann_json
)

target_link_libraries(cli 
    ${ADDITIONAL_LIBS}
    nlohmann_json::nlohmann_json
)


# 添加 nlohmann_json 库支持（如果需要）
find_package(nlohmann_json REQUIRED)

# 添加 safetensors 库支持（如果需要）
# find_package(safetensors REQUIRED)

# 链接库
target_link_libraries(app 
    ${ADDITIONAL_LIBS}
    nlohmann_json::nlohmann_json
    # safetensors::safetensors
)

target_link_libraries(cli 
    ${ADDITIONAL_LIBS}
    nlohmann_json::nlohmann_json
    # safetensors::safetensors
)


if(ENABLE_TESTS)
    message(STATUS "Building with tests")

enable_testing()

# Add Google Test configuration
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# // 添加CPU矩阵操作动态库
add_library(cpu_matrix_ops SHARED
    src/ops/cpu/cpu_matrix_ops.cpp
)

target_include_directories(cpu_matrix_ops PUBLIC include/ops/cpu)

if(USE_CUDA)
    # // 添加CUDA矩阵操作动态库
    add_library(cuda_matrix_ops SHARED
        src/ops/cuda/cuda_matrix_ops.cu
    )
    target_include_directories(cuda_matrix_ops PUBLIC include/ops/cuda)
    set_target_properties(cuda_matrix_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    # 根据是否使用 cuBLAS 添加相应的注意力操作源文件
    if(USE_CUBLAS)
        target_sources(cuda_matrix_ops PRIVATE
            src/ops/cuda/cuda_attention_cublas_ops.cu
        )
    else()
        target_sources(cuda_matrix_ops PRIVATE
            src/ops/cuda/cuda_attention_ops.cu
        )
    endif()
    
    # Add attention ops sources to CPU library
    target_sources(cpu_matrix_ops PRIVATE
        src/ops/cpu/cpu_attention_ops.cpp
    )
endif()

# // 修复测试目标链接
add_executable(matrix_ops_test tests/matrix_ops_test.cpp)
if(USE_CUDA)
    target_link_libraries(matrix_ops_test
        GTest::GTest GTest::Main
        cpu_matrix_ops
        cuda_matrix_ops
    )
else()
    target_link_libraries(matrix_ops_test
        GTest::GTest GTest::Main
        cpu_matrix_ops
    )
endif()
add_test(NAME matrix_ops_test COMMAND matrix_ops_test)

# 添加注意力操作测试
add_executable(attention_ops_test tests/attention_ops_test.cpp ${CPP_SOURCES} ${CUDA_SOURCES})
if(USE_CUDA)
    target_link_libraries(attention_ops_test
        GTest::GTest GTest::Main
        cpu_matrix_ops
        cuda_matrix_ops
        CUDA::cublas
    )
else()
    target_link_libraries(attention_ops_test
        GTest::GTest
        GTest::Main
        cpu_matrix_ops
        cuda_matrix_ops
    )
endif()
add_test(NAME attention_ops_test COMMAND attention_ops_test)

else()
    message(STATUS "Building without tests")
endif()

if (ENABLE_SERVER)
    message(STATUS "Building with embedding server")

    # 设置 Crow 库路径
    set(CROW_DIR "${CMAKE_SOURCE_DIR}/third_party/crow")

    # 确保 Crow 目录存在
    if(NOT EXISTS ${CROW_DIR}/CMakeLists.txt)
        message(FATAL_ERROR "Crow directory not found at ${CROW_DIR}. Please ensure Crow is cloned in the third_party directory.")
    endif()

# 添加Crow HTTP库
add_subdirectory(${CROW_DIR})

# 添加嵌入服务可执行文件
add_executable(server http_server.cpp ${CPP_SOURCES} ${CUDA_SOURCES})

target_link_libraries(server
    PRIVATE
       ${ADDITIONAL_LIBS}
        Crow::Crow
)

endif()