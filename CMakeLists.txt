cmake_minimum_required(VERSION 3.10)
project(Embedding_APP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 添加选项，默认不使用 CUDA
option(USE_CUDA "Use CUDA for acceleration" OFF)


if(USE_CUDA)
    # Enable CUDA support
    enable_language(CUDA)

    # Find CUDA
    find_package(CUDA REQUIRED)

    # Set policy to suppress CMP0146 warning
    cmake_policy(SET CMP0146 NEW)

    # Find CUDA Toolkit (modern CMake)
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA architecture and compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75")

    # Source files
    file(GLOB_RECURSE CUDA_SOURCES "src/ops/cuda/*.cu")
    file(GLOB_RECURSE CPP_SOURCES "src/*.cpp" "src/ops/cuda/*.cpp")

    # Include directories
    include_directories(${CUDA_INCLUDE_DIRS} include include/ops)

    # Link libraries
    set(ADDITIONAL_LIBS ${CUDA_LIBRARIES} CUDA::cublas)
else()
    # 仅查找非 CUDA 相关的 C++ 源文件
    file(GLOB_RECURSE CPP_SOURCES "src/*.cpp" "src/ops/cpu/*.cpp")
    # list(FILTER CPP_SOURCES EXCLUDE REGEX "src/ops/cpu/.*\\.cpp")

    # Include directories
    include_directories(include include/ops)

    # 不添加额外库
    set(ADDITIONAL_LIBS "")
endif()

# Executable
if(USE_CUDA)
    add_executable(Embedding_APP app.cpp ${CPP_SOURCES} ${CUDA_SOURCES})
    add_executable(Cli_APP cli_app.cpp ${CPP_SOURCES} ${CUDA_SOURCES})
else()
    add_executable(Embedding_APP app.cpp ${CPP_SOURCES})
    add_executable(Cli_APP cli_app.cpp ${CPP_SOURCES})
endif()


target_link_libraries(Embedding_APP 
    ${ADDITIONAL_LIBS}
)

target_link_libraries(Cli_APP 
    ${ADDITIONAL_LIBS}
)

if (USE_CUDA)
    message(STATUS "Building with CUDA support")
    target_link_libraries(Embedding_APP 
        ${ADDITIONAL_LIBS}
         CUDA::cublas
    )
    target_link_libraries(Cli_APP 
        ${ADDITIONAL_LIBS}
         CUDA::cublas
    )
else()
    message(STATUS "Building without CUDA support")
    target_link_libraries(Embedding_APP 
        ${ADDITIONAL_LIBS}
    )
    target_link_libraries(Cli_APP 
        ${ADDITIONAL_LIBS}
    )
endif()

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
add_executable(attention_ops_test tests/attention_ops_test.cpp src/ops/cuda/cuda_attention_cublas_ops.cu)
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
        $<$<BOOL:${USE_CUDA}>:cuda_matrix_ops>
    )
endif()
add_test(NAME attention_ops_test COMMAND attention_ops_test)