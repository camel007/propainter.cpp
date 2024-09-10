# 项目根目录的 CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(propaint CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 设置
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置 CUDA 架构

# 启用 CUDA 语言
enable_language(CUDA)

# 添加 CUDA 头文件路径
include_directories(/usr/local/cuda-11.0/targets/x86_64-linux/include)

# 添加 CUDA 库路径
link_directories(/usr/local/cuda-11.0/targets/x86_64-linux/lib)

# 查找 CUDA 包
find_package(CUDAToolkit REQUIRED)

# 查找 cuDNN
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64)

# 查找 cuBLAS (通常包含在 CUDA Toolkit 中)
find_library(CUBLAS_LIBRARY cublas
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64)

message("toolkit: ${${CUDAToolkit_LIBRARY_DIR}}")

# 添加 include 目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 收集源文件
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp" PROPERTIES LANGUAGE CUDA)
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

# 创建静态库
add_library(${PROJECT_NAME}_static STATIC ${CPP_SOURCES} ${CUDA_SOURCES})
set_target_properties(${PROJECT_NAME}_static PROPERTIES 
    OUTPUT_NAME ${PROJECT_NAME}
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON)


# 创建共享库
add_library(${PROJECT_NAME}_shared SHARED ${CPP_SOURCES} ${CUDA_SOURCES})
set_target_properties(${PROJECT_NAME}_shared PROPERTIES 
    OUTPUT_NAME ${PROJECT_NAME}
    CUDA_SEPARABLE_COMPILATION ON)

# 链接 CUDA 运行时库
target_link_libraries(${PROJECT_NAME}_static PRIVATE 
    CUDA::cudart
    ${CUDNN_LIBRARY}
    ${CUBLAS_LIBRARY} 
     z)

target_link_libraries(${PROJECT_NAME}_shared PRIVATE 
    CUDA::cudart
    ${CUDNN_LIBRARY}
    ${CUBLAS_LIBRARY} 
     z)

# 添加 examples 子目录
add_subdirectory(examples)

# 添加测试子目录
add_subdirectory(test)

set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE PATH "Default install path" FORCE)

# 安装规则
install(TARGETS ${PROJECT_NAME}_static ${PROJECT_NAME}_shared
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)
