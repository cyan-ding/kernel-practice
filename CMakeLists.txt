cmake_minimum_required(VERSION 3.18)
project(kernel-practice LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RTX 3060 / 3090 — Ampere
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

add_library(matmul_common STATIC
    kernels/matmul/common.cpp
    kernels/matmul/cuda_host.cu
)
target_link_libraries(matmul_common PUBLIC CUDA::cudart)
target_include_directories(matmul_common PUBLIC kernels/matmul)
set_target_properties(matmul_common PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

function(add_matmul_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE matmul_common)
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endfunction()

add_matmul_executable(naive_matmul kernels/matmul/naive.cu)
add_matmul_executable(tiled_matmul kernels/matmul/tiled.cu)
