cmake_minimum_required(VERSION 3.18)
project(kernel-practice LANGUAGES CXX CUDA)

# RTX 3060 / 3090 — Ampere
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

add_executable(hello hello.cu)

add_executable(naive_matmul kernels/matmul/naive.cu)
target_link_libraries(naive_matmul PRIVATE CUDA::cudart)

set_target_properties(hello naive_matmul PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)