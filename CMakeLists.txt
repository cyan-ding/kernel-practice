cmake_minimum_required(VERSION 3.18)
project(kernel-practice LANGUAGES CXX CUDA)

# RTX 3060 / 3090 — Ampere
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(hello hello.cu)
set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)