#include "cuda_host.hpp"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

MatmulDeviceBuffers matmul_alloc_and_upload(const std::vector<float> &h_A,
                                            const std::vector<float> &h_B,
                                            int M, int K, int N) {
    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    const size_t c_count = static_cast<size_t>(M) * N;

    MatmulDeviceBuffers buffers{};
    buffers.host_C.assign(c_count, 0.0f);

    CUDA_CHECK(cudaMalloc(&buffers.A, a_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.B, b_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.C, c_count * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(buffers.A, h_A.data(), a_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.B, h_B.data(), b_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    return buffers;
}

void matmul_download_and_free(MatmulDeviceBuffers *buffers) {
    const size_t c_count = buffers->host_C.size();

    CUDA_CHECK(cudaMemcpy(buffers->host_C.data(), buffers->C,
                          c_count * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(buffers->A));
    CUDA_CHECK(cudaFree(buffers->B));
    CUDA_CHECK(cudaFree(buffers->C));

    buffers->A = nullptr;
    buffers->B = nullptr;
    buffers->C = nullptr;
}

void matmul_sync_after_launch() {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
