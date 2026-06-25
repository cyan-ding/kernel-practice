#include "common.hpp"

#include <cuda_runtime.h>

// TODO: implement a tiled matmul kernel using __shared__ memory.
__global__ void matmul_tiled(const float *A, const float *B, float *C, int M,
                             int K, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void launch_tiled(const float *A, const float *B, float *C, int M, int K, int N) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_tiled<<<grid, block>>>(A, B, C, M, K, N);
}

int main(int argc, char **argv) {
    return run_matmul_cli(argc, argv, launch_tiled);
}
