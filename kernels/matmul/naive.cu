#include <cstdio>
#include <cstdlib>
#include <vector>

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

// One thread computes one output element: C[row, col].
__global__ void matmul_naive(const float *A, const float *B, float *C, int M,
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

static std::vector<float> read_matrix(const char *path, size_t count) {
    FILE *file = fopen(path, "rb");
    if (file == nullptr) {
        fprintf(stderr, "Failed to open input file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    std::vector<float> data(count);
    const size_t read_count = fread(data.data(), sizeof(float), count, file);
    fclose(file);

    if (read_count != count) {
        fprintf(stderr, "Expected %zu floats in %s, got %zu\n", count, path,
                read_count);
        exit(EXIT_FAILURE);
    }

    return data;
}

static void write_matrix(const char *path, const float *data, size_t count) {
    FILE *file = fopen(path, "wb");
    if (file == nullptr) {
        fprintf(stderr, "Failed to open output file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    const size_t written = fwrite(data, sizeof(float), count, file);
    fclose(file);

    if (written != count) {
        fprintf(stderr, "Failed to write %zu floats to %s\n", count, path);
        exit(EXIT_FAILURE);
    }
}

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s <M> <K> <N> <A.bin> <B.bin> <C.bin>\n"
            "\n"
            "  A is MxK, B is KxN, C is MxN (row-major float32).\n",
            program);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const int M = atoi(argv[1]);
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);
    const char *a_path = argv[4];
    const char *b_path = argv[5];
    const char *c_path = argv[6];

    if (M <= 0 || K <= 0 || N <= 0) {
        fprintf(stderr, "M, K, and N must be positive.\n");
        return EXIT_FAILURE;
    }

    const std::vector<float> h_A = read_matrix(a_path, static_cast<size_t>(M) * K);
    const std::vector<float> h_B = read_matrix(b_path, static_cast<size_t>(K) * N);
    std::vector<float> h_C(static_cast<size_t>(M) * N, 0.0f);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(M) * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(K) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, static_cast<size_t>(M) * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), static_cast<size_t>(M) * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), static_cast<size_t>(K) * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, static_cast<size_t>(M) * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    write_matrix(c_path, h_C.data(), static_cast<size_t>(M) * N);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
