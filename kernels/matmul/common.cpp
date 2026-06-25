#include "common.hpp"
#include "cuda_host.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct MatmulArgs {
    int M;
    int K;
    int N;
    const char *a_path;
    const char *b_path;
    const char *c_path;
};

void print_usage(const char *program) {
    fprintf(stderr,
            "Usage: %s <M> <K> <N> <A.bin> <B.bin> <C.bin>\n"
            "\n"
            "  A is MxK, B is KxN, C is MxN (row-major float32).\n",
            program);
}

bool parse_args(int argc, char **argv, MatmulArgs *args) {
    if (argc != 7) {
        print_usage(argv[0]);
        return false;
    }

    args->M = atoi(argv[1]);
    args->K = atoi(argv[2]);
    args->N = atoi(argv[3]);
    args->a_path = argv[4];
    args->b_path = argv[5];
    args->c_path = argv[6];

    if (args->M <= 0 || args->K <= 0 || args->N <= 0) {
        fprintf(stderr, "M, K, and N must be positive.\n");
        return false;
    }

    return true;
}

std::vector<float> read_matrix(const char *path, size_t count) {
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

void write_matrix(const char *path, const float *data, size_t count) {
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

} // namespace

int run_matmul_cli(int argc, char **argv, MatmulLaunchFn launch) {
    MatmulArgs args{};
    if (!parse_args(argc, argv, &args)) {
        return EXIT_FAILURE;
    }

    const size_t a_count = static_cast<size_t>(args.M) * args.K;
    const size_t b_count = static_cast<size_t>(args.K) * args.N;
    const size_t c_count = static_cast<size_t>(args.M) * args.N;

    const std::vector<float> h_A = read_matrix(args.a_path, a_count);
    const std::vector<float> h_B = read_matrix(args.b_path, b_count);

    MatmulDeviceBuffers buffers =
        matmul_alloc_and_upload(h_A, h_B, args.M, args.K, args.N);

    launch(buffers.A, buffers.B, buffers.C, args.M, args.K, args.N);
    matmul_sync_after_launch();
    matmul_download_and_free(&buffers);

    write_matrix(args.c_path, buffers.host_C.data(), c_count);

    return EXIT_SUCCESS;
}
