#pragma once

// Launch a matmul kernel that has already been copied to the device.
using MatmulLaunchFn = void (*)(const float *A, const float *B, float *C, int M,
                                int K, int N);

// Shared CLI entry point: <M> <K> <N> <A.bin> <B.bin> <C.bin>
int run_matmul_cli(int argc, char **argv, MatmulLaunchFn launch);

// Called after the kernel launch, before copying results back to the host.
void matmul_sync_after_launch();
