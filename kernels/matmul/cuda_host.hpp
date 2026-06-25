#pragma once

#include <vector>

struct MatmulDeviceBuffers {
    float *A;
    float *B;
    float *C;
    std::vector<float> host_C;
};

MatmulDeviceBuffers matmul_alloc_and_upload(const std::vector<float> &h_A,
                                            const std::vector<float> &h_B,
                                            int M, int K, int N);

void matmul_download_and_free(MatmulDeviceBuffers *buffers);
