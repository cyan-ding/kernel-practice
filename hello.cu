#include <cstdio>

__global__ void hello() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}


int main() {
    hello<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
