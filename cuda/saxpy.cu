#include "utils/cuda_utils.cuh"

#include <cstdio>
#include <vector>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    int n = 1 << 20;
    float a = 2.0f;
    size_t bytes = n * sizeof(float);

    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = float(i);
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    saxpy<<<blocks, threads>>>(n, a, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    int wrong = 0;
    for (int i = 0; i < n; i++) {
        if (y[i] != a * 1.0f + float(i)) wrong++;
    }
    printf("n = %d, blocks = %d, threads per block = %d, wrong = %d\n", n, blocks, threads, wrong);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return wrong != 0;
}
