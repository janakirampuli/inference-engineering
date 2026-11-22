#pragma once

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                   \
                    cudaGetErrorString(err_), __FILE__, __LINE__);         \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st_ = (call);                                       \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error: %s at %s:%d\n",                 \
                    cublasGetStatusString(st_), __FILE__, __LINE__);       \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

namespace cuda_utils {

__host__ __device__ constexpr int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

inline void init_matrix(float* m, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++) {
        m[i] = rand() / (float)RAND_MAX;
    }
}

// FP32 peak of GPU 0 in TFLOPS: SMs x max SM clock x FP32 cores per SM x 2 (an FFMA is 2 FLOPs).
// FP32 cores per SM: 64 on V100 and A100 (7.x, 8.0), 128 on later GPUs (A10, H100, B200, ...)
inline double fp32_peak_tflops()
{
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int clock_khz;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    int cores = (p.major == 7 || (p.major == 8 && p.minor == 0)) ? 64 : 128;
    return (double)p.multiProcessorCount * clock_khz * 1e3 * cores * 2 / 1e12;
}

// dense BF16 tensor core speed of the B200 (HGX B200, 1000 W) from NVIDIA's spec sheet, in TFLOPS.
// 4.5 PFLOPS is the figure with 2:4 sparsity. The runtime has no attribute to compute it from
constexpr double B200_BF16_TFLOPS = 2250.0;

}
