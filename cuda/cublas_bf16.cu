#include "utils/cuda_utils.cuh"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_bf16.h>

int main(){
    const int M = 4096;
    const int K = 4096;
    const int N = 4096;
    const double peak_tflops = cuda_utils::B200_BF16_TFLOPS;

    size_t size_A = (size_t)M * K * sizeof(__nv_bfloat16);
    size_t size_B = (size_t)K * N * sizeof(__nv_bfloat16);
    size_t size_C = (size_t)M * N * sizeof(float);

    // random numbers in [0, 1], rounded to BF16. h_Af and h_Bf keep the rounded values as floats
    // for the spot check, so the check measures only the error of the sums
    float *h_Af = (float*)malloc((size_t)M * K * sizeof(float));
    float *h_Bf = (float*)malloc((size_t)K * N * sizeof(float));
    __nv_bfloat16 *h_A = (__nv_bfloat16*)malloc(size_A);
    __nv_bfloat16 *h_B = (__nv_bfloat16*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    cuda_utils::init_matrix(h_Af, M, K);
    cuda_utils::init_matrix(h_Bf, K, N);
    for (size_t i = 0; i < (size_t)M * K; i++) { h_A[i] = __float2bfloat16(h_Af[i]); h_Af[i] = __bfloat162float(h_A[i]); }
    for (size_t i = 0; i < (size_t)K * N; i++) { h_B[i] = __float2bfloat16(h_Bf[i]); h_Bf[i] = __bfloat162float(h_B[i]); }

    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    // BF16 inputs, FP32 output, FP32 sums.
    // row-major C = A * B is column-major C^T = B^T * A^T, so B goes first
    auto gemm = [&](){
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B, CUDA_R_16BF, N,   // Matrix B
                                  d_A, CUDA_R_16BF, K,   // Matrix A
                                  &beta,
                                  d_C, CUDA_R_32F, N,    // Matrix C
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };

    // warmup: run for at least 0.5 s so the GPU clock reaches its maximum
    auto warmup_start = std::chrono::steady_clock::now();
    do {
        gemm();
        CUDA_CHECK(cudaDeviceSynchronize());
    } while(std::chrono::steady_clock::now() - warmup_start < std::chrono::milliseconds(500));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < num_iterations; iter++) {
        gemm();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / num_iterations;

    double flops_total = 2.0 * M * N * K;
    double achieved_tflops = (flops_total / (avg_ms * 1e-3)) / 1e12;

    printf("average time: %f ms\n", avg_ms);
    printf("achieved compute: %f TFLOPS\n", achieved_tflops);
    printf("theoretical max compute: %.2f TFLOPS (dense BF16 tensor cores)\n", peak_tflops);
    printf("percent of peak: %.1f%%\n", 100.0 * achieved_tflops / peak_tflops);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    int num_checks = 100;
    int num_wrong = 0;
    double max_rel_err = 0.0;
    for (int t = 0; t < num_checks; t++) {
        int i = rand() % M;
        int j = rand() % N;
        double ref = 0.0;
        for (int k = 0; k < K; k++) {
            ref += (double)h_Af[(size_t)i * K + k] * h_Bf[(size_t)k * N + j];
        }
        double rel_err = fabs(h_C[(size_t)i * N + j] - ref) / fabs(ref);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err > 1e-3) num_wrong++;
    }
    printf("spot check: %d of %d wrong, max relative error %.2e\n", num_wrong, num_checks, max_rel_err);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_Af);
    free(h_Bf);
    free(h_A);
    free(h_B);
    free(h_C);

    return num_wrong != 0;
}
/*

$ nvcc -O3 -arch=sm_100 cuda/cublas_bf16.cu -o cublas_bf16 -lcublas && ./cublas_bf16      (B200, CUDA 13.5)
average time: 0.099389 ms
achieved compute: 1382.841454 TFLOPS
theoretical max compute: 2250.00 TFLOPS (dense BF16 tensor cores)
percent of peak: 61.5%
spot check: 0 of 100 wrong, max relative error 1.21e-05

cuBLAS picks nvjet_sm100_tss_128x256_64x6_2x2f_2cta_h_bz_NNT (a second run took 0.101510 ms).

FP32 inputs on the tensor cores, for comparison with the FP32 kernels (cuBLAS.cu, B200):
$ ./cublas                                       average time: 2.156818 ms   max relative error 2.72e-06
$ CUBLAS_EMULATE_SINGLE_PRECISION=1 ./cublas     average time: 0.839314 ms   max relative error 6.94e-07
(BF16x9 emulation: cutlass3x_sm100_tensorop_s256x128x16_9xbf16gemm_..., plus scan and patch kernels for inf/NaN;
 a second run took 0.899134 ms)

*/
