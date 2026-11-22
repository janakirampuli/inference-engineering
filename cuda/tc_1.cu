#include "utils/cuda_utils.cuh"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>

#define WARPS 2   // warps per block, stacked along M: a block computes a 32 x 8 piece of C

// X[row][col] of a row-major matrix, or 0 outside it
__device__ __forceinline__ __nv_bfloat16 load_bf16(const __nv_bfloat16* X, int rows, int cols, int row, int col){
    return (row < rows && col < cols) ? X[row * cols + col] : __float2bfloat16(0.0f);
}

// two BF16 in one 32-bit register, the first in the low 16 bits, as mma expects
__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi){
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}

__global__ void tc_1(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int K, int N){
    // each warp computes one 16 x 8 tile of C
    int tile_row = (blockIdx.y * WARPS + threadIdx.y) * 16;
    int tile_col = blockIdx.x * 8;

    // the fragment layout of mma.m16n8k16 splits the 32 lanes into 8 groups of 4
    int lane = threadIdx.x;
    int group = lane / 4;   // 0..7
    int t = lane % 4;       // 0..3

    // this lane's 4 numbers of the 16 x 8 tile: rows group and group + 8, columns 2t and 2t + 1
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for(int k0 = 0; k0 < K; k0 += 16){
        // A fragment, 16 x 16: rows group and group + 8, columns 2t, 2t + 1 and 2t + 8, 2t + 9 of this step
        int r0 = tile_row + group, r1 = r0 + 8;
        int ka = k0 + 2 * t;
        uint32_t a[4];
        a[0] = pack_bf16(load_bf16(A, M, K, r0, ka),     load_bf16(A, M, K, r0, ka + 1));
        a[1] = pack_bf16(load_bf16(A, M, K, r1, ka),     load_bf16(A, M, K, r1, ka + 1));
        a[2] = pack_bf16(load_bf16(A, M, K, r0, ka + 8), load_bf16(A, M, K, r0, ka + 9));
        a[3] = pack_bf16(load_bf16(A, M, K, r1, ka + 8), load_bf16(A, M, K, r1, ka + 9));

        // B fragment, 16 x 8: column group, rows 2t, 2t + 1 and 2t + 8, 2t + 9 of this step
        int col = tile_col + group;
        uint32_t b[2];
        b[0] = pack_bf16(load_bf16(B, K, N, ka, col),     load_bf16(B, K, N, ka + 1, col));
        b[1] = pack_bf16(load_bf16(B, K, N, ka + 8, col), load_bf16(B, K, N, ka + 9, col));

        // the whole warp: c (16 x 8, FP32) += a (16 x 16, BF16) * b (16 x 8, BF16)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }

    int row = tile_row + group;
    int col = tile_col + 2 * t;
    if(row < M && col < N)         C[row * N + col] = c[0];
    if(row < M && col + 1 < N)     C[row * N + col + 1] = c[1];
    if(row + 8 < M && col < N)     C[(row + 8) * N + col] = c[2];
    if(row + 8 < M && col + 1 < N) C[(row + 8) * N + col + 1] = c[3];
}

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

    // 32 lanes x WARPS warps per block, one 16 x 8 tile of C per warp
    dim3 blockDim(32, WARPS, 1);
    dim3 gridDim(cuda_utils::ceil_div(N, 8), cuda_utils::ceil_div(M, 16 * WARPS), 1);

    // warmup: run for at least 0.5 s so the GPU clock reaches its maximum
    auto warmup_start = std::chrono::steady_clock::now();
    do {
        tc_1<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    } while(std::chrono::steady_clock::now() - warmup_start < std::chrono::milliseconds(500));
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));

    for(int iter = 0; iter < num_iterations; iter++) {
        tc_1<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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

$ nvcc -O3 -arch=sm_100 cuda/tc_1.cu -o tc_1 && ./tc_1      (B200, CUDA 13.5)
average time: 10.286818 ms
achieved compute: 13.360687 TFLOPS
theoretical max compute: 2250.00 TFLOPS (dense BF16 tensor cores)
percent of peak: 0.6%
spot check: 0 of 100 wrong, max relative error 1.21e-05

*/
