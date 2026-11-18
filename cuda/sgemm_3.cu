#include "utils/cuda_utils.cuh"
#include <cstddef>
#include <cstdlib>

#define BLOCK_SIZE 32

__global__ void sgemm_3(const float* A, const float* B, float* C, int M, int K, int N){
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = ty + BLOCK_SIZE * by;
    int col = tx + BLOCK_SIZE * bx;

    float tmp = 0.0f;
    int num_tiles = cuda_utils::ceil_div(K, BLOCK_SIZE);

    for(int i = 0; i < num_tiles; i++){
        if(row < M && i * BLOCK_SIZE + tx < K){
            s_A[ty][tx] = A[tx + row * K + i * BLOCK_SIZE];
        }
        else{
            s_A[ty][tx] = 0.0f;
        }
        if(col < N && i * BLOCK_SIZE + ty < K){
            s_B[ty][tx] = B[ty * N + col + i * BLOCK_SIZE * N];
        }
        else {
            s_B[ty][tx] = 0.0f;
        }
        __syncthreads();
        for(int j = 0; j < BLOCK_SIZE; j++){
            tmp += s_A[ty][j] * s_B[j][tx];
        }
        __syncthreads();
    }
    if(row < M && col < N){
        C[row * N + col] = tmp;
    }
}

int main(){
    const int M = 4096;
    const int K = 4096;
    const int N = 4096;
    const double a100_peak_tflops = 19.49;

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    cuda_utils::init_matrix(h_A, M, K);
    cuda_utils::init_matrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(cuda_utils::ceil_div(N, BLOCK_SIZE), cuda_utils::ceil_div(M, BLOCK_SIZE), 1);

    // warmup
    for(int i = 0; i < 5; i++){
        sgemm_3<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));

    for(int iter = 0; iter < num_iterations; iter++) {
        sgemm_3<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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
    printf("theoretical max compute: %.2f TFLOPS\n", a100_peak_tflops);
    printf("percent of peak: %.1f%%\n", 100.0 * achieved_tflops / a100_peak_tflops);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    int num_checks = 100;
    int num_wrong = 0;
    double max_rel_err = 0.0;
    for (int t = 0; t < num_checks; t++) {
        int i = rand() % M;
        int j = rand() % N;
        double ref = 0.0;
        for (int k = 0; k < K; k++) {
            ref += (double)h_A[(size_t)i * K + k] * h_B[(size_t)k * N + j];
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
    free(h_A);
    free(h_B);
    free(h_C);

    return num_wrong != 0;
}
/*

$ nvcc -O3 -arch=sm_80 cuda/sgemm_3.cu -o sgemm_3 && ./sgemm_3
average time: 25.819290 ms
achieved compute: 5.323111 TFLOPS
theoretical max compute: 19.49 TFLOPS
percent of peak: 27.3%
spot check: 0 of 100 wrong, max relative error 2.72e-06

*/
