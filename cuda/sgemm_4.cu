#include "utils/cuda_utils.cuh"
#include <cstddef>
#include <chrono>
#include <cstdlib>

#define BM 64
#define BN 64
#define TM 16                      // results for each thread: a column of TM elements of C
#define BK (BM / TM)               // step along k: s_A (BM x BK) and s_B (BK x BN) then hold
                                   // BM * BN / TM numbers, one for each thread of the block

__global__ void sgemm_4(const float* A, const float* B, float* C, int M, int K, int N){
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // top-left corner of this block's tile of C
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // what this thread computes: rows c_row .. c_row + TM - 1 of column c_col, inside the C tile
    int c_col = tx;
    int c_row = ty * TM;

    // what this thread copies into s_A (BM x BK): flat index laid out BK per row
    int tid = ty * BN + tx;
    int a_row = tid / BK;
    int a_col = tid % BK;

    // what this thread copies into s_B (BK x BN): matches the block shape
    int b_row = ty;
    int b_col = tx;

    float results[TM] = {0.0f};

    int num_tiles = cuda_utils::ceil_div(K, BK);
    for(int i = 0; i < num_tiles; i++){
        // step i: tile i of this block's strip of A (moves right) and of B (moves down).
        // out-of-range numbers are stored as 0, so they add nothing to the sums
        if(block_row + a_row < M && i * BK + a_col < K){
            s_A[a_row][a_col] = A[(block_row + a_row) * K + (i * BK + a_col)];
        }
        else{
            s_A[a_row][a_col] = 0.0f;
        }
        if(i * BK + b_row < K && block_col + b_col < N){
            s_B[b_row][b_col] = B[(i * BK + b_row) * N + (block_col + b_col)];
        }
        else{
            s_B[b_row][b_col] = 0.0f;
        }

        // wait until both tiles are complete before reading them
        __syncthreads();

        // k outside, r inside: one s_B load per k, reused for all TM results
        for(int k = 0; k < BK; k++){
            float b = s_B[k][c_col];
            for(int r = 0; r < TM; r++){
                results[r] += s_A[c_row + r][k] * b;
            }
        }

        // wait until every thread is done before the next step overwrites the tiles
        __syncthreads();
    }

    for(int r = 0; r < TM; r++){
        int row = block_row + c_row + r;
        int col = block_col + c_col;
        if(row < M && col < N){
            C[row * N + col] = results[r];
        }
    }
}

int main(){
    const int M = 4096;
    const int K = 4096;
    const int N = 4096;
    const double peak_tflops = cuda_utils::fp32_peak_tflops();

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

    dim3 blockDim(BN, BM / TM, 1);
    dim3 gridDim(cuda_utils::ceil_div(N, BN), cuda_utils::ceil_div(M, BM), 1);

    // warmup: run for at least 0.5 s so the GPU clock reaches its maximum
    auto warmup_start = std::chrono::steady_clock::now();
    do {
        sgemm_4<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    } while(std::chrono::steady_clock::now() - warmup_start < std::chrono::milliseconds(500));
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));

    for(int iter = 0; iter < num_iterations; iter++) {
        sgemm_4<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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
    printf("theoretical max compute: %.2f TFLOPS\n", peak_tflops);
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

$ nvcc -O3 -arch=sm_100 cuda/sgemm_4.cu -o sgemm_4 && ./sgemm_4      (B200, CUDA 13.5)
average time: 6.347925 ms
achieved compute: 21.651006 TFLOPS
theoretical max compute: 74.45 TFLOPS
percent of peak: 29.1%
spot check: 0 of 100 wrong, max relative error 2.72e-06

*/
