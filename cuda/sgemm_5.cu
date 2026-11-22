#include "utils/cuda_utils.cuh"
#include <cstddef>
#include <chrono>
#include <cstdlib>

#define BM 64
#define BN 64                        // the tile of C is square (BM = BN), see BK
#define TM 4
#define TN 4
#define BK (BM / (TM * TN))          // step along k: s_A (BM x BK) and s_B (BK x BN) then hold
                                     // BM * BN / (TM * TN) numbers, one for each thread: 64 / 16 = 4

#define NT ((BM / TM) * (BN / TN))   // threads per block: 16 x 16 = 256
static_assert(BM == BN && BM * BK == NT && BK * BN == NT, "one number of each tile per thread");

__global__ void sgemm_5(const float* A, const float* B, float* C, int M, int K, int N){
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // top-left corner of this block's tile of C
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // what this thread computes: a TM x TN square of the C tile,
    // rows c_row .. c_row + TM - 1 and columns c_col .. c_col + TN - 1
    int c_row = ty * TM;
    int c_col = tx * TN;

    // flat index 0..NT-1 for the copies, so neighboring threads copy neighboring numbers
    int tid = ty * (BN / TN) + tx;

    // what this thread copies: one number of s_A (BM x BK) and one of s_B (BK x BN)
    int a_row = tid / BK;
    int a_col = tid % BK;
    int b_row = tid / BN;
    int b_col = tid % BN;

    float results[TM][TN] = {0.0f};
    float reg_A[TM];  // column k of s_A, this thread's TM rows
    float reg_B[TN];  // row k of s_B, this thread's TN columns

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

        for(int k = 0; k < BK; k++){
            // load TM numbers of s_A and TN numbers of s_B into registers once ...
            for(int m = 0; m < TM; m++){
                reg_A[m] = s_A[c_row + m][k];
            }
            for(int n = 0; n < TN; n++){
                reg_B[n] = s_B[k][c_col + n];
            }
            // ... then use each of them TN or TM times: TM x TN FFMAs from TM + TN loads
            for(int m = 0; m < TM; m++){
                for(int n = 0; n < TN; n++){
                    results[m][n] += reg_A[m] * reg_B[n];
                }
            }
        }

        // wait until every thread is done before the next step overwrites the tiles
        __syncthreads();
    }

    for(int m = 0; m < TM; m++){
        for(int n = 0; n < TN; n++){
            int row = block_row + c_row + m;
            int col = block_col + c_col + n;
            if(row < M && col < N){
                C[row * N + col] = results[m][n];
            }
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

    dim3 blockDim(BN / TN, BM / TM, 1);
    dim3 gridDim(cuda_utils::ceil_div(N, BN), cuda_utils::ceil_div(M, BM), 1);

    // warmup: run for at least 0.5 s so the GPU clock reaches its maximum
    auto warmup_start = std::chrono::steady_clock::now();
    do {
        sgemm_5<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    } while(std::chrono::steady_clock::now() - warmup_start < std::chrono::milliseconds(500));
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));

    for(int iter = 0; iter < num_iterations; iter++) {
        sgemm_5<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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

$ nvcc -O3 -arch=sm_100 cuda/sgemm_5.cu -o sgemm_5 && ./sgemm_5      (B200, CUDA 13.5)
average time: 5.527702 ms
achieved compute: 24.863668 TFLOPS
theoretical max compute: 74.45 TFLOPS
percent of peak: 33.4%
spot check: 0 of 100 wrong, max relative error 2.72e-06

*/
