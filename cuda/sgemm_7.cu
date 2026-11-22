#include "utils/cuda_utils.cuh"
#include <cstddef>
#include <chrono>
#include <cstdlib>

#define BM 64
#define BN 256
#define BK 32
#define TM 8
#define TN 4

#define NT ((BM / TM) * (BN / TN))   // threads per block: 8 x 64 = 512
// each thread copies A_COPIES float4s (4 neighboring numbers each) of the tile of A per step,
// and B_COPIES of the tile of B
#define A_COPIES (BM * BK / (4 * NT))
#define B_COPIES (BK * BN / (4 * NT))
static_assert(A_COPIES >= 1 && A_COPIES * 4 * NT == BM * BK, "the threads copy the tile of A evenly");
static_assert(B_COPIES >= 1 && B_COPIES * 4 * NT == BK * BN, "the threads copy the tile of B evenly");

__global__ void sgemm_7(const float* A, const float* B, float* C, int M, int K, int N){
    // __align__(16) so that every group of 4 starts on a 16-byte boundary for float4 stores
    // s_A is stored transposed, k first: s_A[k][row] is A[block_row + row][i * BK + k].
    // A thread's TM numbers of A for one k then sit next to each other in row k
    __shared__ __align__(16) float s_A[BK][BM];
    __shared__ __align__(16) float s_B[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // top-left corner of this block's tile of C
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // what this thread computes: a TM x TN square of the C tile,
    // rows c_row .. c_row + TM - 1 and columns c_col .. c_col + TN - 1
    int c_row = ty * TM;
    int c_col = tx * TN;

    // flat index 0..NT-1 for the copies, so neighboring threads copy neighboring groups of 4
    int tid = ty * (BN / TN) + tx;

    // what this thread copies: 4 neighboring numbers in one row of each tile, and the same
    // columns again every a_stride rows of A and every b_stride rows of s_B
    int a_row = tid / (BK / 4);        // a row of the tile of A has BK / 4 groups of 4
    int a_col = (tid % (BK / 4)) * 4;
    int b_row = tid / (BN / 4);        // a row of s_B has BN / 4 groups of 4
    int b_col = (tid % (BN / 4)) * 4;
    const int a_stride = NT / (BK / 4);
    const int b_stride = NT / (BN / 4);

    float results[TM][TN] = {0.0f};
    float reg_A[TM];  // column k of s_A, this thread's TM rows
    float reg_B[TN];  // row k of s_B, this thread's TN columns

    int num_tiles = cuda_utils::ceil_div(K, BK);
    for(int i = 0; i < num_tiles; i++){
        // step i: tile i of this block's strip of A (moves right) and of B (moves down).
        // each copy is one float4 load from global memory.
        // assumes K and N are multiples of 4, so every float4 address in A and B is 16-byte aligned.
        // out-of-range groups are stored as 0, so they add nothing to the sums
        // the 4 numbers of A are neighbors along k, so in the transposed s_A they go into
        // 4 different rows, with 4 single stores
        for(int c = 0; c < A_COPIES; c++){
            int r = a_row + c * a_stride;
            float4 a4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if(block_row + r < M && i * BK + a_col < K){
                a4 = *reinterpret_cast<const float4*>(&A[(block_row + r) * K + (i * BK + a_col)]);
            }
            s_A[a_col + 0][r] = a4.x;
            s_A[a_col + 1][r] = a4.y;
            s_A[a_col + 2][r] = a4.z;
            s_A[a_col + 3][r] = a4.w;
        }
        for(int c = 0; c < B_COPIES; c++){
            int r = b_row + c * b_stride;
            if(i * BK + r < K && block_col + b_col < N){
                *reinterpret_cast<float4*>(&s_B[r][b_col]) =
                    *reinterpret_cast<const float4*>(&B[(i * BK + r) * N + (block_col + b_col)]);
            }
            else{
                *reinterpret_cast<float4*>(&s_B[r][b_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // wait until both tiles are complete before reading them
        __syncthreads();

        for(int k = 0; k < BK; k++){
            // load TM numbers of s_A and TN numbers of s_B into registers once ...
            for(int m = 0; m < TM; m++){
                reg_A[m] = s_A[k][c_row + m];   // TM neighboring numbers in row k
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
        sgemm_7<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    } while(std::chrono::steady_clock::now() - warmup_start < std::chrono::milliseconds(500));
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));

    for(int iter = 0; iter < num_iterations; iter++) {
        sgemm_7<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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

$ nvcc -O3 -arch=sm_100 cuda/sgemm_7.cu -o sgemm_7 && ./sgemm_7      (B200, CUDA 13.5)
average time: 3.067947 ms
achieved compute: 44.798345 TFLOPS
theoretical max compute: 74.45 TFLOPS
percent of peak: 60.2%
spot check: 0 of 100 wrong, max relative error 2.72e-06

*/
