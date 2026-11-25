#include "utils/cuda_utils.cuh"
#include <cuda_pipeline.h>
#include <cstddef>
#include <chrono>
#include <cstdlib>

#define BM 64
#define BN 128
#define BK 16
#define WM 32                        // each warp computes a WM x WN piece of the C tile
#define WN 64
#define TM 4                         // each thread computes WMITER x WNITER squares of TM x TN,
#define TN 4                         // spaced apart inside its warp's piece
#define WMITER 2
#define WNITER 2
#define STAGES 4                     // buffers of each tile in shared memory
#define MIN_BLOCKS 2                 // the compiler keeps the registers low enough for this many blocks on an SM

#define NT (32 * (BM / WM) * (BN / WN))  // threads per block: 32 per warp x 4 warps = 128

// a warp's piece is WMITER x WNITER sub-pieces of WSUBM x WSUBN, and the 32 threads of the warp
// cover one sub-piece as WT_ROWS x WT_COLS squares of TM x TN
#define WSUBM (WM / WMITER)          // 16
#define WSUBN (WN / WNITER)          // 32
#define WT_ROWS (WSUBM / TM)         // 4
#define WT_COLS (WSUBN / TN)         // 8

static_assert(BM % WM == 0 && BN % WN == 0, "the warps cover the C tile");
static_assert(TM % 2 == 0, "each FFMA2 works on 2 neighboring rows of a square");
static_assert(WT_ROWS * WT_COLS == 32, "the 32 threads of a warp cover one sub-piece");
static_assert(BK % 2 == 0, "the last k of a step loads the next tile into set 0");
// each thread copies A_COPIES single numbers of the tile of A per step, and B_COPIES groups of
// 4 neighboring numbers of the tile of B
#define A_COPIES (BM * BK / NT)
#define B_COPIES (BK * BN / (4 * NT))
#define B_STRIDE (NT / (BN / 4))     // a thread's copies of B are B_STRIDE rows apart
static_assert(A_COPIES >= 1 && A_COPIES * NT == BM * BK && BK % 8 == 0 && BM % 4 == 0, "the threads copy the tile of A evenly");
static_assert(B_COPIES >= 1 && B_COPIES * 4 * NT == BK * BN, "the threads copy the tile of B evenly");

// each row of s_A has PAD extra numbers at its end, so that row k + 1 starts 4 banks after row k
#define PAD 4
#define SMEM_BYTES (STAGES * BK * ((BM + PAD) + BN) * 4)

// start copying tile i of A and of B into one buffer of s_A and s_B.
// the copies are asynchronous: they go from global memory straight into shared memory, without registers,
// and the thread goes on while they are on the way. a number or group outside the matrix is filled with zeros.
// assumes N is a multiple of 4, so every 16-byte copy of B is 16-byte aligned
__device__ __forceinline__ void copy_tiles(const float* A, const float* B, int M, int K, int N, int i,
                                           int block_row, int block_col, float (*s_A)[BM + PAD], float (*s_B)[BN],
                                           int tid, int b_row, int b_col){
    // each copy of A moves one number into the transposed s_A. the 32 copies of a warp take 8 neighboring
    // values of k in each of 4 neighboring rows of A, so they read whole sectors, and with the padding
    // number (k, r) of s_A sits in bank (4k + r) % 32, so the 32 numbers land in 32 different banks
    for(int c = 0; c < A_COPIES; c++){
        int e = tid + c * NT;
        int r = (e / 8) % BM;
        int k = (e / 8 / BM) * 8 + e % 8;
        int row = block_row + r, col = i * BK + k;
        bool inside = row < M && col < K;
        __pipeline_memcpy_async(&s_A[k][r], inside ? &A[row * K + col] : A, 4, inside ? 0 : 4);
    }
    for(int c = 0; c < B_COPIES; c++){
        int r = b_row + c * B_STRIDE;
        int row = i * BK + r, col = block_col + b_col;
        bool inside = row < K && col < N;
        __pipeline_memcpy_async(&s_B[r][b_col], inside ? &B[row * N + col] : B, 16, inside ? 0 : 16);
    }
}

// read this thread's numbers of s_A and s_B for one k into one set of registers:
// TM numbers of s_A for each of the WMITER sub-pieces, in pairs, and TN numbers of s_B for each of the WNITER
__device__ __forceinline__ void load_k(float2 (&reg_A)[WMITER * TM / 2], float (&reg_B)[WNITER * TN],
                                       float (*s_A)[BM + PAD], float (*s_B)[BN], int k, int a_base, int b_base){
    for(int wm = 0; wm < WMITER; wm++){
        for(int m = 0; m < TM; m += 2){
            int row = a_base + wm * WSUBM + m;
            reg_A[(wm * TM + m) / 2] = make_float2(s_A[k][row], s_A[k][row + 1]);
        }
    }
    for(int wn = 0; wn < WNITER; wn++){
        for(int n = 0; n < TN; n++){
            reg_B[wn * TN + n] = s_B[k][b_base + wn * WSUBN + n];
        }
    }
}

// __launch_bounds__(NT, MIN_BLOCKS): blocks of NT threads, and at least MIN_BLOCKS blocks on an SM,
// so the compiler may use up to 65536 / (MIN_BLOCKS * NT) registers for each thread
__global__ void __launch_bounds__(NT, MIN_BLOCKS) sgemm_11(const float* A, const float* B, float* C, int M, int K, int N){
    // STAGES buffers of each tile: the block computes on one while the copies of the next tiles
    // are on the way into the others.
    // s_A is stored transposed, k first: s_A[buf][k][row] is A[block_row + row][i * BK + k].
    // the buffers need more than the 48 KB that a block can declare, so they sit in dynamic shared memory
    extern __shared__ __align__(16) float smem[];
    float (*s_A)[BK][BM + PAD] = reinterpret_cast<float (*)[BK][BM + PAD]>(smem);
    float (*s_B)[BK][BN] = reinterpret_cast<float (*)[BK][BN]>(smem + STAGES * BK * (BM + PAD));

    int tid = threadIdx.x;

    // top-left corner of this block's tile of C
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // which piece of the C tile this thread's warp computes
    int warp_id = tid / 32;
    int warp_row = (warp_id / (BN / WN)) * WM;
    int warp_col = (warp_id % (BN / WN)) * WN;

    // where this thread's TM x TN square sits inside each sub-piece of its warp.
    // neighboring threads compute neighboring squares, so a warp reads neighboring numbers of s_A and s_B
    int lane = tid % 32;
    int t_row = (lane / WT_COLS) * TM;
    int t_col = (lane % WT_COLS) * TN;

    // what this thread copies of B: 4 neighboring numbers in one row, and the same columns again
    // every B_STRIDE rows
    int b_row = tid / (BN / 4);        // a row of s_B has BN / 4 groups of 4
    int b_col = (tid % (BN / 4)) * 4;

    // the thread's rows go in pairs of 2 neighboring rows: results[p][n] holds rows 2p and 2p + 1
    // of column n
    float2 results[WMITER * TM / 2][WNITER * TN];
    for(int p = 0; p < WMITER * TM / 2; p++){
        for(int n = 0; n < WNITER * TN; n++){
            results[p][n] = make_float2(0.0f, 0.0f);
        }
    }
    // two sets of registers for the numbers of s_A and s_B: the FFMA2s for k use set k % 2,
    // while the loads for k + 1 fill the other set
    float2 reg_A[2][WMITER * TM / 2];  // row k of s_A, this thread's rows in each sub-piece, in pairs
    float reg_B[2][WNITER * TN];       // row k of s_B, this thread's columns in each sub-piece
    int a_base = warp_row + t_row;
    int b_base = warp_col + t_col;

    int num_tiles = cuda_utils::ceil_div(K, BK);

    // before the loop: start copying tiles 0 to STAGES - 2 into buffers 0 to STAGES - 2.
    // each commit closes a group of copies, and later the thread can wait for all but the newest groups
    for(int s = 0; s < STAGES - 1; s++){
        if(s < num_tiles){
            copy_tiles(A, B, M, K, N, s, block_row, block_col, s_A[s], s_B[s], tid, b_row, b_col);
        }
        __pipeline_commit();
    }
    // wait for tile 0, and load its numbers for k = 0 into set 0
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    load_k(reg_A[0], reg_B[0], s_A[0], s_B[0], 0, a_base, b_base);

    for(int i = 0; i < num_tiles; i++){
        int cur = i % STAGES;  // buffer that holds tile i

        // start copying tile i + STAGES - 1 into the buffer of tile i - 1, which no thread reads any more:
        // the last loads from it came before the wait at the end of the last step. the commit comes even
        // when there is no tile left, so that the waits always count the same number of groups
        int next = i + STAGES - 1;
        if(next < num_tiles){
            copy_tiles(A, B, M, K, N, next, block_row, block_col, s_A[next % STAGES], s_B[next % STAGES],
                       tid, b_row, b_col);
        }
        __pipeline_commit();

        // the loop is written out in full, so that k % 2 is known when the program is compiled
        // and the two sets stay in registers
        #pragma unroll
        for(int k = 0; k < BK; k++){
            if(k < BK - 1){
                // start loading the numbers for k + 1 into the other set. the FFMA2s below don't use
                // that set, so the loads arrive while they run
                load_k(reg_A[(k + 1) % 2], reg_B[(k + 1) % 2], s_A[cur], s_B[cur], k + 1, a_base, b_base);
            } else {
                // before the last k: wait until this thread's copies of tile i + 1 have arrived, then
                // for every thread, and start loading the numbers of tile i + 1 for k = 0, so that they
                // too arrive while the last FFMA2s of this step run
                __pipeline_wait_prior(STAGES - 2);
                __syncthreads();
                if(i + 1 < num_tiles){
                    load_k(reg_A[(k + 1) % 2], reg_B[(k + 1) % 2], s_A[(i + 1) % STAGES], s_B[(i + 1) % STAGES],
                           0, a_base, b_base);
                }
            }
            // one FFMA2 multiplies a pair of rows of s_A by one number of s_B and adds to 2 results
            for(int p = 0; p < WMITER * TM / 2; p++){
                for(int n = 0; n < WNITER * TN; n++){
                    float b = reg_B[k % 2][n];
                    results[p][n] = __ffma2_rn(reg_A[k % 2][p], make_float2(b, b), results[p][n]);
                }
            }
        }
    }

    for(int wm = 0; wm < WMITER; wm++){
        for(int m = 0; m < TM; m++){
            for(int wn = 0; wn < WNITER; wn++){
                for(int n = 0; n < TN; n++){
                    int row = block_row + warp_row + wm * WSUBM + t_row + m;
                    int col = block_col + warp_col + wn * WSUBN + t_col + n;
                    if(row < M && col < N){
                        float2 pair = results[(wm * TM + m) / 2][wn * TN + n];
                        C[row * N + col] = (m % 2 == 0) ? pair.x : pair.y;
                    }
                }
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

    dim3 blockDim(NT, 1, 1);
    dim3 gridDim(cuda_utils::ceil_div(N, BN), cuda_utils::ceil_div(M, BM), 1);

    // a block may use more than 48 KB of dynamic shared memory only after this call
    CUDA_CHECK(cudaFuncSetAttribute(sgemm_11, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    // warmup: run for at least 0.5 s so the GPU clock reaches its maximum
    auto warmup_start = std::chrono::steady_clock::now();
    do {
        sgemm_11<<<gridDim, blockDim, SMEM_BYTES>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    } while(std::chrono::steady_clock::now() - warmup_start < std::chrono::milliseconds(500));
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));

    for(int iter = 0; iter < num_iterations; iter++) {
        sgemm_11<<<gridDim, blockDim, SMEM_BYTES>>>(d_A, d_B, d_C, M, K, N);
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

$ nvcc -O3 -arch=sm_100 cuda/sgemm_11.cu -o sgemm_11 && ./sgemm_11      (B200, CUDA 13.5)
average time: 2.394867 ms
achieved compute: 57.388967 TFLOPS
theoretical max compute: 74.45 TFLOPS
percent of peak: 77.1%
spot check: 0 of 100 wrong, max relative error 2.72e-06

*/
