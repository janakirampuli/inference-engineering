# Kernel results

## SGEMM on the B200, M = N = K = 4096, FP32

### Method

- GPU: B200 on Modal (148 SMs, 1965 MHz max, 128 FP32 cores per SM), CUDA 13.5, compiled with `nvcc -O3 -arch=sm_100`. Details in `cuda/gpu_info_b200.txt`.
- Matrices are row major, filled with random values in [0, 1].
- Timing: warm up for at least 0.5 s so the clock reaches its maximum (the kernels launch and sync until 0.5 s has passed), then 20 timed calls with CUDA events, averaged. Each binary runs twice; the two runs agree within 0.2%.
- TFLOPS = 2 × M × N × K / time. Peak is computed at runtime by `cuda_utils::fp32_peak_tflops()`: 148 × 128 × 2 × 1.965 GHz = 74.45 TFLOPS. Under load the SM clock is 1.93 GHz (Nsight Compute).
- Check: 100 random elements of C against a CPU dot product in double, relative error under 1e-3. Kernels 4 and up are also checked at M = 1000, K = 996, N = 1004 under `compute-sanitizer`.

### Results

| Kernel | Time (ms) | TFLOPS | % of peak | % of cuBLAS |
|---|---|---|---|---|
| cuBLAS `cublasSgemm` (picks `cutlass3x_sm100_simt_sgemm_..._128x64x16_..._3`) | 2.16 | 63.60 | 85.4% | 100% |
| 1. Naive, 8×8 blocks | 66.66 | 2.06 | 2.8% | 3.2% |
| 2. Coalesced, 32×32 blocks | 19.50 | 7.05 | 9.5% | 11.1% |
| 3. Shared memory tiles, 32×32 | 13.33 | 10.31 | 13.9% | 16.2% |
| 4. 1D blocktiling, 64×64 tile, TM = 16, BK = BM / TM = 4 (60 regs) | 6.35 | 21.65 | 29.1% | 34.0% |
| 4, TM = 4, BK 16 | 9.86 | 13.93 | 18.7% | 21.9% |
| 4, TM = 8, BK 8 | 7.72 | 17.80 | 23.9% | 28.0% |
| 4, TM = 32, BK 2 | 7.48 | 18.36 | 24.7% | 28.9% |
| 5. 2D blocktiling, one copy per thread, 64×64, 4×4, BK = BM / (TM·TN) = 4 (56 regs) | 5.53 | 24.86 | 33.4% | 39.1% |
| 5, 64×64, 8×4, BK 2 (72 regs) | 5.50 | 24.98 | 33.6% | 39.3% |
| 5, 64×64, 4×8, BK 2 | 6.92 | 19.85 | 26.7% | 31.2% |
| 5, 64×64, 8×8, BK 1 | 7.63 | 18.00 | 24.2% | 28.3% |
| 5, 128×128, 4×4, BK 8 | 5.78 | 23.79 | 32.0% | 37.4% |
| 5, 128×128, 8×4, BK 4 (80 regs, 1 block/SM) | 7.54 | 18.22 | 24.5% | 28.6% |
| 5, 128×128, 4×8, BK 4 | 8.69 | 15.81 | 21.2% | 24.9% |
| 5, 128×128, 8×8, BK 2 (127 regs) | 7.07 | 19.43 | 26.1% | 30.5% |
| 6. float4 copies (each thread copies BM·BK / (4·NT) float4s of A and BK·BN / (4·NT) of B per step), 128×128, 8×4, BK 32 (64 regs) | 3.45 | 39.79 | 53.4% | 62.6% |
| 7. transposed s_A (s_A[BK][BM]), 64×256, 8×4, BK 32 (64 regs) | 3.07 | 44.80 | 60.2% | 70.4% |
| 8. double buffering (s_A[2][BK][BM], s_B[2][BK][BN], one sync per step), 64×256, 8×8, BK 16, `__launch_bounds__(256, 2)` (128 regs, 2 blocks/SM) | 2.91 | 47.22 | 63.4% | 74.2% |
| 9. warp tiling, 128×128, warp 32×64, thread 2×2 squares of 4×4, 256 threads, BK 16, `__launch_bounds__(256, 2)` (127 regs, 2 blocks/SM) | 2.67 | 51.57 | 69.3% | 81.1% |

### Notes

cuBLAS: FP32 cores (SIMT), no tensor cores; 128×64 tile of C, BK = 16, 3 pipeline stages; 128 threads per block with 248 registers each (2 blocks, 8 warps per SM); FMA pipe 85.6% busy.

Kernel 1 on the B200 (Nsight Compute, 32×32 blocks): L1 throughput 99%, L1 hit rate 99%, DRAM 0.04%. Each warp load needs about 16.5 sectors on average (70.9 G sectors for 4.29 G requests), so the L1 pipe, not memory, sets the time.

Kernel 2 on the B200 (Nsight Compute): 2.5 sectors per warp load (10.7 G sectors for 4.29 G requests, against 16.5 in kernel 1), L1 hit rate 95%, DRAM 0.4%. The LSU pipe is busy 76% and the FMA pipe 29%; the top stalls are `lg_throttle` 12.7 (the queue for global loads is full) and `long_scoreboard` 6.9. Two global loads for every `FFMA` keep the load pipe full.

Kernel 4: BK = BM / TM keeps one copy of each tile per thread; TM = 16 (BK 4) is fastest.

Kernel 5: one copy per thread needs a square tile and BK = BM / (TM · TN); 64×64 with 4×4 (BK 4) ties with 64×64, 8×4 (BK 2). 128×128 with 8×4 needs 80 registers, so only one block of 512 threads fits on an SM.

Kernel 6 (Nsight Compute): shared memory pipe 87% busy, FMA pipe 54%, DRAM 1%; top stall `short_scoreboard` 4.86 (waiting for shared loads), `long_scoreboard` 0.70, `barrier` 0.63. Per warp per step: 512 `FFMA` (128 cycles) against 144 shared loads (192 passes): 8 single `LDS` for the column of `s_A` and 1 `LDS.128` for `s_B` at each k. Kernel 5 (BK 4): `long_scoreboard` 6.06 and `barrier` 2.44 dominate.

Kernel 7 (same session: kernel 6 at 3.54 ms): shared load instructions 604 M → 201 M (3 per k instead of 9: 2 `LDS.128` for `s_A`, 1 for `s_B`), shared load passes 805 M → 539 M; shared store passes 35 M → 86 M (4-way bank conflicts in the transposed stores, 53 M conflicts). Shared pipe 87% → 72%, FMA pipe 54% → 58%. Top stall `short_scoreboard` 4.86 → 0.70; now `not_selected` 4.76, `barrier` 1.40, `dispatch_stall` 1.19, `long_scoreboard` 0.94.

Kernels 6 and 7 size sweep (one float4 of each tile per thread, so BM = BN and BK = 4 · BM / (TM · TN)), times in ms, kernel 6 / kernel 7: 64×64 4×4 BK16: 3.68 / 3.95; 64×64 8×4 BK8: 3.84 / 3.41; 64×64 4×8 BK8: 5.68 / 5.34; 64×64 8×8 BK4: 5.09 / 4.72; 128×128 4×4 BK32: 3.88 / 4.18; **128×128 8×4 BK16: 3.54 / 3.26**; 128×128 4×8 BK16: 7.03 / 4.89; 128×128 8×8 BK8: 5.28 / 4.39; 256×256 8×8 BK16: does not launch (1,024 threads × 89 registers).

Kernel 8 (same session: kernel 7 at 3.253 ms, kernel 8 at 3.129 ms): SASS has the 2 `LDG.E.128` of tile i+1 before the `FFMA` loop and the `STS` after it, one `BAR` per step. Stalls vs kernel 7: `long_scoreboard` 0.94 → 0.01, `barrier` 1.40 → 0.21, `mio_throttle` 0.49 → 0.01, `not_selected` 4.75 → 1.94 (still top), `short_scoreboard` 0.71 → 1.18 (second). 71 registers → 1 block of 512 per SM (warps active 47% → 25%). FMA pipe 58% → 61%, issue active 69% → 74%. Shared memory 32 KB per block. compute-sanitizer memcheck and racecheck clean at M, K, N = 1000, 996, 1004.

Kernel 8 size sweep, times in ms without / with `__launch_bounds__(NT, 1024 / NT)` (64-register cap), registers in brackets: 64×64 4×4 BK16: 4.14 (56) / 3.98 (63); 64×64 8×4 BK8: 3.58 (77) / 3.34 (64); 64×64 4×8 BK8: 5.29 (72) / 5.48 (64, 16 B spill); 64×64 8×8 BK4: 4.84 (109) / 21.59 (64, 712 B spill); **128×128 8×4 BK16: 3.13 (71)** / 3.53 (64, 64 B spill); 128×128 4×8 BK16: 4.86 (71) / 5.22 (64, 64 B spill); 128×128 8×8 BK8: 3.44 (95) / 20.52 (64, 284 B spill). Left out: 128×128 4×4 BK32 and 256×256 8×8 BK16 (two buffers need 64 KB, above the 48 KB static shared memory limit).

Kernel 9 (same session: kernel 8 at 3.129 ms, kernel 9 at 2.744 ms; Nsight Compute, kernel 8 / kernel 9 without bounds / kernel 9): shared load instructions 201 M / 134 M / 134 M (per thread per k: 3 `LDS.128` for 32 `FFMA` vs 4 for 64), shared load passes 537 M / 406 M / 403 M (about 3 passes per `LDS.128` in all three), load bank conflicts 0 / 3.6 M / 0.05 M, store conflicts 50 M / 17 M / 17 M. Registers 71 / 105 / 127, blocks per SM 1 / 2 / 2 (16 warps in all three). `short_scoreboard` 1.18 / 1.10 / 0.57, `not_selected` 1.94 / 1.88 / 1.91 (top), `barrier` 0.21 / 0.19 / 0.39. FMA pipe 61% / 61% / 69%, issue active 74% / 76% / 81%. Without bounds the SASS loads the 4 `LDS.128` of each k just before its 64 `FFMA`s; with `__launch_bounds__(256, 2)` (127 registers) it issues 7 `LDS.128` up front and the loads of later k between the `FFMA`s. compute-sanitizer memcheck and racecheck clean at M, K, N = 1000, 996, 1004.

Control: kernel 8 at 128×128 with 8×8 per thread (no warp tiling, 256 threads, BK 8) and `__launch_bounds__(256, 2)`: 127 registers, 3.47 ms (3.45 without bounds, 95 registers). Same 134 M shared load instructions as kernel 9, but 268 M load bank conflicts and 671 M load passes (each thread's 8 columns are neighbors, so neighboring threads read `s_B` 32 bytes apart). FMA pipe 62%, `short_scoreboard` 1.94.

Kernel 9 ablation (old kernel 9, before the copy loop; test file deleted; flags `-DNO_TRANSPOSE` puts `s_A` back row by row as in kernel 6, `-DNO_DB` uses one buffer and two syncs per step as in kernel 7, `-DNO_LB` drops `__launch_bounds__`; best kernel 9 layout; 3 runs each, ms with / without the bound, registers in brackets): kernel 9 2.746 (127) / 3.095 (105); no double buffering 3.834 (99) / 3.946 (91); no transpose 5.184 (107) / 5.973 (95); neither 5.517 (96) / 5.583 (91). All 2 blocks of 256 per SM (warps active 23 to 25%). Nsight Compute with the bound, kernel 9 / no double buffering / no transpose / neither: shared load instructions 134 M / 134 M / 336 M / 336 M (no transpose: 8 single `LDS` for `s_A` + 2 `LDS.128` for `s_B` per k), shared load passes 403 M / 403 M / 1,342 M / 1,342 M, load bank conflicts 0.05 M / 0.07 M / 805 M / 805 M (4 passes per load: lanes 0, 8, 16, 24 read rows 4 apart, 32 floats apart, same bank), FMA pipe 69% / 57% / 39% / 35%. No double buffering: `long_scoreboard` stall 29× kernel 9's and `barrier` 1.4×. In kernel 7 → 8 double buffering gained only 1.04× because kernel 8 also dropped from 32 warps per SM to 16; with 16 warps and no double buffering, kernel 9 is 1.40× slower.

Copy loop (2026-11-20): kernels 6 to 9 first had each thread copy exactly one float4 of each tile per step, which forced square tiles and BK = 4·NT/BM; the rows for kernels 6 to 9 above and the profile notes for kernels 7 to 9 below that mention 128×128 8×4 BK 16 (kernels 6 to 8) or BK 8 (kernel 9) are from that version. Kernels 6 to 9 now loop over several float4 copies per thread (`cuda/sweep.py`, results in `runs/sweep_k6` to `runs/sweep_k9` on the volume). Sweep: BM, BN in {64, 128, 256}, TM, TN in {4, 8} (kernel 9: 4×4 squares, WM, WN in {16..128}, WMITER × WNITER in {2, 4, 8}), BK in {8, 16, 32}; kernel 8 without bound, with (NT, 1024/NT) and (NT, 2); kernel 9 without and with (NT, 2). Kernel 6: 52 configs, best 128×128 8×4 BK32 3.458 (old size BK16 3.541). Kernel 7: best 64×256 8×4 BK32 3.072, then 64×128 8×4 BK16/32 3.145, 128×128 8×4 BK16 3.255, BK32 3.324; best 8×8 64×128 BK16 3.229. Kernel 8: 35 sizes, best 64×256 8×8 BK16 (NT, 2) 2.913 (128 regs), 128×128 8×8 BK16 (NT, 2) 2.936, 64×128 8×4 BK16 no bound 2.956; 64×256 8×8 BK16 without bound 3.342 (129 regs, 1 block/SM); (NT, 4) spills (20 ms). Kernel 9: 268 sizes, best 64×256 BK16 warp 16×128 (NT, 2) 2.665, 128×128 BK16 warp 32×64 (NT, 2) 2.667; without bound best 2.697; old size (BK 8) 2.747; cuBLAS's shape 128×64 BK16 128 threads warp 64×32 2×2 (NT, 2) 2.948.

Kernels 5 to 9 after the copy loop (same session, Nsight Compute, `runs/ncu/sgemm_5..9.ncu-repz`, kernel 5 / 6 / 7 / 8 / 9): time 5.51 / 3.454 / 3.068 / 2.910 / 2.665 ms; FMA pipe 36.9 / 54.8 / 61.8 / 65.4 / 71.1%; shared pipe (wavefronts) 55 / 83 / 73 / 87 / 63%; issue active 50.6 / 73.8 / 72.4 / 75.6 / 81.6%; global load instructions 67.1 / 8.4 / 10.5 / 10.5 / 8.4 M; shared load instructions 268 / 587 / 201 / 134 / 134 M; shared load passes 805 / 790 / 537 / 671 / 403 M; shared load bank conflicts 0.16 / 1.2 / 0.6 / 268 / 0.02 M; shared store passes 72 / 34 / 103 / 67 / 84 M; store conflicts 4.8 / 0.3 / 60.6 / 25.5 / 50.5 M; registers 56 / 64 / 64 / 128 / 127; warps active 49 / 47 / 47 / 23 / 24%. Stalls per issued instruction, `long_scoreboard` 6.05 / 0.46 / 0.57 / 0.05 / 0.01, `barrier` 2.43 / 0.87 / 1.11 / 0.31 / 0.26, `short_scoreboard` 2.57 / 0.47 / 2.50 / 1.43 / 0.62, `mio_throttle` 0.37 / 1.20 / 0.80 / 0.10 / 0.07, `not_selected` 1.49 / 4.77 / 3.49 / 1.54 / 2.10. Kernel 8 with 8×8 next to each other: lanes 0 and 4 read s_B 128 bytes apart, 2 passes per LDS.128; kernel 9 removes the load conflicts.

Swizzle test (kernel 9 at 128×128 BK 16, rows of the transposed s_A XORed with ((k >> 2) & 3) << 3, so the 4 threads that copy one row of A store into 4 different banks; s_A read with an explicit float4 at (row ^ swz)): store conflicts 50.5 M → 0.1 to 0.3 M, store passes 84 M → 34 M, load passes unchanged (403 M), shared pipe 63% → 50%. But slower. As written, the compiler stopped unrolling the loop over k (1 k per pass, 4 LDS.128 just before the FFMAs, no loads ahead): 3.029 ms (cuBLAS 2.165, kernel 9 2.672 in the same session). With `#pragma unroll` on the loop over k (full unroll, 2048 FFMA and 128 LDS.128 in the SASS, 3 STL / 2 LDL): 2.981 ms, issue active 73.6% (kernel 9 81.6%), FMA pipe 63.6% (71.1%); without `__launch_bounds__` 3.687 ms (147 regs). 64×256 version 3.075 ms. Stores are 17% of kernel 9's shared memory passes and the shared pipe is 63% busy, so removing the store conflicts can save little; the loss comes from the different loop schedule. Files deleted after the test.

A100 check (A100-SXM4-40GB, `-arch=sm_80`, same code and B200-tuned sizes, 2 runs each, FP32 peak 19.49 TFLOPS): cuBLAS 7.248 ms, 18.96 TFLOPS (97.3% of peak); kernel 8 8.562 ms, 16.05 TFLOPS (82.4% of peak, 84.7% of cuBLAS); kernel 9 8.287 ms, 16.58 TFLOPS (85.1%, 87.5%; 110 registers); kernel 9 without `__launch_bounds__` 8.166 ms, 16.83 TFLOPS (86.4%, 88.8%). On the B200 the same kernel 9 is 67.3% of peak and 78.7% of cuBLAS: the B200 has 128 FP32 cores per SM against 64 on the A100, with shared memory still about one warp load per cycle.

Kernel 9 size sweep (BM = BN, one float4 of each tile per thread), times in ms without / with `__launch_bounds__(NT, 2)`, registers in brackets: 128, warp 32×64, 4×4, 2×2: 3.09 (105) / **2.74 (127)**; 128, warp 64×32, 4×4, 2×2: 3.10 / 2.75 (127); 128, warp 32×64, 8×4, 1×2: 3.09 (105) / 3.09 (127); 128, warp 64×32, 8×4, 1×2: 3.10 (105) / 2.91 to 3.09 (127, varied between runs); 128, warp 16×128, 4×4, 1×4: 3.16 (94) / 3.47 (107); 128, warp 32×32, 4×4, 1×2 (512 threads, BK 16): 3.26 (72) / 3.09 (64); 128, warp 32×32, 4×4, 2×1: 3.15 (72) / 3.41 (64, 68 B spill); 64, warp 32×32, 4×4, 2×1 (128 threads): 3.44 / 3.56; 64, warp 32×32, 4×4, 1×2: 3.69 / 3.73.

Swizzle test on kernel 7 (B200, 64×256, BK 32; row k of s_A keeps column j at j ^ ((((k >> 2) & 7) << 2)), so the 8 threads that copy one row of A store into 8 banks; s_A loaded with explicit float4s; test file deleted, concept kept in the note without a kernel): kernel 7 3.070 to 3.074 ms. Swizzled 3.677 ms: store conflicts 60.5 M → 4.3 M, store wavefronts 102 M → 46 M, load conflicts 0.6 M → 0, shared pipe 73% → 56%, FMA pipe 62% → 52%, but 66 registers, so only 1 block of 512 threads per SM. With `#pragma unroll` on k: 3.384 ms, 87 registers. With `__launch_bounds__(512, 2)` (64 registers, 8 B spill): 3.469 ms, store conflicts 1.7 M, issue active 76.7%, FMA pipe 54.8%; plus `#pragma unroll`: 3.257 ms (52 B spill stores), FMA pipe 58.0%. Every swizzled version is slower than plain kernel 7: the XOR address math costs registers and instructions, and the stores were only about 16% of the shared memory passes.

Waves of kernel 9 (B200, same code with M, K, N from argv; 128×128 blocks, 2 per SM; script deleted after the run): K = 4096, N = 1024 (8 block columns), M = 512 to 10,240 in steps of 128. Time is flat inside each step of 148 blocks and jumps at 148, 296, 444 and 592: 0.423 ms up to 144 blocks, 0.766 to 0.774 ms for 152 to 296, 1.168 to 1.173 ms for 304 to 440, 1.525 to 1.527 ms for 448 to 592, 1.913 to 1.917 ms for 600 to 640. So the step is one more block on the busiest SM, not 296 resident slots; two co-resident blocks each run at 0.42 / 0.77 = 55% of their speed alone. At N = K = 4096: M = 4096 (1,024 blocks, at most 7 per SM) 2.666 ms, 51.56 TFLOPS; 4224 (1,056, 8) 3.026 ms, 46.84; 4352 3.030, 48.20; 4480 3.032, 49.58; 4608 3.033, 50.98; 4736 (1,184, exactly 8 on every SM) 3.035 ms, 52.36 TFLOPS; 4864 (1,216, 9) 3.403 ms, 47.96. All spot checks 0 wrong.

## Tensor cores on the B200, M = N = K = 4096, BF16 inputs, FP32 sums

### Method

Same timing as above (0.5 s warmup, 20 timed runs, CUDA events). Inputs are random numbers in [0, 1] rounded to BF16; the spot check compares 100 elements with an FP64 sum of the same rounded inputs (tolerance 1e-3). Peak is the listed dense BF16 tensor core speed of the B200, 2,250 TFLOPS (4,500 with 2:4 sparsity).

### Results

| Kernel | Time (ms) | TFLOPS | % of peak | % of cuBLAS |
|---|---|---|---|---|
| cuBLAS `cublasGemmEx` BF16 → FP32 (picks `nvjet_sm100_tss_128x256_64x6_2x2f_2cta_h_bz_NNT`) | 0.099 | 1,383 | 61.5% | 100% |
| 1. `mma.sync.m16n8k16` BF16, one warp per 16×8 tile of C, fragments loaded straight from global memory, 2 warps per block (32 regs) | 10.29 | 13.36 | 0.6% | 1.0% |

For comparison with the FP32 kernels (FP32 inputs): `cublasSgemm` 2.157 ms, max relative error 2.72e-06; with `CUBLAS_EMULATE_SINGLE_PRECISION=1` (BF16x9 on the tensor cores, `cutlass3x_sm100_tensorop_s256x128x16_9xbf16gemm_...`) 0.84 to 0.90 ms, max relative error 6.94e-07. `CUBLAS_EMULATION_STRATEGY=eager` gives the same kernels and times.

### Notes

cuBLAS BF16 varies 0.099 to 0.102 ms between runs (1,354 to 1,383 TFLOPS); at 4096 one GEMM is only about 0.1 ms.

Kernel 1: one `HMMA.16816.F32.BF16` per step in the SASS. Warps per block barely matter: 1: 10.59, 2: 10.28, 4: 10.33, 8: 10.37, 16: 10.48 ms. Faster than FP32 kernel 3 (13.3 ms) but slower than FP32 kernel 4 (6.35 ms): every number comes from global memory with 2-byte loads, and nothing is shared between warps. compute-sanitizer memcheck clean at M, K, N = 1000, 996, 1004.
