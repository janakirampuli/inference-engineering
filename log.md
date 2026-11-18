# Kernel results

## SGEMM, M = N = K = 4096, FP32

### Method

- GPU: A100 40GB SXM on Modal, compiled with `nvcc -O3 -arch=sm_80`.
- Matrices are row major, filled with random values in [0, 1].
- Timing: warm up for at least 0.5 s so the clock reaches 1410 MHz (100 calls for cuBLAS, 5 for slower kernels), then 20 timed calls with CUDA events, averaged.
- TFLOPS = 2 × M × N × K / time. Peak is 108 SMs × 64 FP32 cores × 2 × 1.41 GHz = 19.49 TFLOPS.
- Check: 100 random elements of C against a CPU dot product in double, relative error under 1e-3.

### Results

| Kernel | Time (ms) | TFLOPS | % of peak | % of cuBLAS |
|---|---|---|---|---|
| cuBLAS `cublasSgemm` | 7.22 | 19.03 | 97.6% | 100% |
| 1. Naive, 16×16 blocks, row from x | 243.39 | 0.56 | 2.9% | 3.0% |
| 1. Naive, 32×32 blocks, row from x | 470.95 | 0.29 | 1.5% | 1.5% |
| 2. Coalesced, 32×32 blocks, column from x | 45.22 | 3.04 | 15.6% | 16.0% |
| 3. Shared memory tiles, 32×32 blocks and tiles | 25.82 | 5.32 | 27.3% | 28.0% |
