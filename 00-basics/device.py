import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import benchmark, device_info, empty_cache, gbytes_per_s, get_device, tflops

device = get_device()

print("== what this machine is ==")
for key, value in device_info(device).items():
    print(f"{key}: {value}")


print("\n== memory bandwidth ==")
# 256MB is far too big to sit in cache, so this is real trips to memory
n = 256 * 1024**2 // 4
a = torch.randn(n, device=device)
out = torch.empty_like(a)
# a copy reads one buffer and writes another, so 2n elements move
bw = gbytes_per_s(2 * n * 4, benchmark(lambda: out.copy_(a), device))
print(f"{bw:.0f} GB/s")
del a, out
empty_cache(device)


print("\n== compute ==")
# keep the matrices small enough to stay in cache. memory is then not the
# limit, so what we measure is the actual compute ceiling.
peak = {}
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    best = 0
    for m in [1024, 2048, 4096]:
        try:
            x = torch.randn(m, m, device=device, dtype=dtype)
            y = torch.randn(m, m, device=device, dtype=dtype)
            # 1 multiply and 1 add per element of k
            best = max(best, tflops(2 * m**3, benchmark(lambda: x @ y, device)))
            del x, y
            empty_cache(device)
        except Exception:
            pass
    peak[dtype] = best
    print(f"{str(dtype).replace('torch.', ''):>9}: {best:>6.1f} TFLOP/s")


print("\n== ridge point ==")
fast = max(peak, key=peak.get)
ridge = peak[fast] * 1e12 / (bw * 1e9)
print(f"{ridge:.0f} FLOP/byte  ({peak[fast]:.1f} TFLOP/s / {bw:.0f} GB/s)")


print("\n== arithmetic intensity vs batch size ==")
# this is the shape of generating a token: a small batch of vectors
# multiplied against one big weight matrix.
k = 4096
w = torch.randn(k, k, device=device, dtype=fast)
itemsize = torch.finfo(fast).bits // 8
print(f"{'batch':>6} {'FLOP/byte':>10} {'TFLOP/s':>9} {'% of peak':>10}")
for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    x = torch.randn(batch, k, device=device, dtype=fast)
    flops = 2 * batch * k * k
    # we read the batch and the weights, then write the answer
    nbytes = (batch * k + k * k + batch * k) * itemsize
    got = tflops(flops, benchmark(lambda: x @ w, device))
    print(f"{batch:>6} {flops / nbytes:>10.1f} {got:>9.2f} {100 * got / peak[fast]:>9.1f}%")
    del x
    empty_cache(device)
del w
empty_cache(device)


print("\n== kernel launch overhead ==")
tiny = torch.randn(1, device=device)
print(f"{benchmark(lambda: tiny.add_(1.0), device) * 1e6:.1f} mu_s to launch one kernel")


print("\n== host to device ==")
host = torch.randn(64 * 1024**2 // 4)
dev = torch.empty(host.numel(), device=device)
t = benchmark(lambda: dev.copy_(host), device)
print(f"{gbytes_per_s(64 * 1024**2, t):.0f} GB/s")
