import functools
import platform
import subprocess
import time

import torch


def get_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    # gpu calls are async, so without this we time the launch, not the work
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


@functools.cache
def _sysctl(key):
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _cpu_brand():
    return _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()


def device_info(device):
    info = {
        "device": device.type,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info |= {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "sms": props.multi_processor_count,
            "vram_gb": round(props.total_memory / 1024**3, 1),
            "l2_mb": round(getattr(props, "L2_cache_size", 0) / 1024**2, 1),
        }
    elif device.type == "mps":
        # no gpu core count or vram to report - the gpu uses the same
        # memory as the cpu, over the same bus
        mem = _sysctl("hw.memsize")
        info |= {
            "name": _cpu_brand(),
            "unified_memory_gb": round(int(mem) / 1024**3, 1) if mem else None,
            "note": "unified memory: CPU and GPU share one pool and one bus",
        }
    else:
        info |= {
            "name": _cpu_brand(),
            "cores": _sysctl("hw.physicalcpu"),
            "torch_threads": torch.get_num_threads(),
        }

    return info


def benchmark(fn, device, target_s=0.05, trials=7):
    # warmup
    for _ in range(3):
        fn()
    sync(device)

    # time one call, then run however many fit in target_s.
    # keeps the same code usable on a cpu, a laptop gpu and an h100.
    start = time.perf_counter()
    fn()
    sync(device)
    single = max(time.perf_counter() - start, 1e-9)
    iters = max(1, int(target_s / single))

    times = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        sync(device)
        times.append((time.perf_counter() - start) / iters)
    # fastest = least polluted
    return min(times)


def gbytes_per_s(total_bytes, seconds):
    return total_bytes / seconds / 1e9


def tflops(total_flops, seconds):
    return total_flops / seconds / 1e12
