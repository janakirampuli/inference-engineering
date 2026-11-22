import itertools, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

KERNEL = int(sys.argv[1])
OUT = f"runs/sweep_k{KERNEL}"
os.makedirs(OUT, exist_ok=True)
S = [64, 128, 256]


def copies_ok(BM, BN, BK, NT):
    return (BM * BK % (4 * NT) == 0 and BK * BN % (4 * NT) == 0 and BM * BK >= 4 * NT and BK * BN >= 4 * NT
            and NT % (BK // 4) == 0 and NT % (BN // 4) == 0)


configs = []
if KERNEL in (6, 7, 8):
    bufs = 2 if KERNEL == 8 else 1
    for BM, BN, TM, TN, BK in itertools.product(S, S, [4, 8], [4, 8], [8, 16, 32]):
        NT = (BM // TM) * (BN // TN)
        if not 64 <= NT <= 1024 or not copies_ok(BM, BN, BK, NT) or bufs * (BM + BN) * BK * 4 > 49152:
            continue
        d = dict(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN)
        bounds = [None] if KERNEL != 8 else [None, f"__launch_bounds__({NT}, {max(1, 1024 // NT)})", f"__launch_bounds__({NT}, 2)"]
        for lb in dict.fromkeys(bounds):
            configs.append((d, lb))
else:
    for BM, BN, BK, WM, WN, WMI, WNI in itertools.product(S, S, [8, 16, 32], [16, 32, 64, 128], [16, 32, 64, 128], [1, 2, 4], [1, 2, 4]):
        TM = TN = 4
        if BM % WM or BN % WN or WM % WMI or WN % WNI:
            continue
        SM, SN = WM // WMI, WN // WNI
        if SM % TM or SN % TN or (SM // TM) * (SN // TN) != 32 or WMI * WNI not in (2, 4, 8):
            continue
        NT = 32 * (BM // WM) * (BN // WN)
        if not 64 <= NT <= 1024 or not copies_ok(BM, BN, BK, NT) or 2 * (BM + BN) * BK * 4 > 49152:
            continue
        d = dict(BM=BM, BN=BN, BK=BK, WM=WM, WN=WN, TM=TM, TN=TN, WMITER=WMI, WNITER=WNI)
        for lb in (None, f"__launch_bounds__({NT}, 2)"):
            configs.append((d, lb))

src = open(f"cuda/sgemm_{KERNEL}.cu").read()
src = re.sub(r"__launch_bounds__\(NT, 2\) ", "", src)


def name(d, lb):
    return "_".join(f"{k}{v}" for k, v in d.items()) + ("_lb" + lb.split(",")[1].strip(" )") if lb else "")


def build(cfg):
    d, lb = cfg
    s = src
    for k, v in d.items():
        s = re.sub(rf"^#define {k} .*$", f"#define {k} {v}", s, count=1, flags=re.M)
    if lb:
        s = s.replace(f"__global__ void sgemm_{KERNEL}(", f"__global__ void {lb} sgemm_{KERNEL}(")
    n = name(d, lb)
    path = f"/tmp/sw/{n}.cu"
    with open(path, "w") as f:
        f.write(s)
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_100", "-Xptxas", "-v", "-I", os.path.abspath("cuda"), path, "-o", f"/tmp/sw/{n}"],
                       capture_output=True, text=True)
    regs = re.search(r"Used (\d+) registers", r.stderr)
    spill = re.search(r"(\d+) bytes spill stores", r.stderr)
    return cfg, r.returncode == 0, regs.group(1) if regs else "?", spill.group(1) if spill else "?", r.stderr[-300:]


os.makedirs("/tmp/sw", exist_ok=True)
print(f"kernel {KERNEL}: {len(configs)} configs", flush=True)
with ThreadPoolExecutor(os.cpu_count()) as ex:
    built = list(ex.map(build, configs))

rows = []
for (d, lb), ok, regs, spill, err in built:
    n = name(d, lb)
    if not ok:
        print("BUILD FAIL", n, err.replace("\n", " ")[-200:], flush=True)
        continue
    r = subprocess.run([f"/tmp/sw/{n}"], capture_output=True, text=True)
    t = re.search(r"average time: ([\d.]+)", r.stdout)
    w = re.search(r"spot check: (\d+) of", r.stdout)
    ms = float(t.group(1)) if t else float("nan")
    rows.append((ms, n, regs, spill, w.group(1) if w else "?"))
    print(f"{ms:.3f} ms  {n}  regs {regs} spill {spill} wrong {w.group(1) if w else '?'}", flush=True)

rows.sort()
with open(f"{OUT}/results.txt", "w") as f:
    for row in rows:
        f.write(f"{row[0]:.3f} {row[1]} regs={row[2]} spill={row[3]} wrong={row[4]}\n")
print("== best 15")
for row in rows[:15]:
    print(f"{row[0]:.3f} {row[1]} regs={row[2]} spill={row[3]} wrong={row[4]}")
