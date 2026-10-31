"""Run commands from this folder on a Modal GPU. Invoke from the repo root.

The default GPU is a cheap L4, every run has a timeout (default 10 minutes), and any run whose
worst-case cost (running until the timeout) exceeds $2 asks for confirmation.

    uv run modal run modal_run.py --cmd "nvidia-smi && nvcc --version"
    uv run modal run modal_run.py --gpu H100 --cmd "..."

Inside the container this folder is the working directory (/root) and the runs volume is mounted at
/root/runs. Write plots, profiles and logs under runs/ to keep them.

    uv run modal volume ls inference-engineering-runs
    uv run modal volume get inference-engineering-runs <remote_path> <local_path>
"""

import subprocess

import modal

REMOTE_ROOT = "/root"
DEFAULT_TIMEOUT_SECONDS = 10 * 60
CONFIRM_ABOVE_USD = 2.0

CUDA_VERSION = "12.8.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

app = modal.App("inference-engineering")
runs_volume = modal.Volume.from_name("inference-engineering-runs", create_if_missing=True, version=2)
VOLUMES = {f"{REMOTE_ROOT}/runs": runs_volume}

IGNORE = [".venv", ".git", "runs", "build", "**/__pycache__", ".ruff_cache", "*.nsys-rep", "*.ncu-rep", ".DS_Store"]

# The devel image ships nvcc and the CUDA headers, which plain Python images don't have.
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("cuda-nsight-systems-12-8", "cuda-nsight-compute-12-8")
    .pip_install("torch==2.11.0", index_url=TORCH_INDEX)
    .pip_install("numpy", "matplotlib", "pandas", "ninja")
    .env({"PYTHONUNBUFFERED": "1", "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
    .workdir(REMOTE_ROOT)
    .add_local_dir(".", REMOTE_ROOT, ignore=IGNORE)
)


@app.function(image=image, volumes=VOLUMES, timeout=DEFAULT_TIMEOUT_SECONDS)
def run(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=REMOTE_ROOT)
    finally:
        runs_volume.commit()


def worst_case_cost(gpu_spec: str | None, hours: float) -> float | None:
    """Cost if the run lasts until its timeout (GPU + one CPU). None if the GPU rate is unknown."""
    rates = dict(modal.Workspace.from_context().billing.rates())
    per_hour = float(rates["cpu_hour_cost"])
    if gpu_spec:
        name, _, count = gpu_spec.partition(":")
        name = name.lower().replace("-", "_")
        key = f"gpu_hour_cost_{'a100_40gb' if name == 'a100' else name}"
        if key not in rates:
            return None
        per_hour += float(rates[key]) * int(count or 1)
    return per_hour * hours


@app.local_entrypoint()
def main(cmd: str, gpu: str = "L4", timeout: int = DEFAULT_TIMEOUT_SECONDS, yes: bool = False) -> None:
    gpu_spec = None if gpu.lower() in ("", "none", "cpu") else gpu
    cost = worst_case_cost(gpu_spec, timeout / 3600)
    month = float(modal.Workspace.from_context().billing.summary().billed_cost)
    print(
        f"Modal run: gpu={gpu_spec or 'none'} timeout={timeout}s | "
        f"worst case {'unknown' if cost is None else f'~${cost:.2f}'} | billed this month ${month:.2f}",
        flush=True,
    )
    if (cost is None or cost > CONFIRM_ABOVE_USD) and not yes:
        try:
            answer = input("Launch? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            raise SystemExit("Aborted, nothing was launched.")
    run.with_options(gpu=gpu_spec, timeout=timeout).remote(cmd)
