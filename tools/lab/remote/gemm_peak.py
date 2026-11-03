"""Achieved dense GEMM throughput on the local GPU: one JSON line per case.

Checks a GPU's real bf16/fp16 tensor-core ceiling (and its SM clock while
running), e.g. when prefill timings imply more FLOP/s than the datasheet.

  python gemm_peak.py [--iters 50]
"""

import argparse
import json
import subprocess
import time

import torch

# (m, n, k): a square case and Qwen2.5-7B's prefill GEMMs at 4096 tokens
# (fused QKV, attention out, fused gate/up, down).
CASES = [
    (8192, 8192, 8192),
    (4096, 4608, 3584),
    (4096, 3584, 3584),
    (4096, 37888, 3584),
    (4096, 3584, 18944),
]


def sm_clock_mhz() -> int | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return int(out.splitlines()[0]) if out else None


def bench(m: int, n: int, k: int, dtype: torch.dtype, iters: int) -> dict:
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    clock = sm_clock_mhz()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / iters
    return {
        "record": "lab.gemm_peak.v1",
        "dtype": str(dtype).removeprefix("torch."),
        "m": m, "n": n, "k": k,
        "ms": round(seconds * 1e3, 4),
        "tflops": round(2 * m * n * k / seconds / 1e12, 2),
        "sm_clock_mhz": clock,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    meta = {
        "record": "lab.gemm_peak_meta.v1",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    }
    print(json.dumps(meta), flush=True)
    for dtype in (torch.bfloat16, torch.float16):
        for m, n, k in CASES:
            print(json.dumps(bench(m, n, k, dtype, args.iters)), flush=True)


if __name__ == "__main__":
    main()
