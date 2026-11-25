"""Two-rank NCCL collective sweep over message sizes (bf16), run once per node.

Rank 0 prints JSON lines: {"op", "bytes", "median_us", "p10_us", "p90_us", "iters"}.
ops: all_reduce (both ranks), send_0to1 and send_1to0 (one-way point-to-point).
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    dist.init_process_group("nccl", rank=args.rank, world_size=2)
    torch.cuda.set_device(0)
    size = args.min_bytes
    while size <= args.max_bytes:
        elements = max(1, size // 2)
        tensor = torch.ones(elements, dtype=torch.bfloat16, device="cuda")
        iters = args.iters if size <= 16 * 1024 * 1024 else max(8, args.iters // 3)

        cases = {"all_reduce": lambda: dist.all_reduce(tensor)}
        for src, dst in ((0, 1), (1, 0)):
            def send_recv(src=src, dst=dst):
                if args.rank == src:
                    dist.send(tensor, dst)
                else:
                    dist.recv(tensor, src)
            cases[f"send_{src}to{dst}"] = send_recv

        for op, fn in cases.items():
            samples = sorted(timed(fn, iters, args.warmup))
            if args.rank == 0:
                print(json.dumps({
                    "op": op, "bytes": elements * 2,
                    "median_us": round(statistics.median(samples), 2),
                    "p10_us": round(samples[len(samples) // 10], 2),
                    "p90_us": round(samples[(len(samples) * 9) // 10], 2),
                    "iters": iters,
                }), flush=True)
        size *= 2
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
