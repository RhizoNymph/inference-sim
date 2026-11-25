"""N-rank NCCL collective and point-to-point sweep over doubling message sizes.

Run one process per GPU (the lab harness launches them with MASTER_ADDR,
MASTER_PORT, RANK, WORLD_SIZE, and LOCAL_RANK set). Rank 0 writes JSON lines
to --output (default stdout):

* one `lab.collective_meta.v1` line: world size, dtype, and per rank its
  hostname, local GPU, and simulator node id (--sim-node-id or SIM_NODE_ID);
* one `lab.collective.v1` line per (op, message size): median/p10/p90 in us;
* a final `lab.done.v1` line.

Timing. Every rank times every iteration (barrier, synchronize, op,
synchronize). Collective rows use the per-iteration maximum across ranks
(`timed_on = "max_rank"`): a collective is done when its slowest rank is.
Point-to-point rows use the receiver's samples (`timed_on = "receiver"`):
NCCL send returns once the data is buffered, before it is delivered, so the
sender's clock under-reports small messages.

`bytes` is the tensor each rank contributes: the buffer for all_reduce and
send/recv, the per-rank input shard for all_gather, the per-rank output shard
for reduce_scatter, and the per-rank input buffer for all_to_all.

torch is imported inside main() so the helpers are importable (and tested)
without it.
"""

import argparse
import json
import os
import socket
import statistics
import sys
import time

COLLECTIVE_RECORD = "lab.collective.v1"
META_RECORD = "lab.collective_meta.v1"
DONE_RECORD = "lab.done.v1"
COLLECTIVE_OPS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
SEND_RECV = "send_recv"
ALL_OPS = (*COLLECTIVE_OPS, SEND_RECV)
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
LARGE_MESSAGE_BYTES = 16 * 1024 * 1024


def message_sizes(min_bytes, max_bytes):
    """min_bytes, 2*min_bytes, ... up to and including max_bytes."""
    if min_bytes < 1 or max_bytes < min_bytes:
        raise ValueError(f"need 1 <= min_bytes <= max_bytes, got {min_bytes}, {max_bytes}")
    sizes = []
    size = min_bytes
    while size <= max_bytes:
        sizes.append(size)
        size *= 2
    return sizes


def send_pairs(world_size):
    """Every ordered (src, dst) rank pair."""
    return [(src, dst) for src in range(world_size) for dst in range(world_size) if src != dst]


def send_op_name(src, dst):
    return f"send_{src}to{dst}"


def expand_ops(ops, world_size):
    """Op names in run order; `send_recv` becomes one op per ordered rank pair."""
    names = []
    for op in ops:
        if op == SEND_RECV:
            names += [send_op_name(src, dst) for src, dst in send_pairs(world_size)]
        elif op in COLLECTIVE_OPS:
            names.append(op)
        else:
            raise ValueError(f"unknown op {op!r}; known: {list(ALL_OPS)}")
    return names


def parse_ops(text):
    ops = [part.strip() for part in text.split(",") if part.strip()]
    if not ops:
        raise ValueError("--ops needs at least one op")
    for op in ops:
        if op not in ALL_OPS:
            raise ValueError(f"unknown op {op!r}; known: {list(ALL_OPS)}")
    if len(set(ops)) != len(ops):
        raise ValueError(f"duplicate ops in {text!r}")
    return ops


def elements_for(op, size_bytes, element_bytes, world_size):
    """Elements of the per-rank tensor; all_to_all needs a multiple of world_size."""
    elements = max(1, size_bytes // element_bytes)
    if op == "all_to_all":
        elements = max(world_size, elements - elements % world_size)
    return elements


def iters_for(size_bytes, iters):
    """Fewer iterations for large messages (they take 10s to 100s of ms each)."""
    return iters if size_bytes <= LARGE_MESSAGE_BYTES else max(8, iters // 3)


def summarize(samples_us):
    ordered = sorted(samples_us)
    if not ordered:
        raise ValueError("no samples")
    return {
        "median_us": round(statistics.median(ordered), 2),
        "p10_us": round(ordered[len(ordered) // 10], 2),
        "p90_us": round(ordered[(len(ordered) * 9) // 10], 2),
    }


def per_iteration_max(per_rank_samples):
    """Iteration i's time is the slowest rank's time for iteration i."""
    lengths = {len(samples) for samples in per_rank_samples}
    if len(lengths) != 1:
        raise ValueError(f"ranks recorded different iteration counts: {sorted(lengths)}")
    return [max(column) for column in zip(*per_rank_samples, strict=True)]


def parse_send_op(op):
    """`send_1to0` -> (1, 0); None for collectives."""
    if not op.startswith("send_") or "to" not in op:
        return None
    src, _, dst = op[len("send_") :].partition("to")
    return int(src), int(dst)


def result_row(op, size_bytes, dtype, world_size, per_rank_samples, iters):
    """One `lab.collective.v1` record from every rank's samples for this op and size."""
    pair = parse_send_op(op)
    if pair is None:
        samples = per_iteration_max(per_rank_samples)
        timing = {"timed_on": "max_rank"}
    else:
        src, dst = pair
        samples = per_rank_samples[dst]
        timing = {"timed_on": "receiver", "src_rank": src, "dst_rank": dst}
    return {
        "record": COLLECTIVE_RECORD,
        "op": op,
        "bytes": size_bytes,
        "dtype": dtype,
        "world_size": world_size,
        **timing,
        **summarize(samples),
        "iters": iters,
    }


def meta_record(world_size, dtype, ranks):
    return {
        "record": META_RECORD,
        "world_size": world_size,
        "dtype": dtype,
        "ranks": sorted(ranks, key=lambda info: info["rank"]),
    }


def resolve_rank_world(args, env):
    """--rank/--world-size win over RANK/WORLD_SIZE from the environment."""
    rank = args.rank if args.rank is not None else env.get("RANK")
    world = args.world_size if args.world_size is not None else env.get("WORLD_SIZE")
    if rank is None or world is None:
        raise ValueError("set --rank/--world-size or RANK/WORLD_SIZE")
    rank, world = int(rank), int(world)
    if world < 2:
        raise ValueError(f"world size must be >= 2, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} out of range for world size {world}")
    return rank, world


def resolve_sim_node_id(args, env):
    value = args.sim_node_id if args.sim_node_id is not None else env.get("SIM_NODE_ID")
    return None if value is None else int(value)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world-size", type=int, help="default: $WORLD_SIZE")
    parser.add_argument("--rank", type=int, help="default: $RANK")
    parser.add_argument("--local-rank", type=int, help="GPU index on this node (default: $LOCAL_RANK or 0)")
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--ops", default=",".join(ALL_OPS), help=f"comma-separated subset of {ALL_OPS}")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPE_BYTES))
    parser.add_argument("--sim-node-id", type=int, help="simulator node id (default: $SIM_NODE_ID)")
    parser.add_argument("--output", help="JSON lines file written by rank 0 (default: stdout)")
    return parser


def main(argv=None):
    import torch
    import torch.distributed as dist

    args = build_parser().parse_args(argv)
    rank, world = resolve_rank_world(args, os.environ)
    local_rank = args.local_rank if args.local_rank is not None else int(os.environ.get("LOCAL_RANK", "0"))
    ops = expand_ops(parse_ops(args.ops), world)
    sizes = message_sizes(args.min_bytes, args.max_bytes)
    dtype = getattr(torch, args.dtype)
    element_bytes = DTYPE_BYTES[args.dtype]

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    def gather(value):
        gathered = [None] * world
        dist.all_gather_object(gathered, value)
        return gathered

    out = None
    if rank == 0:
        out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout  # noqa: SIM115

    def emit(record):
        if out is not None:
            out.write(json.dumps(record) + "\n")
            out.flush()

    info = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "local_rank": local_rank,
        "sim_node_id": resolve_sim_node_id(args, os.environ),
        "device": torch.cuda.get_device_name(local_rank),
    }
    emit(meta_record(world, args.dtype, gather(info)))

    def make_case(op, elements):
        """(callable, per-rank bytes) for one op at one size; tensors live in the closure."""
        if op == "all_reduce":
            buffer = torch.ones(elements, dtype=dtype, device="cuda")
            return (lambda: dist.all_reduce(buffer)), elements * element_bytes
        if op == "all_gather":
            shard = torch.ones(elements, dtype=dtype, device="cuda")
            gathered = torch.empty(elements * world, dtype=dtype, device="cuda")
            return (lambda: dist.all_gather_into_tensor(gathered, shard)), elements * element_bytes
        if op == "reduce_scatter":
            full = torch.ones(elements * world, dtype=dtype, device="cuda")
            shard = torch.empty(elements, dtype=dtype, device="cuda")
            return (lambda: dist.reduce_scatter_tensor(shard, full)), elements * element_bytes
        if op == "all_to_all":
            source = torch.ones(elements, dtype=dtype, device="cuda")
            target = torch.empty(elements, dtype=dtype, device="cuda")
            return (lambda: dist.all_to_all_single(target, source)), elements * element_bytes
        src, dst = parse_send_op(op)
        buffer = torch.ones(elements, dtype=dtype, device="cuda")

        def send_recv():
            if rank == src:
                dist.send(buffer, dst)
            elif rank == dst:
                dist.recv(buffer, src)

        return send_recv, elements * element_bytes

    def timed(fn, iters):
        for _ in range(args.warmup):
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

    rows = 0
    for size in sizes:
        iters = iters_for(size, args.iters)
        for op in ops:
            fn, size_bytes = make_case(op, elements_for(op, size, element_bytes, world))
            per_rank = gather(timed(fn, iters))
            emit(result_row(op, size_bytes, args.dtype, world, per_rank, iters))
            rows += 1
            del fn
        torch.cuda.empty_cache()

    emit({"record": DONE_RECORD, "rows": rows})
    dist.barrier()
    dist.destroy_process_group()
    if out is not None and out is not sys.stdout:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
