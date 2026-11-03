"""Decode batch sweep (ledger 19): KV-exhaustion preemption on the iteration engine.

Replays lab-runs/2026-10-01-qwen7b-decode-batch-sweep{,-tail} (Qwen2.5-7B on
node1, vLLM 0.29.0, static `LLM.generate` batches of B x 512-token prompts,
max_num_batched_tokens 32768, max_num_seqs 64, prefix caching off) on the
serving iteration engine: B requests at t = 0, one run with 1 output token
(prefill_ms) and one with 1 + 128 (full), decode_ms = full - prefill, exactly
as tools/lab/remote/bench_latency.py measures it. The run's makespan is the
largest end-to-end latency.

Three engine configurations:

* `uncapped`  - no KV limit (what the static solver validations assume);
* `reserved`  - vLLM's logged KV capacity (24,896 tokens = 1,556 blocks),
                whole footprints reserved at admission (the old engine);
* `preempt`   - the same capacity with decode_capacity_policy = "preempt"
                (vLLM V1 paged allocation and recompute preemption).

Calibration: the curve-only profile scalars (recalibrated static constants +
token-dependent efficiency curve), no frontend latency (offline LLM.generate).
Run from the repo root after `cargo build --release`.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
BINARY = "target/release/inference-sim"
CLUSTER = "tools/lab/labs/rtx3090_cluster.toml"
MEASURED = [
    Path("lab-runs/2026-10-01-qwen7b-decode-batch-sweep/measured.jsonl"),
    Path("lab-runs/2026-10-01-qwen7b-decode-batch-sweep-tail/measured.jsonl"),
]
KV_TOKENS = 24_896
BLOCK = 16
PROMPT = 512
DECODE = 128
NODE = 1
MODES = ("uncapped", "reserved", "preempt")

CALIBRATION = """[calibration]
compute_efficiency_curve = [[128, 0.653909], [256, 0.618297], [512, 0.724381], [1024, 0.83777], [2048, 0.846627], [4096, 0.873915]]
compute_efficiency = 0.849357
decode_memory_bandwidth_scale = 0.838309
"""


def workload(batch: int, output: int, mode: str) -> str:
    kv = {
        "uncapped": "",
        "reserved": f"max_resident_tokens = {KV_TOKENS}\nmax_kv_blocks = {KV_TOKENS // BLOCK}\n",
        "preempt": f"max_resident_tokens = {KV_TOKENS}\nmax_kv_blocks = {KV_TOKENS // BLOCK}\n"
        'decode_capacity_policy = "preempt"\n',
    }[mode]
    return f"""schema_version = 1

[model]
id = "qwen2.5-7b-instruct"
layers = 28
hidden_size = 3584
attention_heads = 28
kv_heads = 4
vocab_size = 152064
ffn_hidden_size = 18944
parameters_gb = 15.23
dtype = "bf16"

[request]
batch_size = 1
prompt_tokens = {PROMPT}
decode_tokens = {output}
max_sequence_tokens = {PROMPT + output}
phase = "end_to_end"

{CALIBRATION}
[search]
tensor_ranks = [1]
pipeline_ranks = [1]
expert_ranks = [1]
data_ranks = [1]

[serving]
mode = "colocated"
serving_stack = "vllm"
objective = "e2el"
prefill_nodes = [{NODE}]
decode_nodes = [{NODE}]

[serving.traffic]
request_count = {batch}
arrival = "fixed"
arrival_gap_ms = 0.0
shape_seed = 0
prefill_batching = "continuous"
max_prefill_batch_tokens = 32768
max_prefill_chunk_tokens = 32768
decode_batching = "continuous"
max_decode_batch_tokens = 64
max_decode_sequences = 64
kv_block_tokens = {BLOCK}
{kv}prefix_cache_hit_rate = 0.0
batch_sizes = [1]
prompt_tokens = [{PROMPT}]
decode_tokens = [{output}]

[serving.prefill_search]
tensor_ranks = [1]
pipeline_ranks = [1]
expert_ranks = [1]
data_ranks = [1]

[serving.decode_search]
tensor_ranks = [1]
pipeline_ranks = [1]
expert_ranks = [1]
data_ranks = [1]
"""


def simulate(batch: int, output: int, mode: str) -> dict:
    path = HERE / f"workload.{mode}.{batch}.{output}.tmp.toml"
    path.write_text(workload(batch, output, mode))
    try:
        out = subprocess.run(
            [BINARY, "--cluster", CLUSTER, "--workload", str(path), "--json", "--top-k", "1"],
            capture_output=True, text=True, check=True,
        ).stdout
    finally:
        path.unlink()
    result = json.loads(out)["results"][0]
    metrics = result["metrics"]
    return {
        "feasible": result["feasible"],
        "makespan_ms": metrics["e2el_max_ms"],
        "completed": metrics["completed_requests"],
        "preemptions": metrics["preemptions"],
        "preempted_requests": metrics["preempted_requests"],
        "recompute_tokens": metrics["preemption_recompute_tokens"],
    }


def measured() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for path in MEASURED:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("record") != "lab.static_batch.v1":
                continue
            # The tail run repeats batch 32-40; keep the tail's (later) value.
            rows[record["batch"]] = record
    return dict(sorted(rows.items()))


def main() -> int:
    real = measured()
    out = {"kv_tokens": KV_TOKENS, "rows": []}
    header = f"{'batch':>5} {'real ms/step':>12} " + " ".join(f"{m + ' ms/step':>17}" for m in MODES)
    print(header)
    for batch, record in real.items():
        row = {"batch": batch, "measured": {k: record[k] for k in ("prefill_ms", "decode_ms", "decode_ms_per_step")}}
        for mode in MODES:
            prefill = simulate(batch, 1, mode)
            full = simulate(batch, 1 + DECODE, mode)
            decode_ms = full["makespan_ms"] - prefill["makespan_ms"]
            row[mode] = {
                "feasible": prefill["feasible"] and full["feasible"],
                "prefill_ms": round(prefill["makespan_ms"], 3),
                "decode_ms": round(decode_ms, 3),
                "decode_ms_per_step": round(decode_ms / DECODE, 4),
                "error_pct": round(100.0 * (decode_ms / DECODE / record["decode_ms_per_step"] - 1.0), 2),
                "preemptions": full["preemptions"],
                "preempted_requests": full["preempted_requests"],
                "recompute_tokens": full["recompute_tokens"],
                "prefill_preemptions": prefill["preemptions"],
            }
        out["rows"].append(row)
        cells = " ".join(
            f"{row[m]['decode_ms_per_step']:7.2f} ({row[m]['error_pct']:+6.1f}%)" + ("" if row[m]["feasible"] else "!")
            for m in MODES
        )
        print(f"{batch:>5} {record['decode_ms_per_step']:>12.2f} {cells}  preempt={row['preempt']['preemptions']}")
    (HERE / "predictions.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
