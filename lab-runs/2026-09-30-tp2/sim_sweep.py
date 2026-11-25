"""Run the inference-sim release binary over a static-batch grid (Qwen2.5-7B, lab cluster).

Emits JSON lines: {"batch", "prompt", "decode", "tp", "pp", "phase", "sim_ms"}.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL = """[model]
id = "qwen2.5-7b-instruct"
layers = 28
hidden_size = 3584
attention_heads = 28
kv_heads = 4
vocab_size = 152064
parameters_gb = 15.23
dtype = "bf16"
"""


def workload(batch, prompt, decode, phase, tp, pp, calibration):
    return f"""schema_version = 1
{MODEL}
{calibration}
[request]
batch_size = {batch}
prompt_tokens = {prompt}
decode_tokens = {decode}
max_sequence_tokens = {prompt + decode}
phase = "{phase}"

[search]
tensor_ranks = [{tp}]
pipeline_ranks = [{pp}]
expert_ranks = [1]
data_ranks = [1]
"""


def simulate(binary, cluster, text):
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, dir=Path(__file__).parent) as handle:
        handle.write(text)
        path = handle.name
    completed = subprocess.run(
        [str(binary), "--cluster", str(cluster), "--workload", path, "--json", "--top-k", "1"],
        capture_output=True, text=True, check=False,
    )
    Path(path).unlink(missing_ok=True)
    if completed.returncode != 0:
        raise RuntimeError(f"simulator failed: {completed.stderr[-2000:]}")
    result = json.loads(completed.stdout)["results"][0]
    if not result["feasible"]:
        raise RuntimeError(f"infeasible: {result.get('rejected_reason')}")
    return float(result["estimated_latency_ms"]), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cluster", type=Path, required=True)
    parser.add_argument("--shapes", default="1x512,1x2048,8x512,8x2048,32x512")
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--compute-efficiency", type=float, default=None)
    parser.add_argument("--decode-memory-bandwidth-scale", type=float, default=None)
    parser.add_argument("--dump-ops", action="store_true")
    args = parser.parse_args()

    lines = []
    if args.compute_efficiency is not None:
        lines.append(f"compute_efficiency = {args.compute_efficiency}")
    if args.decode_memory_bandwidth_scale is not None:
        lines.append(f"decode_memory_bandwidth_scale = {args.decode_memory_bandwidth_scale}")
    calibration = "[calibration]\n" + "\n".join(lines) + "\n" if lines else ""

    for shape in args.shapes.split(","):
        batch, prompt = (int(v) for v in shape.split("x"))
        for phase, decode in (("prefill", 1), ("decode", args.decode), ("end_to_end", args.decode)):
            sim_ms, result = simulate(args.binary, args.cluster, workload(batch, prompt, decode, phase, args.tp, args.pp, calibration))
            row = {"batch": batch, "prompt": prompt, "decode": decode, "tp": args.tp, "pp": args.pp,
                   "phase": phase, "sim_ms": round(sim_ms, 3)}
            if args.dump_ops:
                row["bottlenecks"] = result.get("bottlenecks", [])[:4]
            print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
