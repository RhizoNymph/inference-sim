"""Blind predictions for the TP=2 NIC-throttle test (made before the run is read).

Lab runs: tools/lab/specs/rtx3090_qwen7b_static_tp2.toml (unthrottled, same
day) and rtx3090_qwen7b_static_tp2_nic2g.toml (tbf caps bond0 egress at
2 Gbit/s on node0 and node1, via tools/lab/remote/run_with_nic_throttle.sh).

Simulator: examples/rtx3090_lab_cluster_measured_curves.toml (node0 egress
3.61 Gb/s, node1 9.41 Gb/s, measured 09-30 curves) and the entry-14
token-dependent efficiency calibration. Three predictions per shape and phase:

- "curves": the cluster as is (unthrottled);
- "alpha-beta": the same cluster without [[collective_curves]] (unthrottled);
- "throttled": a run-scenario NIC degradation overlay scaling node0's NIC by
  2/3.61 and node1's by 2/9.41, so both egress directions are 2 Gb/s. The
  overlay suspends the curves, so this is alpha-beta on the degraded NICs.

The test is whether the throttled/unthrottled ratio matches the measured one.
Writes predictions.json next to this file. Run from the repo root after
`cargo build --release`:

  python3 lab-runs/2026-11-02-tp2-nic-throttle-sim/predict.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BINARY = REPO / "target" / "release" / "inference-sim"
CLUSTER = REPO / "examples" / "rtx3090_lab_cluster_measured_curves.toml"
SHAPES = [(1, 512), (1, 2048), (8, 512), (32, 512)]
DECODE = 128
THROTTLE_GBPS = 2.0
NODE0_EGRESS_GBPS = 3.61
NODE1_LINE_GBPS = 9.41

MODEL = """[model]
id = "qwen2.5-7b-instruct"
layers = 28
hidden_size = 3584
attention_heads = 28
kv_heads = 4
ffn_hidden_size = 18944
vocab_size = 152064
parameters_gb = 15.23
dtype = "bf16"
"""
CALIBRATION = """[calibration]
compute_efficiency_curve = [[128, 0.653909], [256, 0.618297], [512, 0.724381], [1024, 0.83777], [2048, 0.846627], [4096, 0.873915]]
compute_efficiency = 0.849357
decode_memory_bandwidth_scale = 0.838309
"""


def workload(batch: int, prompt: int, decode: int, phase: str) -> str:
    return f"""schema_version = 1
{MODEL}
{CALIBRATION}
[request]
batch_size = {batch}
prompt_tokens = {prompt}
decode_tokens = {decode}
max_sequence_tokens = {prompt + DECODE}
phase = "{phase}"

[search]
tensor_ranks = [2]
pipeline_ranks = [1]
expert_ranks = [1]
data_ranks = [1]
"""


RUN = f"""schema_version = 1
cluster = "cluster.tmp.toml"
workload = "workload.tmp.toml"

[output]
format = "json"
top_k = 1

[[scenarios]]
name = "unthrottled"

[[scenarios]]
name = "throttled"

[[scenarios.topology.degraded_nics]]
node = 0
nics = [0]
bandwidth_scale = {THROTTLE_GBPS / NODE0_EGRESS_GBPS:.6f}

[[scenarios.topology.degraded_nics]]
node = 1
nics = [0]
bandwidth_scale = {THROTTLE_GBPS / NODE1_LINE_GBPS:.6f}
"""


def without_curves(text: str) -> str:
    head, _, _ = text.partition("\n[[collective_curves]]")
    return head


def scenario_latencies(cluster_text: str, workload_text: str) -> dict[str, tuple[float, list[str]]]:
    (HERE / "cluster.tmp.toml").write_text(cluster_text)
    (HERE / "workload.tmp.toml").write_text(workload_text)
    run = HERE / "run.tmp.toml"
    run.write_text(RUN)
    try:
        completed = subprocess.run([str(BINARY), "--run", str(run)], capture_output=True, text=True, check=True)
    finally:
        for name in ("cluster.tmp.toml", "workload.tmp.toml", "run.tmp.toml"):
            (HERE / name).unlink(missing_ok=True)
    out = {}
    for scenario in json.loads(completed.stdout)["scenarios"]:
        result = scenario["result"]["results"][0]
        if not result["feasible"]:
            raise SystemExit(f"infeasible: {scenario['name']}")
        codes = sorted({a["code"] for a in result.get("approximations", [])})
        out[scenario["name"]] = (float(result["estimated_latency_ms"]), codes)
    return out


def main() -> int:
    curves_text = CLUSTER.read_text()
    clusters = {"curves": curves_text, "alpha-beta": without_curves(curves_text)}
    predictions: dict[str, dict] = {}
    codes: dict[str, set[str]] = {}
    for batch, prompt in SHAPES:
        shape = f"{batch}x{prompt}"
        row: dict[str, dict[str, float]] = {}
        for phase, decode in (("prefill", 1), ("decode", DECODE), ("end_to_end", DECODE)):
            for label, text in clusters.items():
                for scenario, (ms, used) in scenario_latencies(text, workload(batch, prompt, decode, phase)).items():
                    if scenario == "throttled" and label != "alpha-beta":
                        continue  # the overlay suspends curves: identical to alpha-beta
                    key = label if scenario == "unthrottled" else "throttled"
                    row.setdefault(key, {})[phase] = round(ms, 3)
                    codes.setdefault(key, set()).update(used)
        for values in row.values():
            values["decode_step"] = round(values.pop("decode") / DECODE, 4)
        predictions[shape] = row
        t, u = row["throttled"], row["alpha-beta"]
        print(f"{shape:>7}: prefill {row['curves']['prefill']:8.1f} / {u['prefill']:8.1f} -> {t['prefill']:8.1f} ms | "
              f"step {row['curves']['decode_step']:6.2f} / {u['decode_step']:6.2f} -> {t['decode_step']:6.2f} ms "
              f"(curves / alpha-beta -> throttled)")
    out = {"predictions": predictions, "approximation_codes": {k: sorted(v) for k, v in codes.items()}}
    (HERE / "predictions.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
