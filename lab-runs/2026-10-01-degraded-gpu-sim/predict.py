"""Re-predict the RTX 3090 clock-locked (1200 MHz) static runs with the
degraded-GPU clock model (validation ledger 22).

Measured data (nothing new was measured for this entry):

* full clock: lab-runs/2026-09-28-static-batch/real_pp1.jsonl (node0, the
  baseline the recalibrated profile was fitted on and ledger 17 compared to);
* locked 1200 MHz: lab-runs/2026-10-01-qwen7b-static-node2-clock1200/measured.jsonl
  (node2, `nvidia-smi -lgc 1200,1200`, memory clock untouched).

Calibration: the recalibrated scalar profile
(lab-runs/2026-09-28-static-batch/calibration_profile-pp1-recalibrated.toml,
compute_efficiency 0.8494, decode_memory_bandwidth_scale 0.8383, 88 TFLOPs
peak), as in ledger 17. The curve profile is reported as a secondary variant.

The clock model has two clocks, and with one degraded clock BOTH are fitted
from the 1200 MHz run (measured/measured ratios, never simulator output):

* sustained_sm_clock_mhz = 1200 x median over shapes of
  prefill(1200) / prefill(full)   (compute-bound prefill scales with clock);
* bandwidth_knee_sm_clock_mhz = 1200 x median over shapes of
  decode_step(1200) / decode_step(full)   (memory-bound decode).

So the in-sample variant is a fit, not a prediction. The leave-one-shape-out
variant fits both clocks on four shapes and predicts the fifth: it tests that
one compute factor and one bandwidth factor explain every shape and phase, NOT
that the model extrapolates to other clocks (two clocks cannot test that).

Variants:

* naive:        ledger 17's cluster (peak_f16_tflops scaled 88 -> 50.29 by
                1200/2100, bandwidth unchanged), re-run on this binary;
* compute-only: sustained clock fitted, no bandwidth knee (bandwidth assumed
                independent of SM clock);
* clock-fit:    sustained clock and knee fitted on all five shapes (in-sample);
* clock-loo:    both clocks fitted on four shapes, fifth predicted;
* clock-fit-curve: clock-fit clocks with the efficiency-curve profile.

Run from the repo root after `cargo build --release`:
    python3 lab-runs/2026-10-01-degraded-gpu-sim/predict.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BINARY = REPO / "target" / "release" / "inference-sim"
FULL = REPO / "lab-runs/2026-09-28-static-batch/real_pp1.jsonl"
LOCKED = REPO / "lab-runs/2026-10-01-qwen7b-static-node2-clock1200/measured.jsonl"
SPEC = REPO / "tools/lab/specs/rtx3090_qwen7b_static_node2_clock1200.toml"
SCALAR_PROFILE = REPO / "lab-runs/2026-09-28-static-batch/calibration_profile-pp1-recalibrated.toml"
CURVE_PROFILE = REPO / "lab-runs/2026-10-01-qwen7b-prefill-token-sweep/calibration_profile-curve-only.toml"
NAIVE_CLUSTER = REPO / "lab-runs/2026-10-01-degraded-sim/rtx3090_cluster_clock1200.toml"
LOCKED_MHZ = 1200.0
# Generated workload/cluster TOMLs go to a throwaway directory.
WORK = Path(tempfile.mkdtemp(prefix="degraded-gpu-sim-"))

Shape = tuple[int, int]


def load_records(path: Path) -> dict[Shape, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "prefill_ms" in record:
            rows[(record["batch"], record["prompt"])] = record
    return rows


def fit_clocks(full: dict[Shape, dict], locked: dict[Shape, dict], shapes: list[Shape]) -> tuple[float, float]:
    prefill = statistics.median(locked[s]["prefill_ms"] / full[s]["prefill_ms"] for s in shapes)
    decode = statistics.median(locked[s]["decode_ms_per_step"] / full[s]["decode_ms_per_step"] for s in shapes)
    return LOCKED_MHZ * prefill, LOCKED_MHZ * decode


def cluster_text(sustained: float | None, knee: float | None, locked: float | None) -> str:
    clock = []
    if sustained is not None:
        clock.append(f"sustained_sm_clock_mhz = {sustained:.1f}")
    if knee is not None:
        clock.append(f"bandwidth_knee_sm_clock_mhz = {knee:.1f}")
    if locked is not None:
        clock.append(f"locked_sm_clock_mhz = {locked:.1f}")
    clock_lines = "\n".join(clock)
    return f"""schema_version = 1

# node2 (RTX 3090) alone: a100_40gb base overridden with 3090 specs, 88 TFLOPs
# at the 2100 MHz max clock (lab-runs/2026-09-30-recalibration), plus the
# degraded-GPU clock keys (docs/features/degraded_gpu_model.md).
[cluster]
preset = "custom"

[interconnect]
kind = "ethernet"
variant = "10g"
oversubscription = 1.0

[[nodes]]
id = 2
gpu = "a100_40gb"
gpu_count = 1
hbm_gb = 24.0
hbm_bandwidth_gb_s = 936.0
peak_f16_tflops = 88.0
intra = "pcie_gen4"
nics = {{ count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }}
{clock_lines}
"""


def workload_text(model: dict, shape: Shape, decode: int, phase: str, profile: Path) -> str:
    batch, prompt = shape
    keys = ["layers", "hidden_size", "attention_heads", "kv_heads", "vocab_size", "ffn_hidden_size", "parameters_gb"]
    model_lines = "\n".join(f"{key} = {model[key]}" for key in keys)
    return f"""schema_version = 1

[model]
id = "qwen2.5-7b-instruct"
{model_lines}
dtype = "{model["dtype"]}"

[request]
batch_size = {batch}
prompt_tokens = {prompt}
decode_tokens = {decode}
max_sequence_tokens = {prompt + decode}
phase = "{phase}"

[calibration_profile]
path = "{profile}"

[search]
tensor_ranks = [1]
pipeline_ranks = [1]
expert_ranks = [1]
data_ranks = [1]
"""


def simulate(cluster: Path, workload: Path) -> dict:
    out = subprocess.run(
        [str(BINARY), "--cluster", str(cluster), "--workload", str(workload), "--json", "--top-k", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)["results"][0]


def predict(name: str, cluster: str | Path, shapes: list[Shape], profile: Path) -> dict[Shape, dict]:
    """Simulated prefill / decode-step / end-to-end ms and clock evidence per shape."""
    spec = tomllib.loads(SPEC.read_text(encoding="utf-8"))
    model = spec["model"]["sim"]
    decode = spec["static_batch"]["decode_tokens"]
    WORK.mkdir(exist_ok=True)
    if isinstance(cluster, Path):
        cluster_path = cluster
    else:
        cluster_path = WORK / f"{name}-cluster.toml"
        cluster_path.write_text(cluster, encoding="utf-8")
    jobs = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for shape in shapes:
            for phase in ("prefill", "decode", "end_to_end"):
                workload = WORK / f"{name}-{shape[0]}x{shape[1]}-{phase}.toml"
                workload.write_text(workload_text(model, shape, decode, phase, profile), encoding="utf-8")
                jobs[(shape, phase)] = pool.submit(simulate, cluster_path, workload)
    rows = {}
    for shape in shapes:
        first = {phase: jobs[(shape, phase)].result() for phase in ("prefill", "decode", "end_to_end")}
        evidence = [
            a["message"]
            for a in first["end_to_end"].get("approximations", [])
            if a.get("code") == "gpu_clock_degradation"
        ]
        rows[shape] = {
            "prefill_ms": first["prefill"]["estimated_latency_ms"],
            "decode_ms_per_step": first["decode"]["estimated_latency_ms"] / decode,
            "end_to_end_ms": first["end_to_end"]["estimated_latency_ms"],
            "clock_evidence": evidence,
        }
    return rows


METRICS = ("prefill_ms", "decode_ms_per_step", "end_to_end_ms")


def errors(sim: dict[Shape, dict], measured: dict[Shape, dict]) -> dict:
    per_shape = {
        f"{b}x{p}": {m: 100.0 * (sim[(b, p)][m] - measured[(b, p)][m]) / measured[(b, p)][m] for m in METRICS}
        for (b, p) in sim
    }
    mean_abs = {m: statistics.fmean(abs(row[m]) for row in per_shape.values()) for m in METRICS}
    return {"per_shape_pct": per_shape, "mean_abs_pct": mean_abs}


def main() -> None:
    if not BINARY.is_file():
        sys.exit(f"build the release binary first: {BINARY}")
    full = load_records(FULL)
    locked = load_records(LOCKED)
    shapes = sorted(locked)
    sustained, knee = fit_clocks(full, locked, shapes)
    results: dict[str, object] = {
        "fitted_on_1200mhz_data": {"sustained_sm_clock_mhz": sustained, "bandwidth_knee_sm_clock_mhz": knee},
        "measured_slowdown": {
            f"{b}x{p}": {
                "prefill": locked[(b, p)]["prefill_ms"] / full[(b, p)]["prefill_ms"],
                "decode_step": locked[(b, p)]["decode_ms_per_step"] / full[(b, p)]["decode_ms_per_step"],
            }
            for (b, p) in shapes
        },
        "variants": {},
    }
    variants = results["variants"]
    assert isinstance(variants, dict)

    # Full-clock sanity check: a declared but unlocked clock changes nothing.
    nominal = predict("nominal", cluster_text(None, None, None), shapes, SCALAR_PROFILE)
    declared = predict("declared-unlocked", cluster_text(sustained, knee, None), shapes, SCALAR_PROFILE)
    results["unlocked_matches_no_clock"] = all(
        nominal[s][m] == declared[s][m] for s in shapes for m in METRICS
    )
    variants["full-clock baseline vs full-clock measured"] = errors(nominal, full)

    runs = {
        "naive": predict("naive", NAIVE_CLUSTER, shapes, SCALAR_PROFILE),
        "compute-only": predict("compute-only", cluster_text(sustained, None, LOCKED_MHZ), shapes, SCALAR_PROFILE),
        "clock-fit": predict("clock-fit", cluster_text(sustained, knee, LOCKED_MHZ), shapes, SCALAR_PROFILE),
    }
    loo: dict[Shape, dict] = {}
    loo_clocks = {}
    for held in shapes:
        rest = [s for s in shapes if s != held]
        s_mhz, k_mhz = fit_clocks(full, locked, rest)
        loo_clocks[f"{held[0]}x{held[1]}"] = {"sustained_sm_clock_mhz": s_mhz, "bandwidth_knee_sm_clock_mhz": k_mhz}
        loo[held] = predict(
            f"loo-{held[0]}x{held[1]}", cluster_text(s_mhz, k_mhz, LOCKED_MHZ), [held], SCALAR_PROFILE
        )[held]
    runs["clock-loo"] = loo
    runs["clock-fit-curve"] = predict(
        "clock-fit-curve", cluster_text(sustained, knee, LOCKED_MHZ), shapes, CURVE_PROFILE
    )
    results["loo_clocks"] = loo_clocks
    for name, rows in runs.items():
        variants[name] = errors(rows, locked)
        variants[name]["simulated"] = {f"{b}x{p}": rows[(b, p)] for (b, p) in shapes}

    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"Fitted on the 1200 MHz run: sustained_sm_clock_mhz = {sustained:.0f}, "
        f"bandwidth_knee_sm_clock_mhz = {knee:.0f}. Unlocked clock identical to no clock: "
        f"{results['unlocked_matches_no_clock']}.",
        "",
        "| variant | prefill | decode step | end-to-end |",
        "|---|---:|---:|---:|",
    ]
    for name, payload in variants.items():
        m = payload["mean_abs_pct"]
        lines.append(
            f"| {name} | {m['prefill_ms']:.1f}% | {m['decode_ms_per_step']:.1f}% | {m['end_to_end_ms']:.1f}% |"
        )
    lines += ["", "| shape | naive prefill / decode | clock-fit prefill / decode | clock-loo prefill / decode |",
              "|---|---|---|---|"]  # fmt: skip
    for b, p in shapes:
        key = f"{b}x{p}"
        cells = []
        for name in ("naive", "clock-fit", "clock-loo"):
            row = variants[name]["per_shape_pct"][key]
            cells.append(f"{row['prefill_ms']:+.1f}% / {row['decode_ms_per_step']:+.1f}%")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    (HERE / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
