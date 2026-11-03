"""Re-predict the TP=2, PP=2 and disaggregated lab runs with the receiver-timed curves.

The 2026-09-30 sweep timed node0's sends on the sender, so the send_recv 0->1
curve was measured only at 64-256 MiB and derived below. The 2026-10-01 sweep
(`../collective_curve.jsonl`) times every send on the receiver and adds
all_gather, reduce_scatter and all_to_all. This script reruns the same
predictions with old and new curves and nothing else changed:

- TP=2 / PP=2 static batch: entry-12/13 setup (asymmetric cluster, PP=1
  compute constants, no fitted network scalar); old curves =
  `../../2026-09-30-tp2/rtx3090_lab_cluster_measured_curves.toml`, new =
  `rtx3090_lab_cluster_receiver_timed_curves.toml` (`lab.py curves` on the
  same asymmetric base).
- Disaggregated 1P1D serving: entry-18 "curve+frontend" calibration on the
  example cluster; old curves = the example as of entry 18 (git show
  7ef3439:examples/rtx3090_lab_cluster_measured_curves.toml), new = the
  example now.

Writes predictions.json next to this file. Run from the repo root after
`cargo build --release`:

  python3 lab-runs/2026-10-01-nccl-curve/repredict/repredict.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TP2_DIR = REPO / "lab-runs" / "2026-09-30-tp2"
DISAGG_SIM = REPO / "lab-runs" / "2026-10-01-disagg-sim"
DISAGG_REAL = REPO / "lab-runs" / "2026-10-01-disagg-1p1d"
sys.path.insert(0, str(TP2_DIR))
sys.path.insert(0, str(DISAGG_SIM))

import predict as disagg  # noqa: E402
from sim_sweep import simulate, workload  # noqa: E402

BINARY = REPO / "target" / "release" / "inference-sim"
STATIC_CALIBRATION = "[calibration]\ncompute_efficiency = 0.8507\ndecode_memory_bandwidth_scale = 0.8383\n"
DECODE = 128
OLD_EXAMPLE_REV = "7ef3439"
EXAMPLE = "examples/rtx3090_lab_cluster_measured_curves.toml"

STATIC_CLUSTERS = {
    "old curves": TP2_DIR / "rtx3090_lab_cluster_measured_curves.toml",
    "receiver-timed curves": HERE / "rtx3090_lab_cluster_receiver_timed_curves.toml",
}
STATIC_RUNS = {
    "TP=2": (TP2_DIR / "real_tp2.jsonl", 2, 1),
    "PP=2": (REPO / "lab-runs" / "2026-09-28-static-batch" / "real_pp2.jsonl", 1, 2),
}


def err(sim: float, real: float) -> float:
    return round((sim - real) / real * 100.0, 2)


def static_row(cluster: Path, real: dict, tp: int, pp: int) -> dict:
    sim = {}
    for phase, decode in (("prefill", 1), ("decode", DECODE), ("end_to_end", DECODE)):
        sim[phase], _ = simulate(
            BINARY, cluster, workload(real["batch"], real["prompt"], decode, phase, tp, pp, STATIC_CALIBRATION)
        )
    step = sim["decode"] / DECODE
    return {
        "shape": f"{real['batch']}x{real['prompt']}",
        "real": {"prefill_ms": real["prefill_ms"], "decode_step_ms": real["decode_ms_per_step"],
                 "e2e_ms": real["end_to_end_ms"]},
        "sim": {"prefill_ms": round(sim["prefill"], 3), "decode_step_ms": round(step, 4),
                "e2e_ms": round(sim["end_to_end"], 3)},
        "err_pct": {"prefill": err(sim["prefill"], real["prefill_ms"]),
                    "decode_step": err(step, real["decode_ms_per_step"]),
                    "e2e": err(sim["end_to_end"], real["end_to_end_ms"])},
    }


def static_predictions() -> dict:
    out: dict = {}
    for run, (path, tp, pp) in STATIC_RUNS.items():
        reals = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for label, cluster in STATIC_CLUSTERS.items():
            rows = [static_row(cluster, real, tp, pp) for real in reals]
            mean = {k: round(sum(abs(r["err_pct"][k]) for r in rows) / len(rows), 2)
                    for k in ("prefill", "decode_step", "e2e")}
            out.setdefault(run, {})[label] = {"rows": rows, "mean_abs_err_pct": mean}
            print(f"{run} {label:>22}: mean |err| prefill {mean['prefill']:5.1f}% "
                  f"decode {mean['decode_step']:5.1f}% e2e {mean['e2e']:5.1f}%")
    return out


def disagg_predictions() -> dict:
    old_cluster = HERE / "old_example_cluster.tmp.toml"
    old_cluster.write_text(
        subprocess.run(["git", "show", f"{OLD_EXAMPLE_REV}:{EXAMPLE}"], cwd=REPO, capture_output=True,
                       text=True, check=True).stdout
    )
    out: dict = {}
    try:
        for label, cluster in (("old curves", str(old_cluster)), ("receiver-timed curves", EXAMPLE)):
            disagg.CLUSTER = cluster
            out[label] = {rate: disagg.run(disagg.workload(rate, True)) for rate in disagg.RATES}
    finally:
        old_cluster.unlink()
    real = {rate: json.loads((DISAGG_REAL / f"rate_{rate}.json").read_text()) for rate in disagg.RATES}
    out["real"] = {rate: {k: real[rate][k] for k in ("median_ttft_ms", "median_tpot_ms", "output_throughput")}
                   for rate in disagg.RATES}
    for rate in disagg.RATES:
        r = out["real"][rate]["median_ttft_ms"]
        o, n = (out[label][rate]["ttft_p50_ms"] for label in ("old curves", "receiver-timed curves"))
        print(f"disagg rate {rate:>3}: TTFT p50 real {r:8.1f} | old {o:8.1f} ({err(o, r):+6.1f}%) "
              f"| new {n:8.1f} ({err(n, r):+6.1f}%)")
    return out


def main() -> int:
    predictions = {"static": static_predictions(), "disaggregated": disagg_predictions()}
    (HERE / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
