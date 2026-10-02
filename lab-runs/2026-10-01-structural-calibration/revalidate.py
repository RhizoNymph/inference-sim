"""Before/after re-validation for the structural calibration gaps (2026-10-01).

Variants (simulator binary x calibration profile):

* before:  base-commit binary (941339e, built from `git archive`) + the
           recalibrated scalar profile (compute_efficiency 0.8494,
           decode_memory_bandwidth_scale 0.8383, 88 TFLOPs peak);
* floor:   new binary + the same scalar profile (adds the prefill weight-read
           floor only);
* curve:   new binary + the token-dependent compute-efficiency curve fitted
           from the prefill token sweep (lab.py fit-curve, tag curve-only);
* curve+frontend (serving only): curve plus the fitted frontend latency.

The 14B PP=2 and 7B PP=2 runs use the recalibration cluster
(lab-runs/2026-09-30-recalibration/rtx3090_lab_cluster_measured_net.toml) and
`collective_latency_scale = 1.824`, as in recalibrate.py; the profile variants
for them are written next to this script.

Run from the repo root:
    python3 lab-runs/2026-10-01-structural-calibration/revalidate.py [BASELINE_BINARY]
BASELINE_BINARY defaults to target/baseline-target/release/inference-sim.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LAB = REPO / "tools" / "lab" / "lab.py"
SPECS = REPO / "tools" / "lab" / "specs"
NEW_BINARY = REPO / "target" / "release" / "inference-sim"
BASE_PROFILE = REPO / "lab-runs/2026-09-28-static-batch/calibration_profile-pp1-recalibrated.toml"
SWEEP = REPO / "lab-runs/2026-10-01-qwen7b-prefill-token-sweep"
CURVE_PROFILE = SWEEP / "calibration_profile-curve-only.toml"
CURVE_FRONTEND_PROFILE = SWEEP / "calibration_profile-curve.toml"
NET_CLUSTER = REPO / "lab-runs/2026-09-30-recalibration/rtx3090_lab_cluster_measured_net.toml"


def with_latency_scale(profile: Path, name: str) -> Path:
    text = profile.read_text(encoding="utf-8")
    out = HERE / name
    out.write_text(
        text.replace("[calibration]\n", "[calibration]\ncollective_latency_scale = 1.824\n", 1), encoding="utf-8"
    )
    return out


STATIC_REGIMES = [
    # label, tag slug, spec, run dir, measured file, cluster override
    ("prefill token sweep (1x16-1x4096)", "sweep", "rtx3090_qwen7b_prefill_token_sweep.toml", SWEEP,
     "measured.jsonl", None),
    ("7B PP=1 (5 shapes)", "pp1", "rtx3090_qwen7b_static_pp1.toml", REPO / "lab-runs/2026-09-28-static-batch",
     "real_pp1.jsonl", None),
    ("7B long context 4k-16k (node1)", "longctx", "rtx3090_qwen7b_static_longctx_node1.toml",
     REPO / "lab-runs/2026-09-30-qwen7b-static-longctx-node1", "measured.jsonl", None),
    ("7B decode batch 1-32 (node1)", "decode", "rtx3090_qwen7b_decode_batch_sweep.toml",
     REPO / "lab-runs/2026-10-01-qwen7b-decode-batch-sweep", "measured-b1-32.jsonl", None),
    ("7B PP=2", "pp2", "rtx3090_qwen7b_static_pp2.toml", REPO / "lab-runs/2026-09-28-static-batch",
     "real_pp2.jsonl", NET_CLUSTER),
    ("14B PP=2", "14b-pp2", "rtx3090_qwen14b_static_pp2.toml", REPO / "lab-runs/2026-09-30-qwen14b-static-pp2",
     "measured.jsonl", NET_CLUSTER),
]  # fmt: skip


def validate(spec: str, run_dir: Path, tag: str, binary: Path, profile: Path, extra: list[str]) -> dict:
    cmd = [
        sys.executable, str(LAB), "validate", str(SPECS / spec), "--run-dir", str(run_dir), "--profile",
        str(profile), "--binary", str(binary), "--tag", tag, *extra,
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd=REPO)
    return json.loads((run_dir / f"validation-{tag}.json").read_text(encoding="utf-8"))


def main() -> None:
    baseline = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "target/baseline-target/release/inference-sim"
    net_base = with_latency_scale(BASE_PROFILE, "profile-scalar-net.toml")
    net_curve = with_latency_scale(CURVE_PROFILE, "profile-curve-net.toml")
    results: dict[str, dict] = {"static": {}, "serving": {}}
    for label, slug, spec, run_dir, measured, cluster in STATIC_REGIMES:
        extra = ["--measured", measured] + (["--cluster", str(cluster)] if cluster else [])
        scalar, curve = (net_base, net_curve) if cluster else (BASE_PROFILE, CURVE_PROFILE)
        rows = {}
        for variant, binary, profile in (
            ("before", baseline, scalar),
            ("floor", NEW_BINARY, scalar),
            ("curve", NEW_BINARY, curve),
        ):
            payload = validate(spec, run_dir, f"sc-{slug}-{variant}", binary, profile, extra)
            rows[variant] = {
                "summary": payload["summary"],
                "per_shape": {r["shape"]: r["error_pct"] for r in payload["rows"]},
            }
        results["static"][label] = rows
    serving_dir = REPO / "lab-runs/2026-09-30-serving-baseline"
    for variant, binary, profile in (
        ("before", baseline, BASE_PROFILE),
        ("floor", NEW_BINARY, BASE_PROFILE),
        ("curve", NEW_BINARY, CURVE_PROFILE),
        ("curve+frontend", NEW_BINARY, CURVE_FRONTEND_PROFILE),
    ):
        tag = "sc-serving-" + variant.replace("+", "-")
        payload = validate("rtx3090_qwen7b_serving_pp1.toml", serving_dir, tag, binary, profile, [])
        results["serving"][variant] = {
            "mean_abs_pct": payload["mean_abs_pct"],
            "ttft_median_pct": {
                r["request_rate"]: 100.0
                * (r["simulated"]["ttft"]["median_ms"] - r["measured"]["ttft"]["median_ms"])
                / r["measured"]["ttft"]["median_ms"]
                for r in payload["rates"]
            },
        }
    (HERE / "revalidation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = ["| regime | variant | prefill | decode step | end-to-end |", "|---|---|---:|---:|---:|"]
    for label, rows in results["static"].items():
        for variant, row in rows.items():
            s = row["summary"]
            lines.append(
                f"| {label} | {variant} | {s['prefill_mean_abs_pct']:.1f}% | {s['decode_step_mean_abs_pct']:.1f}% "
                f"| {s['end_to_end_mean_abs_pct']:.1f}% |"
            )
    lines += ["", "| serving variant | TTFT p50 | TPOT p50 | ITL p50 | E2EL p50 | tok/s | TTFT p50 at 1/2/4 req/s |",
              "|---|---:|---:|---:|---:|---:|---|"]  # fmt: skip
    for variant, row in results["serving"].items():
        m = row["mean_abs_pct"]
        low = " / ".join(f"{row['ttft_median_pct'][r]:+.0f}%" for r in ("1", "2", "4"))
        lines.append(
            f"| {variant} | {m['ttft_median']:.1f}% | {m['tpot_median']:.1f}% | {m['itl_median']:.1f}% "
            f"| {m['e2el_median']:.1f}% | {m['output_throughput']:.1f}% | {low} |"
        )
    (HERE / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
