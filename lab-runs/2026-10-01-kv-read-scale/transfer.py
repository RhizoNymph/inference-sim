"""Transfer check of the fitted KV-read bandwidth scale on regimes it was not fitted on.

Variants (same simulator binary, feat/kv-read-bandwidth-scale):

* before: the curve + frontend profile
  (lab-runs/2026-10-01-qwen7b-prefill-token-sweep/calibration_profile-curve.toml);
* kv:     the same profile plus `kv_read_bandwidth_scale` fitted by
  `lab.py fit-kv-read --mode after` (calibration_profile-after.toml here).

The PP=2 runs use the recalibration cluster (node0 NIC 3.61 Gb/s) and
`collective_latency_scale = 1.824`, as in the 2026-10-01 structural
re-validation; the profile variants for them are written next to this script.

Run from the repo root after `cargo build --release`:
    python3 lab-runs/2026-10-01-kv-read-scale/transfer.py
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
BINARY = REPO / "target" / "release" / "inference-sim"
BEFORE = REPO / "lab-runs/2026-10-01-qwen7b-prefill-token-sweep/calibration_profile-curve.toml"
KV = HERE / "calibration_profile-after.toml"
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
    ("7B PP=1, 5 shapes (node0)", "pp1", "rtx3090_qwen7b_static_pp1.toml", REPO / "lab-runs/2026-09-28-static-batch",
     "real_pp1.jsonl", None),
    ("7B PP=2", "pp2", "rtx3090_qwen7b_static_pp2.toml", REPO / "lab-runs/2026-09-28-static-batch",
     "real_pp2.jsonl", NET_CLUSTER),
    ("14B PP=2 (model transfer)", "14b-pp2", "rtx3090_qwen14b_static_pp2.toml",
     REPO / "lab-runs/2026-09-30-qwen14b-static-pp2", "measured.jsonl", NET_CLUSTER),
]  # fmt: skip


def validate(spec: str, run_dir: Path, tag: str, profile: Path, extra: list[str]) -> dict:
    cmd = [
        sys.executable, str(LAB), "validate", str(SPECS / spec), "--run-dir", str(run_dir), "--profile",
        str(profile), "--binary", str(BINARY), "--tag", tag, *extra,
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd=REPO)
    return json.loads((run_dir / f"validation-{tag}.json").read_text(encoding="utf-8"))


def main() -> None:
    net_before = with_latency_scale(BEFORE, "profile-before-net.toml")
    net_kv = with_latency_scale(KV, "profile-kv-net.toml")
    results: dict[str, dict] = {"static": {}, "serving": {}}
    for label, slug, spec, run_dir, measured, cluster in STATIC_REGIMES:
        extra = ["--measured", measured] + (["--cluster", str(cluster)] if cluster else [])
        before, kv = (net_before, net_kv) if cluster else (BEFORE, KV)
        rows = {}
        for variant, profile in (("before", before), ("kv", kv)):
            payload = validate(spec, run_dir, f"kv-{slug}-{variant}", profile, extra)
            rows[variant] = {
                "summary": payload["summary"],
                "per_shape": {r["shape"]: r["error_pct"] for r in payload["rows"]},
            }
        results["static"][label] = rows
    serving_dir = REPO / "lab-runs/2026-09-30-serving-baseline"
    for variant, profile in (("before", BEFORE), ("kv", KV)):
        payload = validate("rtx3090_qwen7b_serving_pp1.toml", serving_dir, f"kv-serving-{variant}", profile, [])
        results["serving"][variant] = {
            "mean_abs_pct": payload["mean_abs_pct"],
            "ttft_median_pct": {
                r["request_rate"]: 100.0
                * (r["simulated"]["ttft"]["median_ms"] - r["measured"]["ttft"]["median_ms"])
                / r["measured"]["ttft"]["median_ms"]
                for r in payload["rates"]
            },
            "tpot_median_pct": {
                r["request_rate"]: 100.0
                * (r["simulated"]["tpot"]["median_ms"] - r["measured"]["tpot"]["median_ms"])
                / r["measured"]["tpot"]["median_ms"]
                for r in payload["rates"]
            },
        }
    (HERE / "transfer.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = ["| regime | variant | prefill | decode step | end-to-end |", "|---|---|---:|---:|---:|"]
    for label, rows in results["static"].items():
        for variant, row in rows.items():
            s = row["summary"]
            lines.append(
                f"| {label} | {variant} | {s['prefill_mean_abs_pct']:.1f}% | {s['decode_step_mean_abs_pct']:.1f}% "
                f"| {s['end_to_end_mean_abs_pct']:.1f}% |"
            )
    lines += ["", "| serving variant | TTFT p50 | TPOT p50 | ITL p50 | E2EL p50 | tok/s | TPOT p50 at 4/6/8 req/s |",
              "|---|---:|---:|---:|---:|---:|---|"]  # fmt: skip
    for variant, row in results["serving"].items():
        m = row["mean_abs_pct"]
        tpot = " / ".join(f"{row['tpot_median_pct'][r]:+.1f}%" for r in ("4", "6", "8"))
        lines.append(
            f"| {variant} | {m['ttft_median']:.1f}% | {m['tpot_median']:.1f}% | {m['itl_median']:.1f}% "
            f"| {m['e2el_median']:.1f}% | {m['output_throughput']:.1f}% | {tpot} |"
        )
    (HERE / "transfer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
