"""Before/after validation of pipeline overlap inside one static batch (ledger 23).

Variants (same simulator binary, built from feat/pipeline-batch-overlap):

* before: the structural-calibration profile used for ledger 14
          (lab-runs/2026-10-01-structural-calibration/profile-curve-net.toml:
          efficiency curve, recalibrated scalars, collective_latency_scale
          1.824), whose default `pipeline_microbatching = "whole_batch"`
          reproduces the pre-change simulator exactly;
* after:  the same profile plus `pipeline_microbatching = "vllm_batch_queue"`
          (written next to this script). No constant is refitted.

PP=2 regimes use the recalibration cluster (node0 NIC 3.61 Gb/s), as in
ledger 14. The 7B PP=1 regime checks that single-stage results are unchanged.

Why (vLLM v0.29.0 source, read from the GitHub tag; no local install):
config/vllm.py max_concurrent_batches = pipeline_parallel_size when async
scheduling is off (Ray's executor does not support it); v1/engine/core.py
step_with_batch_queue submits a new step without blocking while the queue has
room; v1/core/sched/scheduler.py skips requests whose prompt is in flight
(num_new_tokens == 0 under PP); entrypoints/offline_utils.py adds offline
requests one at a time, and the engine core steps as soon as the first one
arrives; v1/worker/gpu_worker.py waits on the previous step's pipeline send
before the next forward. Hence a 1 + (B - 1) split whose prompt passes overlap
across stages. Full write-up: docs/features/pipeline_parallel_overlap.md and
docs/validation_ledger.md entry 23; per-shape numbers in summary.md.

Run from the repo root after `cargo build --release`:
    python3 lab-runs/2026-10-01-pipeline-overlap-sim/revalidate.py
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
BEFORE_PROFILE = REPO / "lab-runs/2026-10-01-structural-calibration/profile-curve-net.toml"
NET_CLUSTER = REPO / "lab-runs/2026-09-30-recalibration/rtx3090_lab_cluster_measured_net.toml"

REGIMES = [
    # label, slug, spec, measured file, cluster override
    ("7B PP=2", "pp2", "rtx3090_qwen7b_static_pp2.toml",
     REPO / "lab-runs/2026-09-28-static-batch/real_pp2.jsonl", NET_CLUSTER),
    ("14B PP=2", "14b-pp2", "rtx3090_qwen14b_static_pp2.toml",
     REPO / "lab-runs/2026-09-30-qwen14b-static-pp2/measured.jsonl", NET_CLUSTER),
    ("7B PP=1 (unchanged check)", "pp1", "rtx3090_qwen7b_static_pp1.toml",
     REPO / "lab-runs/2026-09-28-static-batch/real_pp1.jsonl", None),
]  # fmt: skip


def overlap_profile() -> Path:
    text = BEFORE_PROFILE.read_text(encoding="utf-8")
    out = HERE / "profile-curve-net-overlap.toml"
    out.write_text(
        text.replace("[calibration]\n", '[calibration]\npipeline_microbatching = "vllm_batch_queue"\n', 1),
        encoding="utf-8",
    )
    return out


def validate(spec: str, measured: Path, tag: str, profile: Path, cluster: Path | None) -> dict:
    cmd = [
        sys.executable, str(LAB), "validate", str(SPECS / spec), "--run-dir", str(HERE),
        "--measured", str(measured), "--profile", str(profile), "--binary", str(BINARY), "--tag", tag,
    ] + (["--cluster", str(cluster)] if cluster else [])  # fmt: skip
    subprocess.run(cmd, check=True, cwd=REPO)
    return json.loads((HERE / f"validation-{tag}.json").read_text(encoding="utf-8"))


def main() -> None:
    after_profile = overlap_profile()
    results: dict[str, dict] = {}
    for label, slug, spec, measured, cluster in REGIMES:
        rows = {}
        for variant, profile in (("before", BEFORE_PROFILE), ("after", after_profile)):
            payload = validate(spec, measured, f"po-{slug}-{variant}", profile, cluster)
            rows[variant] = {"summary": payload["summary"], "rows": payload["rows"]}
        results[label] = rows
    (HERE / "revalidation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = ["| regime | variant | prefill | decode step | end-to-end |", "|---|---|---:|---:|---:|"]
    for label, rows in results.items():
        for variant, row in rows.items():
            s = row["summary"]
            lines.append(
                f"| {label} | {variant} | {s['prefill_mean_abs_pct']:.1f}% | {s['decode_step_mean_abs_pct']:.1f}% "
                f"| {s['end_to_end_mean_abs_pct']:.1f}% |"
            )
    lines += ["", "Per shape (signed error, before -> after):", "",
              "| regime | shape | prefill | decode step | end-to-end |", "|---|---|---|---|---|"]  # fmt: skip
    for label, rows in results.items():
        before = {r["shape"]: r["error_pct"] for r in rows["before"]["rows"]}
        after = {r["shape"]: r["error_pct"] for r in rows["after"]["rows"]}
        for shape, b in before.items():
            a = after[shape]
            cells = [f"{b[k]:+.1f}% -> {a[k]:+.1f}%" for k in ("prefill", "decode_step", "end_to_end")]
            lines.append(f"| {label} | {shape} | " + " | ".join(cells) + " |")
    (HERE / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
