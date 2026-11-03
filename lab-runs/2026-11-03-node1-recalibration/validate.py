"""Validate the node1 November compute recalibration against every same-period run.

Profiles (all scored on the 2026-11 measurements):
- "old": the entry-14 curve profile (node0, Oct 1; 88 TFLOPs peak);
- "new": scalars from the node1 static run + curve from the node1 prefill sweep,
  against the measured-GEMM-peak cluster (rtx3090_cluster_gemm_peak.toml);
- "new-static512": "new" with the 512-token curve point taken from the static
  run's 1x512 instead of the sweep's (see the profile's header).

Runs `lab.py validate` for each (profile, dataset) pair and writes
validation-summary.json and validation-summary.md next to this file. Run from
the repo root after `cargo build --release`:

  python3 lab-runs/2026-11-03-node1-recalibration/validate.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LAB = REPO / "tools" / "lab"
PYTHON = sys.executable
OLD_CLUSTER = LAB / "labs" / "rtx3090_cluster.toml"
NEW_CLUSTER = HERE / "rtx3090_cluster_gemm_peak.toml"

PROFILES = {
    "old": (REPO / "lab-runs/2026-10-01-qwen7b-prefill-token-sweep/calibration_profile-curve-only.toml", OLD_CLUSTER),
    "new": (REPO / "lab-runs/2026-11-03-qwen7b-prefill-token-sweep-node1/calibration_profile-node1-nov.toml", NEW_CLUSTER),
    "new-static512": (HERE / "calibration_profile-node1-nov-static512.toml", NEW_CLUSTER),
}
# (label, spec, run dir, role)
DATASETS = [
    ("node1 static PP=1", "rtx3090_qwen7b_static_pp1_node1", "2026-11-03-qwen7b-static-pp1-node1", "scalars in-sample"),
    ("node1 prefill sweep", "rtx3090_qwen7b_prefill_token_sweep_node1", "2026-11-03-qwen7b-prefill-token-sweep-node1", "curve in-sample"),
    ("node1 long context 4k-8k", "rtx3090_qwen7b_static_longctx_node1", "2026-11-03-qwen7b-static-longctx-node1", "held out"),
    ("node1 long context 16k, 2x8k, 4x4k", "rtx3090_qwen7b_static_longctx_node1_tail", "2026-11-03-qwen7b-static-longctx-node1-tail", "held out"),
    ("node1 decode sweep 1-32 x 512", "rtx3090_qwen7b_decode_batch_sweep", "2026-11-02-qwen7b-decode-batch-sweep", "held out"),
    ("node0 static PP=1 (transfer)", "rtx3090_qwen7b_static_pp1", "2026-11-02-qwen7b-static-pp1", "held out, other node"),
]
MAX_BATCH = 32  # KV exhaustion beyond (ledger 16)


def validate(profile: str, dataset: tuple[str, str, str, str]) -> dict:
    path, cluster = PROFILES[profile]
    label, spec, run, role = dataset
    tag = f"recal-{profile}"
    completed = subprocess.run(
        [PYTHON, "lab.py", "validate", f"specs/{spec}.toml", "--run-dir", str(REPO / "lab-runs" / run),
         "--profile", str(path), "--cluster", str(cluster), "--tag", tag],
        cwd=LAB, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{profile} / {label}: {completed.stderr[-1500:]}")
    result = json.loads((REPO / "lab-runs" / run / f"validation-{tag}.json").read_text())
    rows = [r for r in result["rows"] if int(r["shape"].split("x")[0]) <= MAX_BATCH]

    def mean(key: str) -> float:
        return round(sum(abs(r["error_pct"][key]) for r in rows) / len(rows), 2)

    return {
        "profile": profile, "dataset": label, "role": role, "validation": f"lab-runs/{run}/validation-{tag}.json",
        "mean_abs_pct": {k: mean(k) for k in ("prefill", "decode_step", "end_to_end")},
        "shapes": {r["shape"]: {k: round(v, 2) for k, v in r["error_pct"].items()} for r in rows},
    }


def main() -> int:
    jobs = [(p, d) for p in PROFILES for d in DATASETS]
    with ThreadPoolExecutor(max_workers=6) as pool:
        out = list(pool.map(lambda job: validate(*job), jobs))
    (HERE / "validation-summary.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["| dataset | role | " + " | ".join(f"{p} prefill / step / e2e" for p in PROFILES) + " |",
             "|---|---|" + "---|" * len(PROFILES)]
    for label, _, _, role in DATASETS:
        cells = []
        for profile in PROFILES:
            m = next(o for o in out if o["profile"] == profile and o["dataset"] == label)["mean_abs_pct"]
            cells.append(f"{m['prefill']:.1f}% / {m['decode_step']:.1f}% / {m['end_to_end']:.1f}%")
        lines.append(f"| {label} | {role} | " + " | ".join(cells) + " |")
    (HERE / "validation-summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
