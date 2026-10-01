"""Compare TP=2 and PP=2 static-batch runs against the simulator under three network models.

Models (all with the 3090 compute constants compute_efficiency 0.8507,
decode_memory_bandwidth_scale 0.8383 fitted on PP=1):

  alpha-beta, fitted   node0 NIC 3.61 Gb/s both ways + collective_latency_scale 1.8238
                       fitted on the 1x512 TP=2 decode step (the previous best)
  alpha-beta, asym     node0 NIC egress capped at 3.61 Gb/s (one-way link), no fitted scale
  measured curves      asym base + [[collective_curves]] from the NCCL sweep, no fitted scale

Writes report-collective-curves.md and comparison-collective-curves.json next to
this file. Usage (from the repo root, after `cargo build --release`):

  python3 lab-runs/2026-09-30-tp2/compare_collective_curves.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from sim_sweep import simulate, workload  # noqa: E402

BINARY = REPO / "target" / "release" / "inference-sim"
CALIBRATION = "[calibration]\ncompute_efficiency = 0.8507\ndecode_memory_bandwidth_scale = 0.8383\n"
FITTED_SCALE = 1.8238
FIT_SHAPE = (1, 512)
DECODE = 128

MODELS = {
    "alpha-beta, fitted (previous best)": (HERE / "rtx3090_lab_cluster_measured_net.toml", FITTED_SCALE),
    "alpha-beta, symmetric 9.41 Gb/s": (HERE.parent / "2026-09-28-static-batch" / "rtx3090_lab_cluster.toml", None),
    "alpha-beta, asymmetric NIC": (HERE / "rtx3090_lab_cluster_asymmetric.toml", None),
    "measured curves": (HERE / "rtx3090_lab_cluster_measured_curves.toml", None),
}

RUNS = {
    "TP=2": (HERE / "real_tp2.jsonl", 2, 1),
    "PP=2": (HERE.parent / "2026-09-28-static-batch" / "real_pp2.jsonl", 1, 2),
}


FINDINGS = """
## Findings

- **TP=2: measured curves, no fitted network scalar, 4.0% mean |e2e| over all five shapes
  (all held out)**, against 9.6% held-out for the previous best (alpha-beta with a
  `collective_latency_scale` fitted on 1x512). Curves alone (old TP trace) gave 7.0%; adding
  the two tensor-parallel collectives the trace was missing (the vocab-parallel embedding
  all-reduce and the LM-head logits all-gather, both real vLLM operations) gave the rest.
- **Batch-8 decode is fixed.** The 57 KiB decode all-reduce sits just past NCCL's 32-64 KiB
  protocol step, so a latency-plus-bandwidth line through the small- and large-message regimes
  under-prices it (previous best: -24.6% / -18.2%). The curve interpolates 338 us, and the
  steps land at +1.7% / +9.8% (real batch-8 steps differ by
  6% between prompt 512 and 2048, so the second shape is within run-to-run spread).
- **Batch-32 decode** went from -27% (curves, old trace) to -5.2% once the logits all-gather
  (32 x 76,032 x 2 B = 4.9 MB per step per rank) is in the trace. That error trended with batch
  size, the signature of a missing operation rather than a constant.
- **Residual TP=2 errors.** Decode at batch 1 is +11.8%: the isolated benchmark's 7 KiB
  all-reduce (161 us) includes eager-launch overhead that vLLM's CUDA-graph decode does not pay
  (implied in-situ cost ~134 us). Prefill is under-predicted (-4% to -25%, worst at 1x512): the
  in-situ all-reduce at 3.7 MB is ~1.4x the isolated benchmark, and 1x512 also carries the
  ~20 ms fixed prefill overhead seen at PP=1. Both look like benchmark-vs-in-situ differences
  (launch path, rank skew while waiting for the slower GPU) rather than curve shape; an in-situ
  NCCL trace or a benchmark that interleaves compute would separate them.
- **PP=2: 2.4% with curves vs 1.6% (asymmetric alpha-beta) and 1.1% (symmetric 9.41 Gb/s).**
  The curves price node0 -> node1 activation sends at the measured single-stream p2p rate
  (0.35 GB/s at 64-256 MiB; derived below 64 MiB). The real PP=2 timings cannot contain a
  serial transfer at that rate: PP=2 prefill at 8x512 and 8x2048 is *faster* than PP=1 on one
  GPU (664.7 vs 780.6 ms, 2,893.7 vs 3,095.5 ms) while the 117 MB 8x2048 activation alone would
  take ~335 ms at 0.35 GB/s. vLLM is overlapping or pipelining the PP=2 prefill (or its
  transfer path does not hit the one-way limit), which the simulator does not model; prefill is
  over-predicted by 20-29% at batch 8 under every network model. The PP=2 e2e difference
  (+0.8 points) sits inside the measured e2e spreads (13-627 ms, up to 10% of a shape).
- **Small-message send_0to1 rows** (sender-timed) were dropped by `lab.py curves`: every row
  from 1 KiB to 32 MiB (32 KiB-32 MiB imply more bandwidth than 1.25x the 256 MiB row; smaller
  rows sit below an unreliable size). The 0 -> 1 curve keeps the measured 64/128/256 MiB rows and
  derives smaller sizes as `t_1to0(b) + b (1/bw_0to1 - 1/bw_1to0)`; lookups there carry the
  `collective_curve_derived_region` approximation. Re-measuring with the receiver-timed
  `tools/lab/remote/collective_bench.py` removes the derivation.

## What this implies for the H100 plan

- Fit curves per fabric scope from `nccl-tests` (all_reduce_perf, all_gather_perf,
  reduce_scatter_perf, alltoall_perf, sendrecv_perf) instead of fitting latency/bandwidth
  scalars: intra-node NVLink/NVSwitch at 2/4/8 ranks (`scope = "intra_node"`), and inter-node IB
  at each node count used (`scope = "inter_node"` or the exact `node_group`). Convert nccl-tests
  sizes to the simulator's key (per-rank contributed bytes: all_gather `size / ranks`), take
  out-of-place time, and keep the full 1 KiB-8 GiB sweep so TP decode messages (tens of KiB) and
  prefill messages (tens to hundreds of MB) are interpolated, never extrapolated.
- Measure p2p in both directions with receiver-side timing on every node pair class (same rail,
  cross rail, cross leaf); asymmetric links are now representable (`egress/ingress_bandwidth_gbps`,
  `from_to_/to_from_bandwidth_gbps`) and directional curves price PP and KV sends per direction.
- Expect the benchmark-vs-in-situ gap seen here (eager launch overhead at small sizes, CUDA graphs
  in decode, rank skew): benchmark with CUDA graphs (`nccl-tests -G`) for the decode regime and
  validate one TP decode and one prefill shape end-to-end before trusting the curves elsewhere.
- Scenario overlays that degrade the network suspend curves (alpha-beta takes over with a
  `collective_curves_suspended` record), so degraded-fabric studies on H100 still need the
  alpha-beta parameters to be sane.
"""


def sim_row(cluster: Path, scale: float | None, batch: int, prompt: int, tp: int, pp: int) -> dict:
    calibration = CALIBRATION + (f"collective_latency_scale = {scale}\n" if scale else "")
    out: dict[str, float] = {}
    approximations: set[str] = set()
    for phase, decode in (("prefill", 1), ("decode", DECODE), ("end_to_end", DECODE)):
        ms, result = simulate(BINARY, cluster, workload(batch, prompt, decode, phase, tp, pp, calibration))
        out[phase] = ms
        approximations |= {a["code"] for a in result.get("approximations", [])}
    out["decode_step"] = out["decode"] / DECODE
    out["codes"] = sorted(approximations)
    return out


def err(sim: float, real: float) -> float:
    return (sim - real) / real * 100.0


def main() -> int:
    results: dict[str, dict[str, list[dict]]] = {}
    for run_name, (path, tp, pp) in RUNS.items():
        real_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for model, (cluster, scale) in MODELS.items():
            rows = []
            for real in real_rows:
                sim = sim_row(cluster, scale, real["batch"], real["prompt"], tp, pp)
                rows.append(
                    {
                        "shape": f"{real['batch']}x{real['prompt']}",
                        "batch": real["batch"],
                        "prompt": real["prompt"],
                        "real": {
                            "prefill_ms": real["prefill_ms"],
                            "decode_step_ms": real["decode_ms_per_step"],
                            "e2e_ms": real["end_to_end_ms"],
                        },
                        "sim": {
                            "prefill_ms": round(sim["prefill"], 3),
                            "decode_step_ms": round(sim["decode_step"], 4),
                            "e2e_ms": round(sim["end_to_end"], 3),
                        },
                        "err_pct": {
                            "prefill": round(err(sim["prefill"], real["prefill_ms"]), 2),
                            "decode_step": round(err(sim["decode_step"], real["decode_ms_per_step"]), 2),
                            "e2e": round(err(sim["end_to_end"], real["end_to_end_ms"]), 2),
                        },
                        "approximation_codes": sim["codes"],
                    }
                )
            results.setdefault(run_name, {})[model] = rows

    lines = [
        "# Measured collective curves vs alpha-beta: TP=2 and PP=2 over 2x10GbE",
        "",
        "Generated by `compare_collective_curves.py`. Errors are `(sim - measured) / measured`.",
        "All models use the PP=1-fitted 3090 compute constants (`compute_efficiency = 0.8507`,",
        "`decode_memory_bandwidth_scale = 0.8383`). Only the previous-best model fits anything on",
        f"TP=2 data (`collective_latency_scale = {FITTED_SCALE}` on the 1x512 decode step), so its",
        "held-out mean excludes 1x512; every other model is fully held out.",
        "",
        "Clusters:",
        "",
        "- alpha-beta, fitted: `rtx3090_lab_cluster_measured_net.toml` (node0 NIC 3.61 Gb/s both ways) + fitted scale.",
        "- alpha-beta, symmetric: `../2026-09-28-static-batch/rtx3090_lab_cluster.toml` (9.41 Gb/s everywhere).",
        "- alpha-beta, asymmetric NIC: `rtx3090_lab_cluster_asymmetric.toml` (node0 egress capped at 3.61 Gb/s).",
        "- measured curves: `rtx3090_lab_cluster_measured_curves.toml` = the asymmetric cluster +",
        "  `[[collective_curves]]` generated by `tools/lab/lab.py curves` from",
        "  `../2026-09-30-nccl-curve/collective_curve.jsonl` (all_reduce nodes 0+1; send_recv 1->0 measured;",
        "  send_recv 0->1 measured at 64-256 MiB, derived below 64 MiB from the 1->0 curve plus the",
        "  bandwidth difference because the benchmark timed node0's sends on the sender).",
        "",
    ]
    summary = []
    for run_name, by_model in results.items():
        lines += [f"## {run_name}", ""]
        for model, rows in by_model.items():
            lines += [
                f"### {run_name}: {model}",
                "",
                "| shape | prefill real / sim ms | err | decode step real / sim ms | err | e2e real / sim ms | err |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for row in rows:
                r, s, e = row["real"], row["sim"], row["err_pct"]
                lines.append(
                    f"| {row['shape']} | {r['prefill_ms']:,.1f} / {s['prefill_ms']:,.1f} | {e['prefill']:+.1f}% "
                    f"| {r['decode_step_ms']:.2f} / {s['decode_step_ms']:.2f} | {e['decode_step']:+.1f}% "
                    f"| {r['e2e_ms']:,.1f} / {s['e2e_ms']:,.1f} | {e['e2e']:+.1f}% |"
                )
            fitted = model.startswith("alpha-beta, fitted") and run_name == "TP=2"
            held = [row for row in rows if not (fitted and (row["batch"], row["prompt"]) == FIT_SHAPE)]

            def mean(key: str, subset: list[dict]) -> float:
                return sum(abs(row["err_pct"][key]) for row in subset) / len(subset)

            entry = {
                "run": run_name,
                "model": model,
                "prefill": round(mean("prefill", rows), 2),
                "decode_step": round(mean("decode_step", rows), 2),
                "e2e": round(mean("e2e", rows), 2),
                "e2e_held_out": round(mean("e2e", held), 2),
                "held_out_shapes": len(held),
            }
            summary.append(entry)
            lines += [
                f"| **mean \\|err\\|** | | **{entry['prefill']:.1f}%** | | **{entry['decode_step']:.1f}%** "
                f"| | **{entry['e2e']:.1f}%** |",
                "",
                f"Held-out mean |e2e err| over {len(held)} shapes: **{entry['e2e_held_out']:.1f}%**.",
                "",
            ]
            codes = sorted({code for row in rows for code in row["approximation_codes"] if "collective" in code})
            if codes:
                lines += [f"Collective evidence codes: {', '.join(f'`{c}`' for c in codes)}.", ""]
    # Recorded with this script before the TP trace gained the embedding
    # all-reduce and logits all-gather (same binary otherwise); kept so the
    # report separates the curve effect from the trace fix.
    summary[:0] = [
        {"run": "TP=2", "model": "alpha-beta, fitted (previous best), old TP trace",
         "prefill": 7.8, "decode_step": 11.6, "e2e": 8.1, "e2e_held_out": 9.6, "held_out_shapes": 4},
        {"run": "TP=2", "model": "measured curves, old TP trace",
         "prefill": 13.9, "decode_step": 9.9, "e2e": 7.0, "e2e_held_out": 7.0, "held_out_shapes": 5},
    ]
    lines += [
        "## Summary (mean |err|)",
        "",
        "| run | model | prefill | decode step | e2e | e2e held out |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for entry in summary:
        lines.append(
            f"| {entry['run']} | {entry['model']} | {entry['prefill']:.1f}% | {entry['decode_step']:.1f}% "
            f"| {entry['e2e']:.1f}% | {entry['e2e_held_out']:.1f}% ({entry['held_out_shapes']} shapes) |"
        )
    lines.append("")
    lines += FINDINGS.splitlines()
    (HERE / "report-collective-curves.md").write_text("\n".join(lines))
    (HERE / "comparison-collective-curves.json").write_text(
        json.dumps({"summary": summary, "rows": results}, indent=2) + "\n"
    )
    print("\n".join(lines[-(len(summary) + 4):]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
