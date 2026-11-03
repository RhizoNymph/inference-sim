"""Re-prediction of the 1-prefill/1-decode NIXL lab run (ledger 18) with
fair-share KV links and the disaggregation proxy constant (ledger 21).

Adapted from lab-runs/2026-10-01-disagg-sim/predict.py. Same workload and the
same curve + frontend calibration; three variants:

- "fifo": ledger 18's curve+frontend prediction (FIFO link queues).
- "fair_share": calibration.kv_link_sharing = "fair_share".
- "fair_share+proxy": plus calibration.disaggregation_proxy_latency_us,
  estimated below from the low-load data.

Proxy estimate (avoiding double counting): the colocated frontend latency
(5081 us + 14.44 us/token = 12.5 ms at 512 tokens) is already applied once to
every request's arrival. What remains at low load (rates 1-4, where nothing
queues) is measured TTFT p50 - simulated TTFT p50 (fair share, no proxy). Part
of that is the KV pull, which the simulator prices from the node0->node1
curve's derived region (81 ms) while vLLM's NIXL metrics in decode_server.log
report ~103 ms per 28 MB transfer at rate 1. The proxy constant is the mean
over rates 1-4 of the residual minus that pull gap (NIXL mean xfer time minus
the simulated mean pull sojourn), so it does not absorb the pull error either.

Run from the repo root after `cargo build --release`.
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
MEASURED = Path("lab-runs/2026-10-01-disagg-1p1d")
BINARY = "target/release/inference-sim"
CLUSTER = "examples/rtx3090_lab_cluster_measured_curves.toml"
BASE = Path("examples/rtx3090_qwen7b_disaggregated_workload.toml").read_text()
CURVE = """[calibration]
compute_efficiency_curve = [[128, 0.653909], [256, 0.618297], [512, 0.724381], [1024, 0.83777], [2048, 0.846627], [4096, 0.873915]]
frontend_latency_us = 5081.43
frontend_latency_per_prompt_token_us = 14.4422
compute_efficiency = 0.849357
decode_memory_bandwidth_scale = 0.838309
kv_link_sharing = "{sharing}"
disaggregation_proxy_latency_us = {proxy_us}
"""
RATES = ["1", "2", "4", "6", "8", "inf"]
LOW_LOAD = ["1", "2", "4"]
FIELDS = ["ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms", "tpot_p50_ms", "itl_p50_ms",
          "throughput_tokens_per_s", "kv_transfer_ms", "kv_resource_queue_ms"]
XFER = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[metrics.py:\d+\] KV Transfer metrics: "
                  r"Num successful transfers=(\d+), Avg xfer time \(ms\)=([\d.]+), P90 xfer time \(ms\)=([\d.]+)")


def workload(rate: str, sharing: str, proxy_us: float) -> str:
    text = BASE
    if rate == "inf":
        text = re.sub(r'^arrival = "poisson"$', 'arrival = "fixed"\narrival_gap_ms = 0.0', text, flags=re.M)
        text = re.sub(r"^arrival_rate_per_s = .*\n", "", text, flags=re.M)
    else:
        text = re.sub(r"^arrival_rate_per_s = .*$", f"arrival_rate_per_s = {float(rate)}", text, flags=re.M)
    calibration = CURVE.format(sharing=sharing, proxy_us=proxy_us)
    return re.sub(r"^\[calibration\]\n(?:[^\[\n].*\n)*", calibration + "\n", text, flags=re.M)


def run(text: str) -> dict:
    path = HERE / "workload.tmp.toml"
    path.write_text(text)
    out = subprocess.run([BINARY, "--cluster", CLUSTER, "--workload", str(path), "--json", "--top-k", "1"],
                         capture_output=True, text=True, check=True).stdout
    path.unlink()
    result = json.loads(out)["results"][0]
    if not result["feasible"]:
        raise SystemExit(f"infeasible: {result.get('status')}")
    return {k: result["metrics"][k] for k in FIELDS}


def measured() -> dict:
    rows = {}
    for rate in RATES:
        bench = json.loads((MEASURED / f"rate_{rate}.json").read_text())
        rows[rate] = {
            "ttft_p50_ms": bench["p50_ttft_ms"], "ttft_p90_ms": bench["p90_ttft_ms"],
            "ttft_p99_ms": bench["p99_ttft_ms"], "tpot_p50_ms": bench["p50_tpot_ms"],
            "itl_p50_ms": bench["p50_itl_ms"], "throughput_tokens_per_s": bench["output_throughput"],
            "end": datetime.strptime(bench["date"], "%Y%m%d-%H%M%S"), "duration_s": bench["duration"],
        }
    return rows


def nixl_transfers(rows: dict) -> dict:
    """Transfer-weighted average NIXL xfer time per benchmark (10 s windows
    whose timestamp falls inside that benchmark's run, plus one window)."""
    windows = []
    for line in (MEASURED / "decode_server.log").read_text().splitlines():
        match = XFER.search(line)
        if match:
            stamp = datetime.strptime("2026-" + match.group(1), "%Y-%m-%d %H:%M:%S")
            windows.append((stamp, int(match.group(2)), float(match.group(3)), float(match.group(4))))
    out = {}
    for rate, row in rows.items():
        start = row["end"] - timedelta(seconds=row["duration_s"])
        inside = [w for w in windows if start <= w[0] <= row["end"] + timedelta(seconds=10)]
        count = sum(w[1] for w in inside)
        out[rate] = {
            "transfers": count,
            "avg_xfer_ms": sum(w[1] * w[2] for w in inside) / count if count else float("nan"),
            "max_window_p90_ms": max((w[3] for w in inside), default=float("nan")),
        }
    return out


def main() -> int:
    real = measured()
    nixl = nixl_transfers(real)
    predictions = {"fifo": {}, "fair_share": {}}
    for label, sharing in (("fifo", "fifo"), ("fair_share", "fair_share")):
        predictions[label] = {rate: run(workload(rate, sharing, 0.0)) for rate in RATES}

    fair = predictions["fair_share"]
    residuals = {r: real[r]["ttft_p50_ms"] - fair[r]["ttft_p50_ms"] for r in LOW_LOAD}
    # Mean simulated pull sojourn (uncontended price + fair-share contention)
    # against NIXL's mean per-transfer time at the same load.
    pull_gaps = {r: nixl[r]["avg_xfer_ms"] - fair[r]["kv_transfer_ms"] - fair[r]["kv_resource_queue_ms"]
                 for r in LOW_LOAD}
    hop = {r: residuals[r] - pull_gaps[r] for r in LOW_LOAD}
    proxy_ms = sum(hop.values()) / len(hop)
    proxy_us = round(proxy_ms * 1000.0, -2)
    predictions["fair_share+proxy"] = {rate: run(workload(rate, "fair_share", proxy_us)) for rate in RATES}

    estimate = {"residual_ttft_p50_ms": residuals, "nixl_minus_sim_pull_ms": pull_gaps,
                "hop_ms": hop, "disaggregation_proxy_latency_us": proxy_us, "nixl": nixl}
    print(json.dumps(estimate, indent=2))
    for label, rows in predictions.items():
        for rate in RATES:
            m, r = rows[rate], real[rate]
            print(f"{label:>17} rate {rate:>3}: TTFT p50 {m['ttft_p50_ms']:8.1f} ({r['ttft_p50_ms']:8.1f}) "
                  f"p90 {m['ttft_p90_ms']:8.1f} ({r['ttft_p90_ms']:8.1f}) | TPOT {m['tpot_p50_ms']:6.2f} "
                  f"({r['tpot_p50_ms']:6.2f}) | tok/s {m['throughput_tokens_per_s']:6.0f} "
                  f"({r['throughput_tokens_per_s']:6.0f}) | pull {m['kv_transfer_ms'] + m['kv_resource_queue_ms']:7.1f}")
    for row in real.values():
        row.pop("end")
    (HERE / "predictions.json").write_text(json.dumps(
        {"measured": real, "proxy_estimate": estimate, "predictions": predictions}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
