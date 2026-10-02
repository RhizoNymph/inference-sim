"""Blind disaggregated-serving predictions for the 1-prefill/1-decode lab run.

Two calibrations: "scalar" (the disaggregation agent's: recalibrated scalars,
no frontend latency) and "curve+frontend" (token-dependent efficiency curve and
measured frontend latency from the prefill token sweep). Run from repo root.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
BINARY = "target/release/inference-sim"
CLUSTER = "examples/rtx3090_lab_cluster_measured_curves.toml"
BASE = Path("examples/rtx3090_qwen7b_disaggregated_workload.toml").read_text()
CURVE = """[calibration]
compute_efficiency_curve = [[128, 0.653909], [256, 0.618297], [512, 0.724381], [1024, 0.83777], [2048, 0.846627], [4096, 0.873915]]
frontend_latency_us = 5081.43
frontend_latency_per_prompt_token_us = 14.4422
compute_efficiency = 0.849357
decode_memory_bandwidth_scale = 0.838309
"""
RATES = ["1", "2", "4", "6", "8", "inf"]
FIELDS = ["ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "itl_p50_ms", "itl_p99_ms",
          "e2el_p50_ms", "throughput_tokens_per_s"]


def workload(rate: str, curve: bool) -> str:
    text = BASE
    if rate == "inf":
        text = re.sub(r'^arrival = "poisson"$', 'arrival = "fixed"\narrival_gap_ms = 0.0', text, flags=re.M)
        text = re.sub(r"^arrival_rate_per_s = .*\n", "", text, flags=re.M)
    else:
        text = re.sub(r"^arrival_rate_per_s = .*$", f"arrival_rate_per_s = {float(rate)}", text, flags=re.M)
    if curve:
        text = re.sub(r"^\[calibration\]\n(?:[^\[\n].*\n)*", CURVE + "\n", text, flags=re.M)
    return text


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


def main() -> int:
    predictions = {}
    for label, curve in (("scalar", False), ("curve+frontend", True)):
        predictions[label] = {rate: run(workload(rate, curve)) for rate in RATES}
        for rate in RATES:
            m = predictions[label][rate]
            print(f"{label:>15} rate {rate:>3}: TTFT p50 {m['ttft_p50_ms']:8.1f} p99 {m['ttft_p99_ms']:8.1f} | "
                  f"TPOT p50 {m['tpot_p50_ms']:6.2f} | ITL p50 {m['itl_p50_ms']:6.2f} | "
                  f"tok/s {m['throughput_tokens_per_s']:6.0f}")
    (HERE / "predictions.json").write_text(json.dumps(predictions, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
