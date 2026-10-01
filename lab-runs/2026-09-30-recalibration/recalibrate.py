"""Refit the 3090 compute constants with correct model FLOPs (explicit
ffn_hidden_size) and a clock-adjusted peak, then re-validate every regime.

Run from the repo root: python3 lab-runs/2026-09-30-recalibration/recalibrate.py
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "2026-09-30-tp2"))
import sim_sweep as s  # noqa: E402

BINARY = "target/release/inference-sim"
CLUSTER = str(HERE / "rtx3090_lab_cluster_measured_net.toml")
MODELS = {
    "7b": dict(layers=28, hidden=3584, heads=28, kv=4, ffn=18944, gb=15.23),
    "14b": dict(layers=48, hidden=5120, heads=40, kv=8, ffn=13824, gb=29.54),
}
LATENCY_SCALE = 1.824  # network fit from TP=2 (unchanged: decode/network, not FLOPs)


def model_toml(name):
    m = MODELS[name]
    return (
        f'[model]\nid = "qwen2.5-{name}"\nlayers = {m["layers"]}\nhidden_size = {m["hidden"]}\n'
        f'attention_heads = {m["heads"]}\nkv_heads = {m["kv"]}\nffn_hidden_size = {m["ffn"]}\n'
        f'vocab_size = 152064\nparameters_gb = {m["gb"]}\ndtype = "bf16"\n'
    )


def sim(name, b, p, phase, d, tp, pp, ce, bw):
    s.MODEL = model_toml(name)
    cal = (
        f"[calibration]\ncompute_efficiency = {ce}\ndecode_memory_bandwidth_scale = {bw}\n"
        f"collective_latency_scale = {LATENCY_SCALE}\n"
    )
    return s.simulate(BINARY, CLUSTER, s.workload(b, p, d, phase, tp, pp, cal))[0]


def load(path):
    return [json.loads(line) for line in open(path) if line.startswith("{") and '"prefill_ms"' in line]


def evaluate(label, name, rows, tp, pp, ce, bw):
    errs = []
    e = lambda a, b: (a - b) / b * 100
    for r in rows:
        b, p, d = r["batch"], r["prompt"], r["decode"]
        pre = sim(name, b, p, "prefill", 1, tp, pp, ce, bw)
        dec = sim(name, b, p, "decode", d, tp, pp, ce, bw) / d
        e2e = sim(name, b, p, "end_to_end", d, tp, pp, ce, bw)
        errs.append((e(pre, r["prefill_ms"]), e(dec, r["decode_ms_per_step"]), e(e2e, r["end_to_end_ms"])))
    mae = lambda i: statistics.mean(abs(x[i]) for x in errs)
    print(f"{label:<34} prefill {mae(0):5.1f}%  decode {mae(1):5.1f}%  e2e {mae(2):5.1f}%   "
          f"(e2e per shape: {', '.join(f'{x[2]:+.1f}' for x in errs)})")
    return {"label": label, "prefill": mae(0), "decode": mae(1), "e2e": mae(2), "per_shape": errs}


def main():
    pp1 = load("lab-runs/2026-09-28-static-batch/real_pp1.jsonl")
    # Linear rescale: prefill time is inversely proportional to compute_efficiency
    # and decode (memory bound) to decode_memory_bandwidth_scale.
    def fit(rows):
        pre = [sim("7b", r["batch"], r["prompt"], "prefill", 1, 1, 1, 1.0, 1.0) / r["prefill_ms"] for r in rows]
        dec = [sim("7b", r["batch"], r["prompt"], "decode", r["decode"], 1, 1, 1.0, 1.0) / r["decode"]
               / r["decode_ms_per_step"] for r in rows]
        return round(statistics.median(pre), 4), round(statistics.median(dec), 4)

    ce, bw = fit(pp1)
    print(f"fitted on 7B PP=1 (5 shapes): compute_efficiency={ce} (of 88 TFLOPs) decode_memory_bandwidth_scale={bw}")
    loo = []
    for i, held in enumerate(pp1):
        ce_i, bw_i = fit(pp1[:i] + pp1[i + 1:])
        loo.append(evaluate(f"  LOO hold {held['batch']}x{held['prompt']}", "7b", [held], 1, 1, ce_i, bw_i)["e2e"])
    print(f"7B PP=1 leave-one-out mean |e2e| = {statistics.mean(loo):.1f}%")

    results = [
        evaluate("7B PP=1 (in-sample)", "7b", pp1, 1, 1, ce, bw),
        evaluate("7B long context 4k-16k (node1)", "7b",
                 load("lab-runs/2026-09-30-qwen7b-static-longctx-node1/measured.jsonl"), 1, 1, ce, bw),
        evaluate("7B PP=2", "7b", load("lab-runs/2026-09-28-static-batch/real_pp2.jsonl"), 1, 2, ce, bw),
        evaluate("7B TP=2", "7b", load("lab-runs/2026-09-30-tp2/real_tp2.jsonl"), 2, 1, ce, bw),
        evaluate("14B PP=2 (model transfer)", "14b",
                 load("lab-runs/2026-09-30-qwen14b-static-pp2/measured.jsonl"), 1, 2, ce, bw),
    ]
    json.dump({"compute_efficiency": ce, "decode_memory_bandwidth_scale": bw, "peak_f16_tflops": 88.0,
               "collective_latency_scale": LATENCY_SCALE, "loo_e2e_mean": statistics.mean(loo),
               "results": results}, open(HERE / "recalibration.json", "w"), indent=2)


if __name__ == "__main__":
    main()
