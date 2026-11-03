## KV-read bandwidth scale: same-day-joint

Mode `joint`; base profile `lab-runs/2026-10-01-qwen7b-prefill-token-sweep/calibration_profile-curve.toml`; profile `/home/nymph/Code/ai/inference-sim/lab-degraded-network-node0-sweep/lab-runs/2026-11-02-kv-read-per-node/calibration_profile-same-day-joint.toml` (loaded and checked by the simulator).

Fitted on all datasets (24 decode steps): `kv_read_bandwidth_scale = 0.3257`, `decode_memory_bandwidth_scale = 0.8673` (base 0.8383). Re-simulated steps match the linear decomposition within 0.0001%.

| dataset | held-out fit kv scale | decode step before | fitted (all) | held out | trend before | trend fitted | trend held out | e2e before | e2e held out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| node1-nov | 0.7899 | 10.2% | 8.7% | 13.4% | -12.48 | -7.62 | -11.48 | 4.1% | 5.9% |
| node0-nov | 0.1543 | 5.4% | 6.5% | 16.0% | -0.42 | +5.58 | +16.19 | 19.1% | 26.3% |

Trend: OLS slope of the signed decode-step error, in percentage points per 10k KV tokens per step. Held out: scales fitted on the other datasets only.

### node1-nov

Measurements `/home/nymph/Code/ai/inference-sim/lab-degraded-network-node0-sweep/lab-runs/2026-11-02-qwen7b-decode-batch-sweep/measured.jsonl`; excluded (batch above the max): 40x512, 48x512, 56x512, 64x512.

| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 576 | 18.46 | 19.45 | +5.3% | 18.87 | +2.2% | 18.39 | -0.4% |
| 2x512 | 1,153 | 20.51 | 19.49 | -4.9% | 18.98 | -7.5% | 18.44 | -10.1% |
| 3x512 | 1,730 | 20.45 | 19.54 | -4.5% | 19.09 | -6.7% | 18.48 | -9.6% |
| 4x512 | 2,306 | 18.77 | 19.58 | +4.3% | 19.20 | +2.3% | 18.53 | -1.3% |
| 6x512 | 3,459 | 20.48 | 19.66 | -4.0% | 19.41 | -5.2% | 18.62 | -9.1% |
| 8x512 | 4,612 | 21.91 | 19.75 | -9.9% | 19.63 | -10.4% | 18.70 | -14.6% |
| 12x512 | 6,918 | 21.88 | 19.92 | -9.0% | 20.06 | -8.3% | 18.88 | -13.7% |
| 16x512 | 9,224 | 22.92 | 20.08 | -12.4% | 20.50 | -10.6% | 19.06 | -16.8% |
| 20x512 | 11,530 | 22.79 | 20.25 | -11.1% | 20.93 | -8.1% | 19.24 | -15.6% |
| 24x512 | 13,836 | 23.83 | 20.42 | -14.3% | 21.36 | -10.4% | 19.42 | -18.5% |
| 28x512 | 16,142 | 26.33 | 20.59 | -21.8% | 21.80 | -17.2% | 19.60 | -25.6% |
| 32x512 | 18,448 | 26.44 | 20.76 | -21.5% | 22.23 | -15.9% | 19.78 | -25.2% |

### node0-nov

Measurements `/home/nymph/Code/ai/inference-sim/lab-degraded-network-node0-sweep/lab-runs/2026-11-02-qwen7b-decode-batch-sweep-node0/measured.jsonl`; excluded (batch above the max): 40x512, 48x512, 56x512, 64x512.

| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 576 | 18.22 | 19.45 | +6.7% | 18.87 | +3.6% | 19.26 | +5.7% |
| 2x512 | 1,153 | 18.12 | 19.49 | +7.6% | 18.98 | +4.7% | 19.49 | +7.6% |
| 3x512 | 1,730 | 18.42 | 19.54 | +6.0% | 19.09 | +3.6% | 19.72 | +7.1% |
| 4x512 | 2,306 | 18.44 | 19.58 | +6.1% | 19.20 | +4.1% | 19.95 | +8.2% |
| 6x512 | 3,459 | 18.74 | 19.66 | +4.9% | 19.41 | +3.6% | 20.41 | +8.9% |
| 8x512 | 4,612 | 18.80 | 19.75 | +5.1% | 19.63 | +4.4% | 20.87 | +11.0% |
| 12x512 | 6,918 | 19.35 | 19.92 | +2.9% | 20.06 | +3.7% | 21.78 | +12.6% |
| 16x512 | 9,224 | 19.49 | 20.08 | +3.0% | 20.50 | +5.1% | 22.70 | +16.4% |
| 20x512 | 11,530 | 19.27 | 20.25 | +5.1% | 20.93 | +8.6% | 23.61 | +22.5% |
| 24x512 | 13,836 | 19.36 | 20.42 | +5.5% | 21.36 | +10.4% | 24.53 | +26.7% |
| 28x512 | 16,142 | 19.43 | 20.59 | +6.0% | 21.80 | +12.2% | 25.44 | +31.0% |
| 32x512 | 18,448 | 19.56 | 20.76 | +6.1% | 22.23 | +13.7% | 26.36 | +34.8% |
