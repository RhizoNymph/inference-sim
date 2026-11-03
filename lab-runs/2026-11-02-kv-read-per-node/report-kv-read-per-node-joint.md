## KV-read bandwidth scale: per-node-joint

Mode `joint`; base profile `lab-runs/2026-10-01-qwen7b-prefill-token-sweep/calibration_profile-curve.toml`; profile `/home/nymph/Code/ai/inference-sim/lab-degraded-network-node0-sweep/lab-runs/2026-11-02-kv-read-per-node/calibration_profile-per-node-joint.toml` (loaded and checked by the simulator).

Fitted on all datasets (36 decode steps): `kv_read_bandwidth_scale = 0.3522`, `decode_memory_bandwidth_scale = 0.8579` (base 0.8383). Re-simulated steps match the linear decomposition within 0.0000%.

| dataset | held-out fit kv scale | decode step before | fitted (all) | held out | trend before | trend fitted | trend held out | e2e before | e2e held out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| node1-nov | 0.5808 | 10.2% | 8.6% | 10.9% | -12.48 | -8.31 | -10.81 | 4.1% | 5.2% |
| node0-nov | 0.2443 | 5.4% | 7.0% | 11.6% | -0.42 | +4.82 | +8.68 | 19.1% | 23.3% |
| node1-oct | 0.3257 | 2.7% | 1.5% | 2.0% | -3.23 | +1.51 | +2.22 | 3.0% | 2.9% |

Trend: OLS slope of the signed decode-step error, in percentage points per 10k KV tokens per step. Held out: scales fitted on the other datasets only.

### node1-nov

Measurements `/home/nymph/Code/ai/inference-sim/lab-degraded-network-node0-sweep/lab-runs/2026-11-02-qwen7b-decode-batch-sweep/measured.jsonl`; excluded (batch above the max): 40x512, 48x512, 56x512, 64x512.

| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 576 | 18.46 | 19.45 | +5.3% | 19.07 | +3.3% | 18.91 | +2.4% |
| 2x512 | 1,153 | 20.51 | 19.49 | -4.9% | 19.17 | -6.5% | 18.97 | -7.5% |
| 3x512 | 1,730 | 20.45 | 19.54 | -4.5% | 19.27 | -5.8% | 19.03 | -6.9% |
| 4x512 | 2,306 | 18.77 | 19.58 | +4.3% | 19.37 | +3.2% | 19.09 | +1.7% |
| 6x512 | 3,459 | 20.48 | 19.66 | -4.0% | 19.57 | -4.4% | 19.21 | -6.2% |
| 8x512 | 4,612 | 21.91 | 19.75 | -9.9% | 19.77 | -9.8% | 19.33 | -11.7% |
| 12x512 | 6,918 | 21.88 | 19.92 | -9.0% | 20.17 | -7.8% | 19.58 | -10.5% |
| 16x512 | 9,224 | 22.92 | 20.08 | -12.4% | 20.57 | -10.3% | 19.82 | -13.5% |
| 20x512 | 11,530 | 22.79 | 20.25 | -11.1% | 20.97 | -8.0% | 20.06 | -11.9% |
| 24x512 | 13,836 | 23.83 | 20.42 | -14.3% | 21.37 | -10.3% | 20.31 | -14.8% |
| 28x512 | 16,142 | 26.33 | 20.59 | -21.8% | 21.77 | -17.3% | 20.55 | -21.9% |
| 32x512 | 18,448 | 26.44 | 20.76 | -21.5% | 22.18 | -16.1% | 20.79 | -21.3% |

### node0-nov

Measurements `/home/nymph/Code/ai/inference-sim/lab-degraded-network-node0-sweep/lab-runs/2026-11-02-qwen7b-decode-batch-sweep-node0/measured.jsonl`; excluded (batch above the max): 40x512, 48x512, 56x512, 64x512.

| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 576 | 18.22 | 19.45 | +6.7% | 19.07 | +4.6% | 19.41 | +6.5% |
| 2x512 | 1,153 | 18.12 | 19.49 | +7.6% | 19.17 | +5.8% | 19.55 | +7.9% |
| 3x512 | 1,730 | 18.42 | 19.54 | +6.0% | 19.27 | +4.6% | 19.70 | +6.9% |
| 4x512 | 2,306 | 18.44 | 19.58 | +6.1% | 19.37 | +5.0% | 19.84 | +7.6% |
| 6x512 | 3,459 | 18.74 | 19.66 | +4.9% | 19.57 | +4.4% | 20.13 | +7.4% |
| 8x512 | 4,612 | 18.80 | 19.75 | +5.1% | 19.77 | +5.2% | 20.42 | +8.6% |
| 12x512 | 6,918 | 19.35 | 19.92 | +2.9% | 20.17 | +4.2% | 21.00 | +8.5% |
| 16x512 | 9,224 | 19.49 | 20.08 | +3.0% | 20.57 | +5.5% | 21.58 | +10.7% |
| 20x512 | 11,530 | 19.27 | 20.25 | +5.1% | 20.97 | +8.8% | 22.16 | +15.0% |
| 24x512 | 13,836 | 19.36 | 20.42 | +5.5% | 21.37 | +10.4% | 22.73 | +17.5% |
| 28x512 | 16,142 | 19.43 | 20.59 | +6.0% | 21.77 | +12.1% | 23.31 | +20.0% |
| 32x512 | 18,448 | 19.56 | 20.76 | +6.1% | 22.18 | +13.4% | 23.89 | +22.1% |

### node1-oct

Measurements `lab-runs/2026-10-01-qwen7b-decode-batch-sweep/measured.jsonl`; excluded (batch above the max): 40x512.

| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 576 | 19.58 | 19.45 | -0.7% | 19.07 | -2.6% | 18.87 | -3.6% |
| 2x512 | 1,153 | 19.25 | 19.49 | +1.3% | 19.17 | -0.4% | 18.98 | -1.4% |
| 3x512 | 1,730 | 19.75 | 19.54 | -1.1% | 19.27 | -2.5% | 19.09 | -3.4% |
| 4x512 | 2,306 | 19.62 | 19.58 | -0.2% | 19.37 | -1.3% | 19.20 | -2.1% |
| 6x512 | 3,459 | 20.06 | 19.66 | -2.0% | 19.57 | -2.5% | 19.41 | -3.2% |
| 8x512 | 4,612 | 20.08 | 19.75 | -1.7% | 19.77 | -1.6% | 19.63 | -2.2% |
| 12x512 | 6,918 | 20.64 | 19.92 | -3.5% | 20.17 | -2.3% | 20.06 | -2.8% |
| 16x512 | 9,224 | 20.97 | 20.08 | -4.2% | 20.57 | -1.9% | 20.50 | -2.2% |
| 20x512 | 11,530 | 20.79 | 20.25 | -2.6% | 20.97 | +0.9% | 20.93 | +0.7% |
| 24x512 | 13,836 | 20.95 | 20.42 | -2.5% | 21.37 | +2.0% | 21.36 | +2.0% |
| 28x512 | 16,142 | 21.89 | 20.59 | -5.9% | 21.77 | -0.5% | 21.80 | -0.4% |
| 32x512 | 18,448 | 22.21 | 20.76 | -6.5% | 22.18 | -0.1% | 22.23 | +0.1% |
