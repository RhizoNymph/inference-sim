## KV-read bandwidth scale: after

Mode `after`; base profile `lab-runs/2026-10-01-qwen7b-prefill-token-sweep/calibration_profile-curve.toml`; profile `lab-runs/2026-10-01-kv-read-scale/calibration_profile-after.toml` (loaded and checked by the simulator).

Fitted on all datasets (17 decode steps): `kv_read_bandwidth_scale = 0.4687`, `decode_memory_bandwidth_scale = 0.8383` (base 0.8383). Re-simulated steps match the linear decomposition within 0.0000%.

| dataset | held-out fit kv scale | decode step before | fitted (all) | held out | trend before | trend fitted | trend held out | e2e before | e2e held out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch-sweep | 0.5277 | 2.7% | 1.1% | 1.3% | -3.23 | -0.61 | -1.27 | 3.0% | 2.1% |
| long-context | 0.4285 | 2.5% | 0.9% | 1.6% | -2.26 | +0.35 | +0.91 | 3.1% | 1.1% |

Trend: OLS slope of the signed decode-step error, in percentage points per 10k KV tokens per step. Held out: scales fitted on the other datasets only.

### batch-sweep

Measurements `lab-runs/2026-10-01-qwen7b-decode-batch-sweep/measured.jsonl`; excluded (batch above the max): 40x512.

| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 576 | 19.58 | 19.45 | -0.7% | 19.49 | -0.5% | 19.48 | -0.5% |
| 2x512 | 1,153 | 19.25 | 19.49 | +1.3% | 19.56 | +1.6% | 19.54 | +1.5% |
| 3x512 | 1,730 | 19.75 | 19.54 | -1.1% | 19.64 | -0.6% | 19.61 | -0.7% |
| 4x512 | 2,306 | 19.62 | 19.58 | -0.2% | 19.71 | +0.5% | 19.68 | +0.3% |
| 6x512 | 3,459 | 20.06 | 19.66 | -2.0% | 19.86 | -1.0% | 19.81 | -1.2% |
| 8x512 | 4,612 | 20.08 | 19.75 | -1.7% | 20.01 | -0.3% | 19.95 | -0.7% |
| 12x512 | 6,918 | 20.64 | 19.92 | -3.5% | 20.31 | -1.6% | 20.21 | -2.1% |
| 16x512 | 9,224 | 20.97 | 20.08 | -4.2% | 20.62 | -1.7% | 20.48 | -2.3% |
| 20x512 | 11,530 | 20.79 | 20.25 | -2.6% | 20.92 | +0.6% | 20.75 | -0.2% |
| 24x512 | 13,836 | 20.95 | 20.42 | -2.5% | 21.22 | +1.3% | 21.02 | +0.3% |
| 28x512 | 16,142 | 21.89 | 20.59 | -5.9% | 21.52 | -1.7% | 21.28 | -2.7% |
| 32x512 | 18,448 | 22.21 | 20.76 | -6.5% | 21.82 | -1.7% | 21.55 | -3.0% |

### long-context

Measurements `lab-runs/2026-09-30-qwen7b-static-longctx-node1/measured.jsonl`; excluded (batch above the max): none.

| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x4096 | 4,160 | 19.89 | 19.71 | -0.9% | 19.95 | +0.3% | 20.00 | +0.6% |
| 1x8192 | 8,256 | 20.25 | 20.01 | -1.2% | 20.49 | +1.2% | 20.59 | +1.7% |
| 1x16384 | 16,448 | 21.20 | 20.61 | -2.8% | 21.56 | +1.7% | 21.76 | +2.6% |
| 2x8192 | 16,513 | 21.28 | 20.62 | -3.1% | 21.57 | +1.4% | 21.77 | +2.3% |
| 4x4096 | 16,642 | 21.58 | 20.63 | -4.4% | 21.58 | +0.0% | 21.79 | +1.0% |
