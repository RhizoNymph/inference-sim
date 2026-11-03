## KV-read bandwidth scale: after-with-node0

Mode `after`; base profile `lab-runs/2026-10-01-qwen7b-prefill-token-sweep/calibration_profile-curve.toml`; profile `lab-runs/2026-10-01-kv-read-scale/calibration_profile-after-with-node0.toml` (loaded and checked by the simulator).

Fitted on all datasets (22 decode steps): `kv_read_bandwidth_scale = 0.5258`, `decode_memory_bandwidth_scale = 0.8383` (base 0.8383). Re-simulated steps match the linear decomposition within 0.0000%.

| dataset | held-out fit kv scale | decode step before | fitted (all) | held out | trend before | trend fitted | trend held out | e2e before | e2e held out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch-sweep | 0.6214 | 2.7% | 1.3% | 1.8% | -3.23 | -1.25 | -2.07 | 3.0% | 2.5% |
| long-context | 0.5248 | 2.5% | 0.5% | 0.5% | -2.26 | -0.29 | -0.28 | 3.1% | 1.8% |
| pp1-node0 | 0.4687 | 0.5% | 2.1% | 2.7% | -0.71 | +1.37 | +2.05 | 0.9% | 2.1% |

Trend: OLS slope of the signed decode-step error, in percentage points per 10k KV tokens per step. Held out: scales fitted on the other datasets only.

### batch-sweep

Measurements `lab-runs/2026-10-01-qwen7b-decode-batch-sweep/measured.jsonl`; excluded (batch above the max): 40x512.

| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 576 | 19.58 | 19.45 | -0.7% | 19.48 | -0.5% | 19.47 | -0.6% |
| 2x512 | 1,153 | 19.25 | 19.49 | +1.3% | 19.54 | +1.5% | 19.52 | +1.4% |
| 3x512 | 1,730 | 19.75 | 19.54 | -1.1% | 19.61 | -0.7% | 19.58 | -0.9% |
| 4x512 | 2,306 | 19.62 | 19.58 | -0.2% | 19.68 | +0.3% | 19.64 | +0.1% |
| 6x512 | 3,459 | 20.06 | 19.66 | -2.0% | 19.81 | -1.2% | 19.75 | -1.5% |
| 8x512 | 4,612 | 20.08 | 19.75 | -1.7% | 19.95 | -0.7% | 19.86 | -1.1% |
| 12x512 | 6,918 | 20.64 | 19.92 | -3.5% | 20.22 | -2.1% | 20.09 | -2.7% |
| 16x512 | 9,224 | 20.97 | 20.08 | -4.2% | 20.48 | -2.3% | 20.32 | -3.1% |
| 20x512 | 11,530 | 20.79 | 20.25 | -2.6% | 20.75 | -0.2% | 20.55 | -1.2% |
| 24x512 | 13,836 | 20.95 | 20.42 | -2.5% | 21.02 | +0.3% | 20.77 | -0.9% |
| 28x512 | 16,142 | 21.89 | 20.59 | -5.9% | 21.29 | -2.7% | 21.00 | -4.0% |
| 32x512 | 18,448 | 22.21 | 20.76 | -6.5% | 21.56 | -2.9% | 21.23 | -4.4% |

### long-context

Measurements `lab-runs/2026-09-30-qwen7b-static-longctx-node1/measured.jsonl`; excluded (batch above the max): none.

| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x4096 | 4,160 | 19.89 | 19.71 | -0.9% | 19.89 | +0.0% | 19.90 | +0.1% |
| 1x8192 | 8,256 | 20.25 | 20.01 | -1.2% | 20.37 | +0.6% | 20.37 | +0.6% |
| 1x16384 | 16,448 | 21.20 | 20.61 | -2.8% | 21.33 | +0.6% | 21.33 | +0.6% |
| 2x8192 | 16,513 | 21.28 | 20.62 | -3.1% | 21.33 | +0.3% | 21.34 | +0.3% |
| 4x4096 | 16,642 | 21.58 | 20.63 | -4.4% | 21.35 | -1.1% | 21.35 | -1.1% |

### pp1-node0

Measurements `lab-runs/2026-09-28-static-batch/real_pp1.jsonl`; excluded (batch above the max): none.

| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 576 | 19.26 | 19.45 | +1.0% | 19.48 | +1.1% | 19.49 | +1.1% |
| 1x2048 | 2,112 | 19.34 | 19.56 | +1.1% | 19.66 | +1.6% | 19.69 | +1.8% |
| 8x512 | 4,612 | 19.75 | 19.75 | +0.0% | 19.95 | +1.0% | 20.01 | +1.3% |
| 8x2048 | 16,900 | 20.76 | 20.64 | -0.6% | 21.38 | +3.0% | 21.62 | +4.1% |
| 32x512 | 18,448 | 20.77 | 20.76 | -0.1% | 21.56 | +3.8% | 21.82 | +5.1% |
