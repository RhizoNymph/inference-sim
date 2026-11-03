## Calibration: node1-nov-gemm-peak

Spec `tools/lab/specs/rtx3090_qwen7b_static_pp1_node1.toml`, measurements `lab-runs/2026-11-03-qwen7b-static-pp1-node1/measured.jsonl`.

- `compute_efficiency` = 0.35 x median(sim/real prefill) = **0.8360** (ratios: 1x512 2.085, 1x2048 2.367, 8x512 2.389, 8x2048 2.492, 32x512 2.501)
- `decode_memory_bandwidth_scale` = 1 x median(sim/real decode step) = **0.8745** (ratios: 1x512 0.883, 1x2048 0.883, 8x512 0.873, 8x2048 0.875, 32x512 0.818)
- leave-one-shape-out folds (held-out shape: compute_efficiency / decode_memory_bandwidth_scale): 1x512: 0.8540 / 0.8738; 1x2048: 0.8540 / 0.8738; 8x512: 0.8503 / 0.8787; 8x2048: 0.8323 / 0.8779; 32x512: 0.8323 / 0.8787
- `lab-runs/2026-11-03-qwen7b-static-pp1-node1/calibration_profile-node1-nov-gemm-peak.toml` loads in the simulator as profile `rtx3090-lab-vllm-qwen2.5-7b-instruct-tp1-pp1` (applicability `within_valid_shape`), reporting compute_efficiency 0.83601 and decode_memory_bandwidth_scale 0.874542; predictions through the profile drift at most 0.0000% from the in-sample scalar run.

### node1-nov-gemm-peak default calibration

Calibration: `default`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 73.1 | 152.3 | +108.5% | 18.46 | 16.31 | -11.7% | 2,420.9 | 2,239.6 | -7.5% |
| 1x2048 | 262.9 | 622.5 | +136.7% | 18.58 | 16.40 | -11.7% | 2,786.4 | 2,721.8 | -2.3% |
| 8x512 | 510.2 | 1,218.5 | +138.9% | 18.96 | 16.55 | -12.7% | 2,972.3 | 3,337.4 | +12.3% |
| 8x2048 | 1,998.6 | 4,979.7 | +149.2% | 19.79 | 17.31 | -12.5% | 4,694.5 | 7,194.9 | +53.3% |
| 32x512 | 1,949.1 | 4,874.2 | +150.1% | 21.28 | 17.40 | -18.2% | 4,748.8 | 7,101.6 | +49.5% |
| **mean \|err\|** | | | **136.7%** | | | **13.4%** | | | **25.0%** |

### node1-nov-gemm-peak fitted (in-sample)

Calibration: `ce0.8360-dmbs0.8745`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 73.1 | 63.8 | -12.7% | 18.46 | 18.65 | +1.0% | 2,420.9 | 2,450.5 | +1.2% |
| 1x2048 | 262.9 | 260.6 | -0.9% | 18.58 | 18.75 | +0.9% | 2,786.4 | 2,661.1 | -4.5% |
| 8x512 | 510.2 | 510.2 | -0.0% | 18.96 | 18.93 | -0.2% | 2,972.3 | 2,933.0 | -1.3% |
| 8x2048 | 1,998.6 | 2,084.8 | +4.3% | 19.79 | 19.79 | -0.0% | 4,694.5 | 4,617.8 | -1.6% |
| 32x512 | 1,949.1 | 2,040.6 | +4.7% | 21.28 | 19.90 | -6.5% | 4,748.8 | 4,587.5 | -3.4% |
| **mean \|err\|** | | | **4.5%** | | | **1.7%** | | | **2.4%** |

### node1-nov-gemm-peak leave-one-shape-out

Calibration: `per-fold scalars`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 73.1 | 62.4 | -14.6% | 18.46 | 18.66 | +1.1% | 2,420.9 | 2,451.3 | +1.3% |
| 1x2048 | 262.9 | 255.1 | -3.0% | 18.58 | 18.77 | +1.0% | 2,786.4 | 2,657.7 | -4.6% |
| 8x512 | 510.2 | 501.6 | -1.7% | 18.96 | 18.84 | -0.7% | 2,972.3 | 2,913.0 | -2.0% |
| 8x2048 | 1,998.6 | 2,094.1 | +4.8% | 19.79 | 19.71 | -0.4% | 4,694.5 | 4,617.5 | -1.6% |
| 32x512 | 1,949.1 | 2,049.7 | +5.2% | 21.28 | 19.80 | -6.9% | 4,748.8 | 4,584.7 | -3.5% |
| **mean \|err\|** | | | **5.8%** | | | **2.0%** | | | **2.6%** |
