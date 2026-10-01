## Calibration: pp1

Spec `tools/lab/specs/rtx3090_qwen7b_static_pp1.toml`, measurements `lab-runs/2026-09-28-static-batch/real_pp1.jsonl`.

- `compute_efficiency` = 0.35 x median(sim/real prefill) = **0.8507** (ratios: 1x512 2.061, 1x2048 2.431, 8x512 2.422, 8x2048 2.509, 32x512 2.503)
- `decode_memory_bandwidth_scale` = 1 x median(sim/real decode step) = **0.8383** (ratios: 1x512 0.846, 1x2048 0.848, 8x512 0.838, 8x2048 0.834, 32x512 0.838)
- leave-one-shape-out folds (held-out shape: compute_efficiency / decode_memory_bandwidth_scale): 1x512: 0.8634 / 0.8381; 1x2048: 0.8619 / 0.8381; 8x512: 0.8634 / 0.8421; 8x2048: 0.8492 / 0.8424; 32x512: 0.8492 / 0.8424
- `lab-runs/2026-09-28-static-batch/calibration_profile-pp1.toml` loads in the simulator as profile `rtx3090-lab-vllm-qwen2.5-7b-instruct-tp1-pp1` (applicability `within_valid_shape`), reporting compute_efficiency 0.850699 and decode_memory_bandwidth_scale 0.838309; predictions through the profile drift at most 0.0000% from the in-sample scalar run.

### pp1 default calibration

Calibration: `default`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 114.7 | 236.3 | +106.1% | 19.26 | 16.31 | -15.4% | 2,562.2 | 2,323.6 | -9.3% |
| 1x2048 | 399.4 | 970.7 | +143.1% | 19.34 | 16.40 | -15.2% | 2,858.3 | 3,070.0 | +7.4% |
| 8x512 | 780.6 | 1,890.5 | +142.2% | 19.75 | 16.55 | -16.2% | 3,288.8 | 4,009.4 | +21.9% |
| 8x2048 | 3,095.5 | 7,765.4 | +150.9% | 20.76 | 17.31 | -16.6% | 5,721.0 | 9,980.6 | +74.5% |
| 32x512 | 3,021.1 | 7,562.1 | +150.3% | 20.77 | 17.40 | -16.2% | 5,648.6 | 9,789.5 | +73.3% |
| **mean \|err\|** | | | **138.5%** | | | **15.9%** | | | **37.3%** |

### pp1 fitted (in-sample)

Calibration: `ce0.8507-dmbs0.8383`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 114.7 | 97.2 | -15.2% | 19.26 | 19.45 | +1.0% | 2,562.2 | 2,587.1 | +1.0% |
| 1x2048 | 399.4 | 399.4 | -0.0% | 19.34 | 19.56 | +1.1% | 2,858.3 | 2,903.6 | +1.6% |
| 8x512 | 780.6 | 777.8 | -0.4% | 19.75 | 19.75 | +0.0% | 3,288.8 | 3,305.4 | +0.5% |
| 8x2048 | 3,095.5 | 3,194.9 | +3.2% | 20.76 | 20.64 | -0.6% | 5,721.0 | 5,837.4 | +2.0% |
| 32x512 | 3,021.1 | 3,111.2 | +3.0% | 20.77 | 20.76 | -0.1% | 5,648.6 | 5,768.3 | +2.1% |
| **mean \|err\|** | | | **4.4%** | | | **0.5%** | | | **1.4%** |

### pp1 leave-one-shape-out

Calibration: `per-fold scalars`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 114.7 | 95.8 | -16.5% | 19.26 | 19.46 | +1.0% | 2,562.2 | 2,586.4 | +0.9% |
| 1x2048 | 399.4 | 394.2 | -1.3% | 19.34 | 19.57 | +1.2% | 2,858.3 | 2,899.1 | +1.4% |
| 8x512 | 780.6 | 766.4 | -1.8% | 19.75 | 19.66 | -0.5% | 3,288.8 | 3,282.4 | -0.2% |
| 8x2048 | 3,095.5 | 3,200.6 | +3.4% | 20.76 | 20.54 | -1.0% | 5,721.0 | 5,830.3 | +1.9% |
| 32x512 | 3,021.1 | 3,116.8 | +3.2% | 20.77 | 20.66 | -0.5% | 5,648.6 | 5,760.9 | +2.0% |
| **mean \|err\|** | | | **5.2%** | | | **0.8%** | | | **1.3%** |
