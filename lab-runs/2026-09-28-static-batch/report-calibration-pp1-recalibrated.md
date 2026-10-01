## Calibration: pp1-recalibrated

Spec `tools/lab/specs/rtx3090_qwen7b_static_pp1.toml`, measurements `lab-runs/2026-09-28-static-batch/real_pp1.jsonl`.

- `compute_efficiency` = 0.35 x median(sim/real prefill) = **0.8494** (ratios: 1x512 2.065, 1x2048 2.423, 8x512 2.427, 8x2048 2.501, 32x512 2.508)
- `decode_memory_bandwidth_scale` = 1 x median(sim/real decode step) = **0.8383** (ratios: 1x512 0.846, 1x2048 0.848, 8x512 0.838, 8x2048 0.834, 32x512 0.838)
- leave-one-shape-out folds (held-out shape: compute_efficiency / decode_memory_bandwidth_scale): 1x512: 0.8623 / 0.8381; 1x2048: 0.8623 / 0.8381; 8x512: 0.8617 / 0.8421; 8x2048: 0.8487 / 0.8424; 32x512: 0.8487 / 0.8424
- `lab-runs/2026-09-28-static-batch/calibration_profile-pp1-recalibrated.toml` loads in the simulator as profile `rtx3090-lab-vllm-qwen2.5-7b-instruct-tp1-pp1` (applicability `within_valid_shape`), reporting compute_efficiency 0.849357 and decode_memory_bandwidth_scale 0.838309; predictions through the profile drift at most 0.0000% from the in-sample scalar run.

### pp1-recalibrated default calibration

Calibration: `default`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 114.7 | 236.8 | +106.5% | 19.26 | 16.31 | -15.4% | 2,562.2 | 2,324.0 | -9.3% |
| 1x2048 | 399.4 | 967.6 | +142.3% | 19.34 | 16.40 | -15.2% | 2,858.3 | 3,066.9 | +7.3% |
| 8x512 | 780.6 | 1,894.3 | +142.7% | 19.75 | 16.55 | -16.2% | 3,288.8 | 4,013.2 | +22.0% |
| 8x2048 | 3,095.5 | 7,741.1 | +150.1% | 20.76 | 17.31 | -16.6% | 5,721.0 | 9,956.4 | +74.0% |
| 32x512 | 3,021.1 | 7,577.1 | +150.8% | 20.77 | 17.40 | -16.2% | 5,648.6 | 9,804.5 | +73.6% |
| **mean \|err\|** | | | **138.5%** | | | **15.9%** | | | **37.2%** |

### pp1-recalibrated fitted (in-sample)

Calibration: `ce0.8494-dmbs0.8383`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 114.7 | 97.6 | -14.9% | 19.26 | 19.45 | +1.0% | 2,562.2 | 2,587.4 | +1.0% |
| 1x2048 | 399.4 | 398.7 | -0.2% | 19.34 | 19.56 | +1.1% | 2,858.3 | 2,903.0 | +1.6% |
| 8x512 | 780.6 | 780.6 | -0.0% | 19.75 | 19.75 | +0.0% | 3,288.8 | 3,308.2 | +0.6% |
| 8x2048 | 3,095.5 | 3,189.9 | +3.0% | 20.76 | 20.64 | -0.6% | 5,721.0 | 5,832.5 | +1.9% |
| 32x512 | 3,021.1 | 3,122.4 | +3.4% | 20.77 | 20.76 | -0.1% | 5,648.6 | 5,779.4 | +2.3% |
| **mean \|err\|** | | | **4.3%** | | | **0.5%** | | | **1.5%** |

### pp1-recalibrated leave-one-shape-out

Calibration: `per-fold scalars`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 114.7 | 96.1 | -16.2% | 19.26 | 19.46 | +1.0% | 2,562.2 | 2,586.7 | +1.0% |
| 1x2048 | 399.4 | 392.8 | -1.7% | 19.34 | 19.57 | +1.2% | 2,858.3 | 2,897.7 | +1.4% |
| 8x512 | 780.6 | 769.5 | -1.4% | 19.75 | 19.66 | -0.5% | 3,288.8 | 3,285.5 | -0.1% |
| 8x2048 | 3,095.5 | 3,192.4 | +3.1% | 20.76 | 20.54 | -1.0% | 5,721.0 | 5,822.1 | +1.8% |
| 32x512 | 3,021.1 | 3,124.8 | +3.4% | 20.77 | 20.66 | -0.5% | 5,648.6 | 5,768.9 | +2.1% |
| **mean \|err\|** | | | **5.2%** | | | **0.8%** | | | **1.3%** |
