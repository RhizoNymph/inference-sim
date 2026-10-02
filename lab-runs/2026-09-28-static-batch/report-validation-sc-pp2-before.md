## Validation: sc-pp2-before

Measurements `lab-runs/2026-09-28-static-batch/real_pp2.jsonl`.

### sc-pp2-before: qwen7b-static-pp2 under profile-profile-scalar-net

Calibration: `profile-profile-scalar-net`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 131.2 | 105.8 | -19.4% | 19.40 | 19.53 | +0.6% | 2,603.1 | 2,605.2 | +0.1% |
| 1x2048 | 453.4 | 431.3 | -4.9% | 19.54 | 19.64 | +0.5% | 2,932.3 | 2,945.1 | +0.4% |
| 8x512 | 664.7 | 845.7 | +27.2% | 22.39 | 19.93 | -11.0% | 3,470.1 | 3,397.1 | -2.1% |
| 8x2048 | 2,893.7 | 3,450.3 | +19.2% | 21.37 | 20.83 | -2.5% | 5,985.7 | 6,116.6 | +2.2% |
| 32x512 | 3,313.9 | 3,382.7 | +2.1% | 22.29 | 21.33 | -4.3% | 5,922.6 | 6,112.3 | +3.2% |
| **mean \|err\|** | | | **14.6%** | | | **3.8%** | | | **1.6%** |
