## Validation: kv-pp1-kv

Measurements `lab-runs/2026-09-28-static-batch/real_pp1.jsonl`.

### kv-pp1-kv: qwen7b-static-pp1 under profile-calibration_profile-after

Calibration: `profile-calibration_profile-after`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 114.7 | 114.4 | -0.2% | 19.26 | 19.49 | +1.1% | 2,562.2 | 2,608.5 | +1.8% |
| 1x2048 | 399.4 | 400.0 | +0.2% | 19.34 | 19.69 | +1.8% | 2,858.3 | 2,919.8 | +2.2% |
| 8x512 | 780.6 | 758.7 | -2.8% | 19.75 | 20.01 | +1.3% | 3,288.8 | 3,320.3 | +1.0% |
| 8x2048 | 3,095.5 | 3,100.3 | +0.2% | 20.76 | 21.62 | +4.1% | 5,721.0 | 5,867.5 | +2.6% |
| 32x512 | 3,021.1 | 3,034.6 | +0.4% | 20.77 | 21.82 | +5.1% | 5,648.6 | 5,827.7 | +3.2% |
| **mean \|err\|** | | | **0.8%** | | | **2.7%** | | | **2.1%** |
