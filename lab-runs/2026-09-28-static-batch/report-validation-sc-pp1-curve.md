## Validation: sc-pp1-curve

Measurements `lab-runs/2026-09-28-static-batch/real_pp1.jsonl`.

### sc-pp1-curve: qwen7b-static-pp1 under profile-calibration_profile-curve-only

Calibration: `profile-calibration_profile-curve-only`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 114.7 | 114.4 | -0.2% | 19.26 | 19.45 | +1.0% | 2,562.2 | 2,604.2 | +1.6% |
| 1x2048 | 399.4 | 400.0 | +0.2% | 19.34 | 19.56 | +1.1% | 2,858.3 | 2,904.2 | +1.6% |
| 8x512 | 780.6 | 758.7 | -2.8% | 19.75 | 19.75 | +0.0% | 3,288.8 | 3,286.2 | -0.1% |
| 8x2048 | 3,095.5 | 3,100.3 | +0.2% | 20.76 | 20.64 | -0.6% | 5,721.0 | 5,742.8 | +0.4% |
| 32x512 | 3,021.1 | 3,034.6 | +0.4% | 20.77 | 20.76 | -0.1% | 5,648.6 | 5,691.6 | +0.8% |
| **mean \|err\|** | | | **0.8%** | | | **0.5%** | | | **0.9%** |
