## Validation: sc-pp1-before

Measurements `lab-runs/2026-09-28-static-batch/real_pp1.jsonl`.

### sc-pp1-before: qwen7b-static-pp1 under profile-calibration_profile-pp1-recalibrated

Calibration: `profile-calibration_profile-pp1-recalibrated`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 114.7 | 97.6 | -14.9% | 19.26 | 19.45 | +1.0% | 2,562.2 | 2,587.4 | +1.0% |
| 1x2048 | 399.4 | 398.7 | -0.2% | 19.34 | 19.56 | +1.1% | 2,858.3 | 2,903.0 | +1.6% |
| 8x512 | 780.6 | 780.6 | -0.0% | 19.75 | 19.75 | +0.0% | 3,288.8 | 3,308.2 | +0.6% |
| 8x2048 | 3,095.5 | 3,189.9 | +3.0% | 20.76 | 20.64 | -0.6% | 5,721.0 | 5,832.5 | +1.9% |
| 32x512 | 3,021.1 | 3,122.4 | +3.4% | 20.77 | 20.76 | -0.1% | 5,648.6 | 5,779.4 | +2.3% |
| **mean \|err\|** | | | **4.3%** | | | **0.5%** | | | **1.5%** |
