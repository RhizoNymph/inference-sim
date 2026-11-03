## Validation: po-pp2-after

Measurements `lab-runs/2026-09-28-static-batch/real_pp2.jsonl`.

### po-pp2-after: qwen7b-static-pp2 under profile-profile-curve-net-overlap

Calibration: `profile-profile-curve-net-overlap`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 131.2 | 122.6 | -6.6% | 19.40 | 19.53 | +0.6% | 2,603.1 | 2,622.0 | +0.7% |
| 1x2048 | 453.4 | 432.6 | -4.6% | 19.54 | 19.64 | +0.5% | 2,932.3 | 2,946.4 | +0.5% |
| 8x512 | 664.7 | 790.2 | +18.9% | 22.39 | 19.93 | -11.0% | 3,470.1 | 3,341.6 | -3.7% |
| 8x2048 | 2,893.7 | 3,173.1 | +9.7% | 21.37 | 20.83 | -2.5% | 5,985.7 | 5,839.5 | -2.4% |
| 32x512 | 3,313.9 | 3,257.4 | -1.7% | 22.29 | 21.33 | -4.3% | 5,922.6 | 5,987.0 | +1.1% |
| **mean \|err\|** | | | **8.3%** | | | **3.8%** | | | **1.7%** |
