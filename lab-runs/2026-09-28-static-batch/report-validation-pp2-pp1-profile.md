## Validation: pp2-pp1-profile

Measurements `lab-runs/2026-09-28-static-batch/real_pp2.jsonl`.

### pp2-pp1-profile: qwen7b-static-pp2 under profile-calibration_profile-pp1

Calibration: `profile-calibration_profile-pp1`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 131.2 | 100.4 | -23.5% | 19.40 | 19.49 | +0.4% | 2,603.1 | 2,595.1 | -0.3% |
| 1x2048 | 453.4 | 411.9 | -9.2% | 19.54 | 19.60 | +0.3% | 2,932.3 | 2,921.0 | -0.4% |
| 8x512 | 664.7 | 802.8 | +20.8% | 22.39 | 19.83 | -11.4% | 3,470.1 | 3,340.8 | -3.7% |
| 8x2048 | 2,893.7 | 3,294.7 | +13.9% | 21.37 | 20.73 | -3.0% | 5,985.7 | 5,947.7 | -0.6% |
| 32x512 | 3,313.9 | 3,211.1 | -3.1% | 22.29 | 20.99 | -5.9% | 5,922.6 | 5,897.2 | -0.4% |
| **mean \|err\|** | | | **14.1%** | | | **4.2%** | | | **1.1%** |
