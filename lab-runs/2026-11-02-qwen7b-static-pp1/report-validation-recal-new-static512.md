## Validation: recal-new-static512

Measurements `lab-runs/2026-11-02-qwen7b-static-pp1/measured.jsonl`.

### recal-new-static512: qwen7b-static-pp1 under profile-calibration_profile-node1-nov-static512

Calibration: `profile-calibration_profile-node1-nov-static512`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 66.1 | 73.1 | +10.5% | 18.23 | 18.65 | +2.3% | 2,382.1 | 2,459.8 | +3.3% |
| 1x2048 | 239.9 | 261.1 | +8.9% | 18.33 | 18.75 | +2.3% | 2,568.9 | 2,661.6 | +3.6% |
| 8x512 | 465.9 | 503.9 | +8.2% | 18.80 | 18.93 | +0.7% | 2,852.4 | 2,926.8 | +2.6% |
| 8x2048 | 1,823.3 | 2,059.2 | +12.9% | 19.62 | 19.79 | +0.9% | 4,316.2 | 4,592.3 | +6.4% |
| 32x512 | 1,771.8 | 2,015.6 | +13.8% | 19.56 | 19.90 | +1.7% | 4,256.4 | 4,562.5 | +7.2% |
| **mean \|err\|** | | | **10.8%** | | | **1.6%** | | | **4.6%** |
