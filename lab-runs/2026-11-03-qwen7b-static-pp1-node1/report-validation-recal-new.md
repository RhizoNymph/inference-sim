## Validation: recal-new

Measurements `lab-runs/2026-11-03-qwen7b-static-pp1-node1/measured.jsonl`.

### recal-new: qwen7b-static-pp1-node1 under profile-calibration_profile-node1-nov

Calibration: `profile-calibration_profile-node1-nov`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 73.1 | 102.5 | +40.3% | 18.46 | 18.65 | +1.0% | 2,420.9 | 2,489.2 | +2.8% |
| 1x2048 | 262.9 | 261.1 | -0.7% | 18.58 | 18.75 | +0.9% | 2,786.4 | 2,661.6 | -4.5% |
| 8x512 | 510.2 | 503.9 | -1.2% | 18.96 | 18.93 | -0.2% | 2,972.3 | 2,926.8 | -1.5% |
| 8x2048 | 1,998.6 | 2,059.2 | +3.0% | 19.79 | 19.79 | -0.0% | 4,694.5 | 4,592.3 | -2.2% |
| 32x512 | 1,949.1 | 2,015.6 | +3.4% | 21.28 | 19.90 | -6.5% | 4,748.8 | 4,562.5 | -3.9% |
| **mean \|err\|** | | | **9.7%** | | | **1.7%** | | | **3.0%** |
