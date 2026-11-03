## Validation: recal-old

Measurements `lab-runs/2026-11-03-qwen7b-static-pp1-node1/measured.jsonl`.

### recal-old: qwen7b-static-pp1-node1 under profile-calibration_profile-curve-only

Calibration: `profile-calibration_profile-curve-only`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 73.1 | 114.4 | +56.6% | 18.46 | 19.45 | +5.4% | 2,420.9 | 2,604.2 | +7.6% |
| 1x2048 | 262.9 | 400.0 | +52.1% | 18.58 | 19.56 | +5.3% | 2,786.4 | 2,904.2 | +4.2% |
| 8x512 | 510.2 | 758.7 | +48.7% | 18.96 | 19.75 | +4.1% | 2,972.3 | 3,286.2 | +10.6% |
| 8x2048 | 1,998.6 | 3,100.3 | +55.1% | 19.79 | 20.64 | +4.3% | 4,694.5 | 5,742.8 | +22.3% |
| 32x512 | 1,949.1 | 3,034.6 | +55.7% | 21.28 | 20.76 | -2.5% | 4,748.8 | 5,691.6 | +19.9% |
| **mean \|err\|** | | | **53.6%** | | | **4.3%** | | | **12.9%** |
