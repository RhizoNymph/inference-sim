## Validation: recal-old

Measurements `lab-runs/2026-11-02-qwen7b-static-pp1/measured.jsonl`.

### recal-old: qwen7b-static-pp1 under profile-calibration_profile-curve-only

Calibration: `profile-calibration_profile-curve-only`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 66.1 | 114.4 | +73.0% | 18.23 | 19.45 | +6.7% | 2,382.1 | 2,604.2 | +9.3% |
| 1x2048 | 239.9 | 400.0 | +66.8% | 18.33 | 19.56 | +6.7% | 2,568.9 | 2,904.2 | +13.1% |
| 8x512 | 465.9 | 758.7 | +62.8% | 18.80 | 19.75 | +5.0% | 2,852.4 | 3,286.2 | +15.2% |
| 8x2048 | 1,823.3 | 3,100.3 | +70.0% | 19.62 | 20.64 | +5.2% | 4,316.2 | 5,742.8 | +33.1% |
| 32x512 | 1,771.8 | 3,034.6 | +71.3% | 19.56 | 20.76 | +6.1% | 4,256.4 | 5,691.6 | +33.7% |
| **mean \|err\|** | | | **68.8%** | | | **6.0%** | | | **20.9%** |
