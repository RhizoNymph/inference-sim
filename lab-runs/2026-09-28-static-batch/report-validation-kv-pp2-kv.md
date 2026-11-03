## Validation: kv-pp2-kv

Measurements `lab-runs/2026-09-28-static-batch/real_pp2.jsonl`.

### kv-pp2-kv: qwen7b-static-pp2 under profile-profile-kv-net

Calibration: `profile-profile-kv-net`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 131.2 | 122.6 | -6.6% | 19.40 | 19.56 | +0.8% | 2,603.1 | 2,626.3 | +0.9% |
| 1x2048 | 453.4 | 432.6 | -4.6% | 19.54 | 19.76 | +1.1% | 2,932.3 | 2,962.0 | +1.0% |
| 8x512 | 664.7 | 823.8 | +23.9% | 22.39 | 20.20 | -9.8% | 3,470.1 | 3,409.2 | -1.8% |
| 8x2048 | 2,893.7 | 3,360.6 | +16.1% | 21.37 | 21.80 | +2.0% | 5,985.7 | 6,151.6 | +2.8% |
| 32x512 | 3,313.9 | 3,294.9 | -0.6% | 22.29 | 22.39 | +0.4% | 5,922.6 | 6,160.7 | +4.0% |
| **mean \|err\|** | | | **10.4%** | | | **2.8%** | | | **2.1%** |
