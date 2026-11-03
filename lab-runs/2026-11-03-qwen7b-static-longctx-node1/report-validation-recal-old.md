## Validation: recal-old

Measurements `lab-runs/2026-11-03-qwen7b-static-longctx-node1/measured.jsonl`.

### recal-old: qwen7b-static-longctx-node1 under profile-calibration_profile-curve-only

Calibration: `profile-calibration_profile-curve-only`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x4096 | 521.6 | 797.0 | +52.8% | 18.78 | 19.71 | +5.0% | 3,309.2 | 3,320.3 | +0.3% |
| 1x8192 | 1,117.6 | 1,681.5 | +50.5% | 19.43 | 20.01 | +3.0% | 3,802.4 | 4,243.2 | +11.6% |
| **mean \|err\|** | | | **51.6%** | | | **4.0%** | | | **6.0%** |
