## Validation: sc-longctx-curve

Measurements `lab-runs/2026-09-30-qwen7b-static-longctx-node1/measured.jsonl`.

### sc-longctx-curve: qwen7b-static-longctx-node1 under profile-calibration_profile-curve-only

Calibration: `profile-calibration_profile-curve-only`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x4096 | 824.5 | 797.0 | -3.3% | 19.89 | 19.71 | -0.9% | 3,363.5 | 3,320.3 | -1.3% |
| 1x8192 | 1,752.9 | 1,681.5 | -4.1% | 20.25 | 20.01 | -1.2% | 4,345.6 | 4,243.2 | -2.4% |
| 1x16384 | 3,940.2 | 3,713.3 | -5.8% | 21.20 | 20.61 | -2.8% | 6,647.1 | 6,351.6 | -4.4% |
| 2x8192 | 3,502.0 | 3,363.0 | -4.0% | 21.28 | 20.62 | -3.1% | 6,208.0 | 6,001.9 | -3.3% |
| 4x4096 | 3,315.5 | 3,187.9 | -3.8% | 21.58 | 20.63 | -4.4% | 6,062.7 | 5,828.0 | -3.9% |
| **mean \|err\|** | | | **4.2%** | | | **2.5%** | | | **3.1%** |
