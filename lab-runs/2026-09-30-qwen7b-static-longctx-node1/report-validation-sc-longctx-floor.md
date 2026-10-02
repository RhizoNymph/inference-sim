## Validation: sc-longctx-floor

Measurements `lab-runs/2026-09-30-qwen7b-static-longctx-node1/measured.jsonl`.

### sc-longctx-floor: qwen7b-static-longctx-node1 under profile-calibration_profile-pp1-recalibrated

Calibration: `profile-calibration_profile-pp1-recalibrated`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x4096 | 824.5 | 820.0 | -0.5% | 19.89 | 19.71 | -0.9% | 3,363.5 | 3,343.4 | -0.6% |
| 1x8192 | 1,752.9 | 1,730.1 | -1.3% | 20.25 | 20.01 | -1.2% | 4,345.6 | 4,291.8 | -1.2% |
| 1x16384 | 3,940.2 | 3,820.7 | -3.0% | 21.20 | 20.61 | -2.8% | 6,647.1 | 6,459.0 | -2.8% |
| 2x8192 | 3,502.0 | 3,460.2 | -1.2% | 21.28 | 20.62 | -3.1% | 6,208.0 | 6,099.2 | -1.8% |
| 4x4096 | 3,315.5 | 3,280.0 | -1.1% | 21.58 | 20.63 | -4.4% | 6,062.7 | 5,920.2 | -2.4% |
| **mean \|err\|** | | | **1.4%** | | | **2.5%** | | | **1.8%** |
