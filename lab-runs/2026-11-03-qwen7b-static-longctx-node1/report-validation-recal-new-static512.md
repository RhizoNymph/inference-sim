## Validation: recal-new-static512

Measurements `lab-runs/2026-11-03-qwen7b-static-longctx-node1/measured.jsonl`.

### recal-new-static512: qwen7b-static-longctx-node1 under profile-calibration_profile-node1-nov-static512

Calibration: `profile-calibration_profile-node1-nov-static512`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x4096 | 521.6 | 529.3 | +1.5% | 18.78 | 18.90 | +0.6% | 3,309.2 | 2,948.2 | -10.9% |
| 1x8192 | 1,117.6 | 1,116.9 | -0.1% | 19.43 | 19.18 | -1.3% | 3,802.4 | 3,572.4 | -6.0% |
| **mean \|err\|** | | | **0.8%** | | | **0.9%** | | | **8.5%** |
