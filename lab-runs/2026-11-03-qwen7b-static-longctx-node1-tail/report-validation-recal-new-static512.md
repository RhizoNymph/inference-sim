## Validation: recal-new-static512

Measurements `lab-runs/2026-11-03-qwen7b-static-longctx-node1-tail/measured.jsonl`.

### recal-new-static512: qwen7b-static-longctx-node1-tail under profile-calibration_profile-node1-nov-static512

Calibration: `profile-calibration_profile-node1-nov-static512`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x16384 | 2,511.0 | 2,466.4 | -1.8% | 20.58 | 19.76 | -4.0% | 5,286.5 | 4,995.4 | -5.5% |
| 2x8192 | 2,231.4 | 2,233.7 | +0.1% | 21.52 | 19.76 | -8.2% | 5,124.0 | 4,763.3 | -7.0% |
| 4x4096 | 2,103.2 | 2,117.4 | +0.7% | 22.05 | 19.77 | -10.3% | 4,975.3 | 4,648.1 | -6.6% |
| **mean \|err\|** | | | **0.9%** | | | **7.5%** | | | **6.4%** |
