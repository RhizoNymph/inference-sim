## Validation: recal-old

Measurements `lab-runs/2026-11-03-qwen7b-static-longctx-node1-tail/measured.jsonl`.

### recal-old: qwen7b-static-longctx-node1-tail under profile-calibration_profile-curve-only

Calibration: `profile-calibration_profile-curve-only`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x16384 | 2,511.0 | 3,713.3 | +47.9% | 20.58 | 20.61 | +0.2% | 5,286.5 | 6,351.6 | +20.1% |
| 2x8192 | 2,231.4 | 3,363.0 | +50.7% | 21.52 | 20.62 | -4.2% | 5,124.0 | 6,001.9 | +17.1% |
| 4x4096 | 2,103.2 | 3,187.9 | +51.6% | 22.05 | 20.63 | -6.5% | 4,975.3 | 5,828.0 | +17.1% |
| **mean \|err\|** | | | **50.1%** | | | **3.6%** | | | **18.1%** |
