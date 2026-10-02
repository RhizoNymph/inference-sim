## Validation: sc-14b-pp2-curve

Measurements `lab-runs/2026-09-30-qwen14b-static-pp2/measured.jsonl`.

### sc-14b-pp2-curve: qwen14b-static-pp2 under profile-profile-curve-net

Calibration: `profile-profile-curve-net`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 239.0 | 238.5 | -0.2% | 34.97 | 37.87 | +8.3% | 4,675.0 | 5,086.2 | +8.8% |
| 1x2048 | 913.4 | 843.4 | -7.7% | 35.34 | 38.26 | +8.2% | 5,432.9 | 5,740.5 | +5.7% |
| 8x512 | 1,362.2 | 1,596.8 | +17.2% | 43.11 | 39.04 | -9.4% | 6,804.7 | 6,594.4 | -3.1% |
| 8x2048 | 5,545.3 | 6,548.0 | +18.1% | 43.49 | 42.12 | -3.1% | 11,366.5 | 11,939.6 | +5.0% |
| 32x512 | 6,372.2 | 6,387.1 | +0.2% | 46.15 | 43.05 | -6.7% | 12,265.2 | 11,898.1 | -3.0% |
| **mean \|err\|** | | | **8.7%** | | | **7.2%** | | | **5.1%** |
