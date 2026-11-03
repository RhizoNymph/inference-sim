## Validation: po-14b-pp2-after

Measurements `lab-runs/2026-09-30-qwen14b-static-pp2/measured.jsonl`.

### po-14b-pp2-after: qwen14b-static-pp2 under profile-profile-curve-net-overlap

Calibration: `profile-profile-curve-net-overlap`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 239.0 | 238.5 | -0.2% | 34.97 | 37.87 | +8.3% | 4,675.0 | 5,086.2 | +8.8% |
| 1x2048 | 913.4 | 843.4 | -7.7% | 35.34 | 38.26 | +8.2% | 5,432.9 | 5,740.5 | +5.7% |
| 8x512 | 1,362.2 | 1,530.3 | +12.3% | 43.11 | 39.04 | -9.4% | 6,804.7 | 6,527.8 | -4.1% |
| 8x2048 | 5,545.3 | 6,174.4 | +11.3% | 43.49 | 42.12 | -3.1% | 11,366.5 | 11,566.1 | +1.8% |
| 32x512 | 6,372.2 | 6,312.6 | -0.9% | 46.15 | 43.05 | -6.7% | 12,265.2 | 11,823.6 | -3.6% |
| **mean \|err\|** | | | **6.5%** | | | **7.2%** | | | **4.8%** |
