## Validation: sc-14b-pp2-floor

Measurements `lab-runs/2026-09-30-qwen14b-static-pp2/measured.jsonl`.

### sc-14b-pp2-floor: qwen14b-static-pp2 under profile-profile-scalar-net

Calibration: `profile-profile-scalar-net`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 239.0 | 205.1 | -14.2% | 34.97 | 37.87 | +8.3% | 4,675.0 | 5,052.9 | +8.1% |
| 1x2048 | 913.4 | 840.9 | -7.9% | 35.34 | 38.26 | +8.2% | 5,432.9 | 5,737.9 | +5.6% |
| 8x512 | 1,362.2 | 1,640.3 | +20.4% | 43.11 | 39.04 | -9.4% | 6,804.7 | 6,637.8 | -2.5% |
| 8x2048 | 5,545.3 | 6,726.5 | +21.3% | 43.49 | 42.12 | -3.1% | 11,366.5 | 12,118.2 | +6.6% |
| 32x512 | 6,372.2 | 6,561.0 | +3.0% | 46.15 | 43.05 | -6.7% | 12,265.2 | 12,072.0 | -1.6% |
| **mean \|err\|** | | | **13.4%** | | | **7.2%** | | | **4.9%** |
