## Validation: kv-14b-pp2-kv

Measurements `lab-runs/2026-09-30-qwen14b-static-pp2/measured.jsonl`.

### kv-14b-pp2-kv: qwen14b-static-pp2 under profile-profile-kv-net

Calibration: `profile-profile-kv-net`

| shape | prefill real ms | prefill sim ms | err | decode step real ms | decode step sim ms | err | e2e real ms | e2e sim ms | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x512 | 239.0 | 238.5 | -0.2% | 34.97 | 37.99 | +8.6% | 4,675.0 | 5,100.8 | +9.1% |
| 1x2048 | 913.4 | 843.4 | -7.7% | 35.34 | 38.68 | +9.4% | 5,432.9 | 5,793.9 | +6.6% |
| 8x512 | 1,362.2 | 1,596.8 | +17.2% | 43.11 | 39.95 | -7.3% | 6,804.7 | 6,711.0 | -1.4% |
| 8x2048 | 5,545.3 | 6,548.0 | +18.1% | 43.49 | 45.46 | +4.5% | 11,366.5 | 12,367.0 | +8.8% |
| 32x512 | 6,372.2 | 6,387.1 | +0.2% | 46.15 | 46.70 | +1.2% | 12,265.2 | 12,364.6 | +0.8% |
| **mean \|err\|** | | | **8.7%** | | | **6.2%** | | | **5.3%** |
