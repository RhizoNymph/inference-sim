## Validation: sc-serving-floor

Measurements `lab-runs/2026-09-30-serving-baseline/rate_*.json`.

### sc-serving-floor: qwen7b-serving-pp1 under profile-calibration_profile-pp1-recalibrated

Calibration: `profile-calibration_profile-pp1-recalibrated`. Ref batch = the serving model's `[request].batch_size`, which the iteration engine does not use (see the feature doc). `n/a` = the simulator rejected the candidate and reported no latency metrics.

| rate req/s | ref batch (sim status) | metric | real median | sim median | err | real p99 | sim p99 | err |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 (ok) | TTFT | 162.3 | 109.0 | -32.8% | 277.9 | 210.3 | -24.3% |
| 1 | 1 (ok) | TPOT | 21.8 | 21.4 | -1.6% | 27.1 | 24.7 | -8.9% |
| 1 | 1 (ok) | ITL | 19.4 | 19.6 | +1.0% | 126.6 | 98.5 | -22.2% |
| 1 | 1 (ok) | E2EL | 2,911.1 | 2,832.1 | -2.7% | 3,641.3 | 3,248.3 | -10.8% |
| 1 | 1 (ok) | output tok/s | 126.3 | 135.9 | +7.6% | | | |
| 2 | 1 (ok) | TTFT | 168.0 | 109.9 | -34.6% | 384.8 | 276.2 | -28.2% |
| 2 | 1 (ok) | TPOT | 24.7 | 23.5 | -4.7% | 32.7 | 29.0 | -11.3% |
| 2 | 1 (ok) | ITL | 19.8 | 19.7 | -0.2% | 129.5 | 99.7 | -23.0% |
| 2 | 1 (ok) | E2EL | 3,308.8 | 3,100.7 | -6.3% | 4,322.5 | 3,805.1 | -12.0% |
| 2 | 1 (ok) | output tok/s | 249.4 | 268.1 | +7.5% | | | |
| 4 | 1 (ok) | TTFT | 193.4 | 118.4 | -38.8% | 574.1 | 393.0 | -31.5% |
| 4 | 1 (ok) | TPOT | 35.9 | 31.0 | -13.6% | 48.6 | 39.4 | -19.0% |
| 4 | 1 (ok) | ITL | 20.3 | 20.2 | -0.4% | 241.4 | 198.8 | -17.6% |
| 4 | 1 (ok) | E2EL | 4,783.0 | 4,069.2 | -14.9% | 6,421.3 | 5,108.3 | -20.4% |
| 4 | 1 (ok) | output tok/s | 483.7 | 521.6 | +7.8% | | | |
| 6 | 1 (ok) | TTFT | 467.8 | 220.3 | -52.9% | 1,686.9 | 740.4 | -56.1% |
| 6 | 1 (ok) | TPOT | 62.6 | 43.4 | -30.7% | 79.9 | 66.6 | -16.7% |
| 6 | 1 (ok) | ITL | 22.5 | 21.2 | -6.0% | 342.1 | 303.7 | -11.2% |
| 6 | 1 (ok) | E2EL | 8,634.8 | 5,766.3 | -33.2% | 11,067.7 | 8,666.1 | -21.7% |
| 6 | 1 (ok) | output tok/s | 700.9 | 758.3 | +8.2% | | | |
| 8 | 1 (ok) | TTFT | 3,702.9 | 2,467.6 | -33.4% | 6,323.2 | 4,068.0 | -35.7% |
| 8 | 1 (ok) | TPOT | 76.4 | 65.4 | -14.5% | 80.5 | 68.2 | -15.3% |
| 8 | 1 (ok) | ITL | 23.3 | 22.1 | -5.0% | 405.4 | 390.4 | -3.7% |
| 8 | 1 (ok) | E2EL | 11,688.2 | 9,262.7 | -20.8% | 15,105.7 | 12,733.6 | -15.7% |
| 8 | 1 (ok) | output tok/s | 746.3 | 863.1 | +15.7% | | | |
| 10 | 1 (ok) | TTFT | 5,107.8 | 4,514.1 | -11.6% | 10,598.0 | 8,712.1 | -17.8% |
| 10 | 1 (ok) | TPOT | 72.9 | 65.5 | -10.2% | 77.7 | 68.8 | -11.6% |
| 10 | 1 (ok) | ITL | 23.3 | 22.1 | -4.9% | 405.7 | 390.4 | -3.8% |
| 10 | 1 (ok) | E2EL | 13,537.2 | 12,463.4 | -7.9% | 17,360.3 | 15,694.6 | -9.6% |
| 10 | 1 (ok) | output tok/s | 768.1 | 858.6 | +11.8% | | | |
| inf | 1 (ok) | TTFT | 13,207.2 | 12,629.2 | -4.4% | 28,599.1 | 27,399.2 | -4.2% |
| inf | 1 (ok) | TPOT | 65.7 | 65.6 | -0.1% | 70.9 | 67.6 | -4.7% |
| inf | 1 (ok) | ITL | 23.3 | 22.1 | -5.2% | 405.7 | 390.4 | -3.8% |
| inf | 1 (ok) | E2EL | 21,660.6 | 21,043.9 | -2.8% | 31,246.0 | 29,923.2 | -4.2% |
| inf | 1 (ok) | output tok/s | 818.3 | 855.5 | +4.6% | | | |

Mean |error| per metric across rates:

| metric | mean \|err\| |
|---|---:|
| ttft median | 29.8% |
| ttft p99 | 28.3% |
| tpot median | 10.8% |
| tpot p99 | 12.5% |
| itl median | 3.3% |
| itl p99 | 12.2% |
| e2el median | 12.7% |
| e2el p99 | 13.5% |
| output tok/s | 9.0% |
