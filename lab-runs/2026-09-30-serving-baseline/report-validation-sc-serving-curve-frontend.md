## Validation: sc-serving-curve-frontend

Measurements `lab-runs/2026-09-30-serving-baseline/rate_*.json`.

### sc-serving-curve-frontend: qwen7b-serving-pp1 under profile-calibration_profile-curve

Calibration: `profile-calibration_profile-curve`. Ref batch = the serving model's `[request].batch_size`, which the iteration engine does not use (see the feature doc). `n/a` = the simulator rejected the candidate and reported no latency metrics.

| rate req/s | ref batch (sim status) | metric | real median | sim median | err | real p99 | sim p99 | err |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 (ok) | TTFT | 162.3 | 137.2 | -15.5% | 277.9 | 258.7 | -6.9% |
| 1 | 1 (ok) | TPOT | 21.8 | 21.8 | +0.3% | 27.1 | 26.4 | -2.4% |
| 1 | 1 (ok) | ITL | 19.4 | 19.6 | +1.1% | 126.6 | 115.3 | -9.0% |
| 1 | 1 (ok) | E2EL | 2,911.1 | 2,908.8 | -0.1% | 3,641.3 | 3,483.8 | -4.3% |
| 1 | 1 (ok) | output tok/s | 126.3 | 135.9 | +7.6% | | | |
| 2 | 1 (ok) | TTFT | 168.0 | 139.0 | -17.3% | 384.8 | 390.2 | +1.4% |
| 2 | 1 (ok) | TPOT | 24.7 | 24.4 | -1.1% | 32.7 | 32.5 | -0.8% |
| 2 | 1 (ok) | ITL | 19.8 | 19.7 | -0.2% | 129.5 | 116.5 | -10.1% |
| 2 | 1 (ok) | E2EL | 3,308.8 | 3,239.8 | -2.1% | 4,322.5 | 4,262.8 | -1.4% |
| 2 | 1 (ok) | output tok/s | 249.4 | 268.1 | +7.5% | | | |
| 4 | 1 (ok) | TTFT | 193.4 | 155.9 | -19.4% | 574.1 | 455.4 | -20.7% |
| 4 | 1 (ok) | TPOT | 35.9 | 33.7 | -6.1% | 48.6 | 45.4 | -6.6% |
| 4 | 1 (ok) | ITL | 20.3 | 20.3 | +0.0% | 241.4 | 202.5 | -16.1% |
| 4 | 1 (ok) | E2EL | 4,783.0 | 4,528.4 | -5.3% | 6,421.3 | 5,973.0 | -7.0% |
| 4 | 1 (ok) | output tok/s | 483.7 | 520.9 | +7.7% | | | |
| 6 | 1 (ok) | TTFT | 467.8 | 298.9 | -36.1% | 1,686.9 | 1,199.9 | -28.9% |
| 6 | 1 (ok) | TPOT | 62.6 | 52.7 | -15.8% | 79.9 | 70.2 | -12.1% |
| 6 | 1 (ok) | ITL | 22.5 | 21.6 | -3.9% | 342.1 | 306.6 | -10.4% |
| 6 | 1 (ok) | E2EL | 8,634.8 | 7,327.1 | -15.1% | 11,067.7 | 9,698.4 | -12.4% |
| 6 | 1 (ok) | output tok/s | 700.9 | 756.9 | +8.0% | | | |
| 8 | 1 (ok) | TTFT | 3,702.9 | 2,804.8 | -24.3% | 6,323.2 | 4,552.4 | -28.0% |
| 8 | 1 (ok) | TPOT | 76.4 | 66.7 | -12.7% | 80.5 | 69.5 | -13.7% |
| 8 | 1 (ok) | ITL | 23.3 | 22.1 | -5.0% | 405.4 | 391.6 | -3.4% |
| 8 | 1 (ok) | E2EL | 11,688.2 | 9,535.3 | -18.4% | 15,105.7 | 13,292.4 | -12.0% |
| 8 | 1 (ok) | output tok/s | 746.3 | 846.8 | +13.5% | | | |
| 10 | 1 (ok) | TTFT | 5,107.8 | 4,709.1 | -7.8% | 10,598.0 | 9,074.9 | -14.4% |
| 10 | 1 (ok) | TPOT | 72.9 | 66.2 | -9.2% | 77.7 | 69.3 | -10.9% |
| 10 | 1 (ok) | ITL | 23.3 | 22.1 | -4.9% | 405.7 | 391.7 | -3.5% |
| 10 | 1 (ok) | E2EL | 13,537.2 | 12,715.1 | -6.1% | 17,360.3 | 16,028.1 | -7.7% |
| 10 | 1 (ok) | output tok/s | 768.1 | 849.2 | +10.6% | | | |
| inf | 1 (ok) | TTFT | 13,207.2 | 12,691.8 | -3.9% | 28,599.1 | 27,502.1 | -3.8% |
| inf | 1 (ok) | TPOT | 65.7 | 65.9 | +0.3% | 70.9 | 67.8 | -4.4% |
| inf | 1 (ok) | ITL | 23.3 | 22.1 | -5.2% | 405.7 | 391.7 | -3.5% |
| inf | 1 (ok) | E2EL | 21,660.6 | 21,129.4 | -2.5% | 31,246.0 | 30,026.1 | -3.9% |
| inf | 1 (ok) | output tok/s | 818.3 | 852.6 | +4.2% | | | |

Mean |error| per metric across rates:

| metric | mean \|err\| |
|---|---:|
| ttft median | 17.7% |
| ttft p99 | 14.9% |
| tpot median | 6.5% |
| tpot p99 | 7.3% |
| itl median | 2.9% |
| itl p99 | 8.0% |
| e2el median | 7.1% |
| e2el p99 | 6.9% |
| output tok/s | 8.4% |
