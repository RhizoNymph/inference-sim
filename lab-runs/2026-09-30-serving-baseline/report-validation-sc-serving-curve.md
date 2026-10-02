## Validation: sc-serving-curve

Measurements `lab-runs/2026-09-30-serving-baseline/rate_*.json`.

### sc-serving-curve: qwen7b-serving-pp1 under profile-calibration_profile-curve-only

Calibration: `profile-calibration_profile-curve-only`. Ref batch = the serving model's `[request].batch_size`, which the iteration engine does not use (see the feature doc). `n/a` = the simulator rejected the candidate and reported no latency metrics.

| rate req/s | ref batch (sim status) | metric | real median | sim median | err | real p99 | sim p99 | err |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 (ok) | TTFT | 162.3 | 124.7 | -23.2% | 277.9 | 246.2 | -11.4% |
| 1 | 1 (ok) | TPOT | 21.8 | 21.8 | +0.3% | 27.1 | 26.4 | -2.4% |
| 1 | 1 (ok) | ITL | 19.4 | 19.6 | +1.1% | 126.6 | 115.3 | -9.0% |
| 1 | 1 (ok) | E2EL | 2,911.1 | 2,896.3 | -0.5% | 3,641.3 | 3,471.3 | -4.7% |
| 1 | 1 (ok) | output tok/s | 126.3 | 135.9 | +7.6% | | | |
| 2 | 1 (ok) | TTFT | 168.0 | 126.6 | -24.7% | 384.8 | 377.7 | -1.8% |
| 2 | 1 (ok) | TPOT | 24.7 | 24.4 | -1.1% | 32.7 | 32.5 | -0.8% |
| 2 | 1 (ok) | ITL | 19.8 | 19.7 | -0.2% | 129.5 | 116.5 | -10.1% |
| 2 | 1 (ok) | E2EL | 3,308.8 | 3,227.3 | -2.5% | 4,322.5 | 4,250.3 | -1.7% |
| 2 | 1 (ok) | output tok/s | 249.4 | 268.1 | +7.5% | | | |
| 4 | 1 (ok) | TTFT | 193.4 | 143.5 | -25.8% | 574.1 | 442.9 | -22.8% |
| 4 | 1 (ok) | TPOT | 35.9 | 33.7 | -6.1% | 48.6 | 45.4 | -6.6% |
| 4 | 1 (ok) | ITL | 20.3 | 20.3 | +0.0% | 241.4 | 202.5 | -16.1% |
| 4 | 1 (ok) | E2EL | 4,783.0 | 4,516.0 | -5.6% | 6,421.3 | 5,960.5 | -7.2% |
| 4 | 1 (ok) | output tok/s | 483.7 | 521.0 | +7.7% | | | |
| 6 | 1 (ok) | TTFT | 467.8 | 286.4 | -38.8% | 1,686.9 | 1,187.4 | -29.6% |
| 6 | 1 (ok) | TPOT | 62.6 | 52.7 | -15.8% | 79.9 | 70.2 | -12.1% |
| 6 | 1 (ok) | ITL | 22.5 | 21.6 | -3.9% | 342.1 | 306.6 | -10.4% |
| 6 | 1 (ok) | E2EL | 8,634.8 | 7,314.6 | -15.3% | 11,067.7 | 9,685.9 | -12.5% |
| 6 | 1 (ok) | output tok/s | 700.9 | 757.2 | +8.0% | | | |
| 8 | 1 (ok) | TTFT | 3,702.9 | 2,792.3 | -24.6% | 6,323.2 | 4,539.9 | -28.2% |
| 8 | 1 (ok) | TPOT | 76.4 | 66.7 | -12.7% | 80.5 | 69.5 | -13.7% |
| 8 | 1 (ok) | ITL | 23.3 | 22.1 | -5.0% | 405.4 | 391.6 | -3.4% |
| 8 | 1 (ok) | E2EL | 11,688.2 | 9,522.8 | -18.5% | 15,105.7 | 13,279.9 | -12.1% |
| 8 | 1 (ok) | output tok/s | 746.3 | 847.2 | +13.5% | | | |
| 10 | 1 (ok) | TTFT | 5,107.8 | 4,696.6 | -8.0% | 10,598.0 | 9,062.4 | -14.5% |
| 10 | 1 (ok) | TPOT | 72.9 | 66.2 | -9.2% | 77.7 | 69.3 | -10.9% |
| 10 | 1 (ok) | ITL | 23.3 | 22.1 | -4.9% | 405.7 | 391.7 | -3.5% |
| 10 | 1 (ok) | E2EL | 13,537.2 | 12,702.7 | -6.2% | 17,360.3 | 16,015.6 | -7.7% |
| 10 | 1 (ok) | output tok/s | 768.1 | 849.6 | +10.6% | | | |
| inf | 1 (ok) | TTFT | 13,207.2 | 12,679.3 | -4.0% | 28,599.1 | 27,489.6 | -3.9% |
| inf | 1 (ok) | TPOT | 65.7 | 65.9 | +0.3% | 70.9 | 67.8 | -4.4% |
| inf | 1 (ok) | ITL | 23.3 | 22.1 | -5.2% | 405.7 | 391.7 | -3.5% |
| inf | 1 (ok) | E2EL | 21,660.6 | 21,116.9 | -2.5% | 31,246.0 | 30,013.6 | -3.9% |
| inf | 1 (ok) | output tok/s | 818.3 | 852.9 | +4.2% | | | |

Mean |error| per metric across rates:

| metric | mean \|err\| |
|---|---:|
| ttft median | 21.3% |
| ttft p99 | 16.0% |
| tpot median | 6.5% |
| tpot p99 | 7.3% |
| itl median | 2.9% |
| itl p99 | 8.0% |
| e2el median | 7.3% |
| e2el p99 | 7.1% |
| output tok/s | 8.5% |
