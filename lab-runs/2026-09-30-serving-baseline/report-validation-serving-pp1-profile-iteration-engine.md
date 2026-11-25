## Validation: serving-pp1-profile-iteration-engine

Measurements `lab-runs/2026-09-30-serving-baseline/rate_*.json`.

### serving-pp1-profile-iteration-engine: qwen7b-serving-pp1 under profile-calibration_profile-pp1

Calibration: `profile-calibration_profile-pp1`. Ref batch = the serving model's `[request].batch_size`, which the iteration engine does not use (see the feature doc). `n/a` = the simulator rejected the candidate and reported no latency metrics.

| rate req/s | ref batch (sim status) | metric | real median | sim median | err | real p99 | sim p99 | err |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 (ok) | TTFT | 162.3 | 108.4 | -33.2% | 277.9 | 211.7 | -23.8% |
| 1 | 1 (ok) | TPOT | 21.8 | 21.4 | -1.6% | 27.1 | 24.6 | -9.0% |
| 1 | 1 (ok) | ITL | 19.4 | 19.6 | +1.0% | 126.6 | 98.2 | -22.5% |
| 1 | 1 (ok) | E2EL | 2,911.1 | 2,831.7 | -2.7% | 3,641.3 | 3,241.2 | -11.0% |
| 1 | 1 (ok) | output tok/s | 126.3 | 135.9 | +7.6% | | | |
| 2 | 1 (ok) | TTFT | 168.0 | 111.8 | -33.5% | 384.8 | 284.6 | -26.0% |
| 2 | 1 (ok) | TPOT | 24.7 | 23.5 | -4.9% | 32.7 | 28.9 | -11.8% |
| 2 | 1 (ok) | ITL | 19.8 | 19.7 | -0.2% | 129.5 | 99.3 | -23.3% |
| 2 | 1 (ok) | E2EL | 3,308.8 | 3,100.9 | -6.3% | 4,322.5 | 3,775.9 | -12.6% |
| 2 | 1 (ok) | output tok/s | 249.4 | 268.2 | +7.5% | | | |
| 4 | 1 (ok) | TTFT | 193.4 | 118.0 | -39.0% | 574.1 | 403.2 | -29.8% |
| 4 | 1 (ok) | TPOT | 35.9 | 30.9 | -14.0% | 48.6 | 39.4 | -18.9% |
| 4 | 1 (ok) | ITL | 20.3 | 20.2 | -0.5% | 241.4 | 198.9 | -17.6% |
| 4 | 1 (ok) | E2EL | 4,783.0 | 4,057.6 | -15.2% | 6,421.3 | 5,110.4 | -20.4% |
| 4 | 1 (ok) | output tok/s | 483.7 | 521.6 | +7.8% | | | |
| 6 | 1 (ok) | TTFT | 467.8 | 213.2 | -54.4% | 1,686.9 | 749.2 | -55.6% |
| 6 | 1 (ok) | TPOT | 62.6 | 42.5 | -32.1% | 79.9 | 66.4 | -17.0% |
| 6 | 1 (ok) | ITL | 22.5 | 21.1 | -6.0% | 342.1 | 302.8 | -11.5% |
| 6 | 1 (ok) | E2EL | 8,634.8 | 5,687.7 | -34.1% | 11,067.7 | 8,654.8 | -21.8% |
| 6 | 1 (ok) | output tok/s | 700.9 | 758.3 | +8.2% | | | |
| 8 | 1 (ok) | TTFT | 3,702.9 | 2,410.8 | -34.9% | 6,323.2 | 3,979.4 | -37.1% |
| 8 | 1 (ok) | TPOT | 76.4 | 65.1 | -14.8% | 80.5 | 68.0 | -15.6% |
| 8 | 1 (ok) | ITL | 23.3 | 22.1 | -5.0% | 405.4 | 389.0 | -4.0% |
| 8 | 1 (ok) | E2EL | 11,688.2 | 9,156.4 | -21.7% | 15,105.7 | 12,612.9 | -16.5% |
| 8 | 1 (ok) | output tok/s | 746.3 | 866.6 | +16.1% | | | |
| 10 | 1 (ok) | TTFT | 5,107.8 | 4,474.3 | -12.4% | 10,598.0 | 8,641.6 | -18.5% |
| 10 | 1 (ok) | TPOT | 72.9 | 65.3 | -10.4% | 77.7 | 68.6 | -11.8% |
| 10 | 1 (ok) | ITL | 23.3 | 22.1 | -4.9% | 405.7 | 389.1 | -4.1% |
| 10 | 1 (ok) | E2EL | 13,537.2 | 12,400.1 | -8.4% | 17,360.3 | 15,625.6 | -10.0% |
| 10 | 1 (ok) | output tok/s | 768.1 | 860.6 | +12.0% | | | |
| inf | 1 (ok) | TTFT | 13,207.2 | 12,593.4 | -4.6% | 28,599.1 | 27,328.9 | -4.4% |
| inf | 1 (ok) | TPOT | 65.7 | 65.5 | -0.4% | 70.9 | 67.5 | -4.9% |
| inf | 1 (ok) | ITL | 23.3 | 22.1 | -5.2% | 405.7 | 389.1 | -4.1% |
| inf | 1 (ok) | E2EL | 21,660.6 | 20,987.2 | -3.1% | 31,246.0 | 29,852.9 | -4.5% |
| inf | 1 (ok) | output tok/s | 818.3 | 857.5 | +4.8% | | | |

Mean |error| per metric across rates:

| metric | mean \|err\| |
|---|---:|
| ttft median | 30.3% |
| ttft p99 | 27.9% |
| tpot median | 11.2% |
| tpot p99 | 12.7% |
| itl median | 3.3% |
| itl p99 | 12.4% |
| e2el median | 13.1% |
| e2el p99 | 13.8% |
| output tok/s | 9.2% |
