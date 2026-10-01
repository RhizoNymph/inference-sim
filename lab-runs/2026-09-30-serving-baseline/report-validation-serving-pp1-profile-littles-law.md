## Validation: serving-pp1-profile-littles-law

Measurements `lab-runs/2026-09-30-serving-baseline/rate_*.json`.

### serving-pp1-profile-littles-law: qwen7b-serving-pp1 under profile-calibration_profile-pp1

Calibration: `profile-calibration_profile-pp1`. Reference batch = the serving model's `[request].batch_size` (see the feature doc). `n/a` = the simulator rejected the candidate and reported no latency metrics.

| rate req/s | ref batch (sim status) | metric | real median | sim median | err | real p99 | sim p99 | err |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 3 (ok) | TTFT | 162.3 | 302.8 | +86.6% | 277.9 | 707.3 | +154.6% |
| 1 | 3 (ok) | TPOT | 21.8 | 144.2 | +562.6% | 27.1 | 241.6 | +792.4% |
| 1 | 3 (ok) | ITL | 19.4 | 143.3 | +639.4% | 126.6 | 388.6 | +206.9% |
| 1 | 3 (ok) | E2EL | 2,911.1 | 18,644.4 | +540.5% | 3,641.3 | 31,229.3 | +757.6% |
| 1 | 3 (ok) | output tok/s | 126.3 | 134.9 | +6.8% | | | |
| 2 | 6 (ok) | TTFT | 168.0 | 388.9 | +131.4% | 384.8 | 900.8 | +134.1% |
| 2 | 6 (ok) | TPOT | 24.7 | 192.0 | +678.7% | 32.7 | 228.6 | +598.6% |
| 2 | 6 (ok) | ITL | 19.8 | 167.1 | +744.9% | 129.5 | 485.9 | +275.1% |
| 2 | 6 (ok) | E2EL | 3,308.8 | 24,832.6 | +650.5% | 4,322.5 | 29,637.5 | +585.7% |
| 2 | 6 (ok) | output tok/s | 249.4 | 245.9 | -1.4% | | | |
| 4 | 11 (reject) | TTFT | 193.4 | n/a | - | 574.1 | n/a | - |
| 4 | 11 (reject) | TPOT | 35.9 | n/a | - | 48.6 | n/a | - |
| 4 | 11 (reject) | ITL | 20.3 | n/a | - | 241.4 | n/a | - |
| 4 | 11 (reject) | E2EL | 4,783.0 | n/a | - | 6,421.3 | n/a | - |
| 4 | 11 (reject) | output tok/s | 483.7 | 387.9 | -19.8% | | | |
| 6 | 17 (reject) | TTFT | 467.8 | n/a | - | 1,686.9 | n/a | - |
| 6 | 17 (reject) | TPOT | 62.6 | n/a | - | 79.9 | n/a | - |
| 6 | 17 (reject) | ITL | 22.5 | n/a | - | 342.1 | n/a | - |
| 6 | 17 (reject) | E2EL | 8,634.8 | n/a | - | 11,067.7 | n/a | - |
| 6 | 17 (reject) | output tok/s | 700.9 | 513.3 | -26.8% | | | |
| 8 | 22 (reject) | TTFT | 3,702.9 | n/a | - | 6,323.2 | n/a | - |
| 8 | 22 (reject) | TPOT | 76.4 | n/a | - | 80.5 | n/a | - |
| 8 | 22 (reject) | ITL | 23.3 | n/a | - | 405.4 | n/a | - |
| 8 | 22 (reject) | E2EL | 11,688.2 | n/a | - | 15,105.7 | n/a | - |
| 8 | 22 (reject) | output tok/s | 746.3 | 593.4 | -20.5% | | | |
| 10 | 28 (reject) | TTFT | 5,107.8 | n/a | - | 10,598.0 | n/a | - |
| 10 | 28 (reject) | TPOT | 72.9 | n/a | - | 77.7 | n/a | - |
| 10 | 28 (reject) | ITL | 23.3 | n/a | - | 405.7 | n/a | - |
| 10 | 28 (reject) | E2EL | 13,537.2 | n/a | - | 17,360.3 | n/a | - |
| 10 | 28 (reject) | output tok/s | 768.1 | 668.9 | -12.9% | | | |
| inf | 64 (reject) | TTFT | 13,207.2 | n/a | - | 28,599.1 | n/a | - |
| inf | 64 (reject) | TPOT | 65.7 | n/a | - | 70.9 | n/a | - |
| inf | 64 (reject) | ITL | 23.3 | n/a | - | 405.7 | n/a | - |
| inf | 64 (reject) | E2EL | 21,660.6 | n/a | - | 31,246.0 | n/a | - |
| inf | 64 (reject) | output tok/s | 818.3 | 905.0 | +10.6% | | | |

Simulator rejections:

- rate 4: decode capacity exceeded: peak_decode_sequences 120 > max_decode_sequences 64
- rate 6: decode capacity exceeded: peak_decode_sequences 162 > max_decode_sequences 64; KV residency capacity exceeded: peak_resident_tokens 103680 > max_resident_tokens 82864; KV block capacity exceeded: peak_kv_blocks 6480 > max_kv_blocks 5179
- rate 8: decode capacity exceeded: peak_decode_sequences 199 > max_decode_sequences 64; KV residency capacity exceeded: peak_resident_tokens 127360 > max_resident_tokens 82864; KV block capacity exceeded: peak_kv_blocks 7960 > max_kv_blocks 5179
- rate 10: decode capacity exceeded: peak_decode_sequences 200 > max_decode_sequences 64; KV residency capacity exceeded: peak_resident_tokens 128000 > max_resident_tokens 82864; KV block capacity exceeded: peak_kv_blocks 8000 > max_kv_blocks 5179
- rate inf: decode capacity exceeded: peak_decode_sequences 200 > max_decode_sequences 64; KV residency capacity exceeded: peak_resident_tokens 128000 > max_resident_tokens 82864; KV block capacity exceeded: peak_kv_blocks 8000 > max_kv_blocks 5179

Mean |error| per metric across rates:

| metric | mean \|err\| |
|---|---:|
| ttft median | 109.0% |
| ttft p99 | 144.3% |
| tpot median | 620.6% |
| tpot p99 | 695.5% |
| itl median | 692.1% |
| itl p99 | 241.0% |
| e2el median | 595.5% |
| e2el p99 | 671.7% |
| output tok/s | 14.1% |
