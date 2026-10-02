## Compute-efficiency curve: curve-only

Measurements `lab-runs/2026-10-01-qwen7b-prefill-token-sweep/measured.jsonl`, base profile `lab-runs/2026-09-28-static-batch/calibration_profile-pp1-recalibrated.toml`, profile `lab-runs/2026-10-01-qwen7b-prefill-token-sweep/calibration_profile-curve-only.toml` (loaded and checked by the simulator).

| tokens per pass | efficiency | shapes |
|---:|---:|---|
| 128 | 0.6539 | 1x128 |
| 256 | 0.6183 | 1x256 |
| 512 | 0.7244 | 1x512 |
| 1024 | 0.8378 | 1x1024 |
| 2048 | 0.8466 | 1x2048 |
| 4096 | 0.8739 | 1x4096 |

Weight-read bound (skipped): 1x16, 1x32, 1x64 (margin 0.25).
