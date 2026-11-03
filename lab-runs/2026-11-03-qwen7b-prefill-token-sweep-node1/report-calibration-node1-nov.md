## Compute-efficiency curve: node1-nov

Measurements `lab-runs/2026-11-03-qwen7b-prefill-token-sweep-node1/measured.jsonl`, base profile `lab-runs/2026-11-03-qwen7b-static-pp1-node1/calibration_profile-node1-nov-gemm-peak.toml`, profile `lab-runs/2026-11-03-qwen7b-prefill-token-sweep-node1/calibration_profile-node1-nov.toml` (loaded and checked by the simulator).

| tokens per pass | efficiency | shapes |
|---:|---:|---|
| 128 | 0.3479 | 1x128 |
| 256 | 0.4048 | 1x256 |
| 512 | 0.5202 | 1x512 |
| 1024 | 0.8228 | 1x1024 |
| 2048 | 0.8343 | 1x2048 |
| 4096 | 0.8464 | 1x4096 |

Weight-read bound (skipped): 1x16, 1x32, 1x64 (margin 0.7).
