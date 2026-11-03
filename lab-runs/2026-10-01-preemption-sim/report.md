# Decode batch sweep on the iteration engine: KV-exhaustion preemption

Script: `predict.py` (run from the repo root after `cargo build --release`);
numbers: `predictions.json`. Measurements:
`lab-runs/2026-10-01-qwen7b-decode-batch-sweep/` (batch 1-40) and `-tail/`
(32-64; its batch 32 and 40 values are used). vLLM logged a KV cache of
24,896 tokens (1,556 blocks of 16) for these settings.

Each shape is B requests at t = 0 with 512-token prompts, simulated once with
1 output token and once with 1 + 128; decode = difference of the two
makespans, as `bench_latency.py` measures. Calibration: curve-only profile
scalars, no frontend latency.

| batch | real ms/step | uncapped | reserved (24,896) | preempt (24,896) | preempted requests | recomputed tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 1-32 | 19.25-21.89 | 2.5% mean \|err\| | identical | identical | 0 | 0 |
| 40 | 28.73 | 21.09 (-26.6%) | 40.59 (+41.3%) | **27.66 (-3.8%)** | 2 | 1,232 |
| 48 | 52.86 | 21.43 (-59.5%) | 40.69 (-23.0%) | **48.90 (-7.5%)** | 10 | 5,664 |
| 56 | 53.49 | 21.77 (-59.3%) | 41.18 (-23.0%) | **49.39 (-7.7%)** | 10 | 5,664 |
| 64 | 53.02 | 22.11 (-58.3%) | 41.52 (-21.7%) | **49.73 (-6.2%)** | 10 | 5,664 |
| **40-64 mean \|err\|** | | 50.9% | 27.2% | **6.3%** | | |

- Batches 1-32 fit the cache (32 x 641 = 20,512 tokens) and are bit-identical
  in all three modes.
- Uncapped decoding stays smooth; reserving whole footprints at admission
  turns 40 into a step function (38 admitted, 2 run as a second wave) and
  under-predicts 48-64.
- Recompute preemption reproduces the cliff: at batch 40 the cache runs out
  around decode step 97 and the two newest requests are preempted, then
  recompute ~610 tokens each and finish alone; at 48-64 the 1,556 blocks hold
  ~48 prompts but not their growth, so ten requests are preempted early and
  decode in a second wave. The 2.4x step time of 48-64 comes out as 2.3-2.4x.
- Remaining -4% to -8% matches the decode-step under-prediction already seen
  at batch 28-32 (KV-read bandwidth, `_todo.md` item 6).
