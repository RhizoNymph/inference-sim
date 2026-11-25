# Iteration-engine serving validation: 2026-09-30-serving-baseline

Real vLLM 0.29.0 serving (Qwen2.5-7B-Instruct bf16, one RTX 3090, 512 in /
128 out, 200 requests, `max_num_batched_tokens = 2048` chunked prefill,
`max_num_seqs = 64`, KV cache 82,864 tokens, prefix caching off) against the
simulator before and after the serving scheduler was rebuilt as an
iteration-level engine (`src/serving/engine/`, see
`docs/features/serving_iteration_engine.md`).

Calibration in both runs: the static-batch PP=1 profile
`lab-runs/2026-09-28-static-batch/calibration_profile-pp1.toml`
(`compute_efficiency = 0.8507`, `decode_memory_bandwidth_scale = 0.8383`),
unchanged. Nothing was fitted to the serving data.

Errors are `(sim - real) / real`. Times in ms; throughput in output tokens/s.

## What changed

- Before: the phase-pipeline scheduler ran every prefill, then every KV
  handoff, then every decode iteration on a greedy list scheduler, and priced
  a decode iteration over k sequences as `decode_one(B) * k / B` (linear in k,
  anchored at the workload's reference batch B). Exceeding `max_num_seqs` or
  the KV capacity rejected the candidate instead of queueing.
- After: a deterministic discrete-event loop of engine steps modeled on vLLM
  V1. Each step decodes one token for every running sequence and fills the
  rest of the 2048-token budget with prefill chunks; waiting requests are
  admitted only when their KV blocks fit, otherwise they queue. Each step's
  latency is one forward pass priced from its own composition by the solver's
  roofline (`IterationCostModel`): `max(dense + attention FLOPs, weights +
  KV-cache bytes)`, plus tensor/pipeline communication and the per-step
  overhead. The first output token is sampled by the step that finishes the
  prompt, as in vLLM.

### Before: phase-pipeline scheduler, Little's-law reference batch, admission limits lifted (`uncapped`)

| rate | sim status | TTFT p50 real / sim (err) | TTFT p99 real / sim (err) | TPOT p50 real / sim (err) | TPOT p99 real / sim (err) | ITL p50 real / sim (err) | ITL p99 real / sim (err) | out tok/s real / sim (err) |
|---|---|---|---|---|---|---|---|---|
| 1 | ok | 162.3 / 302.8 (+86.6%) | 277.9 / 707.3 (+154.6%) | 21.8 / 144.2 (+562.6%) | 27.1 / 241.6 (+792.4%) | 19.4 / 143.3 (+639.4%) | 126.6 / 388.6 (+206.9%) | 126.3 / 134.9 (+6.8%) |
| 2 | ok | 168.0 / 388.9 (+131.4%) | 384.8 / 900.8 (+134.1%) | 24.7 / 192.0 (+678.7%) | 32.7 / 228.6 (+598.6%) | 19.8 / 167.1 (+744.9%) | 129.5 / 485.9 (+275.1%) | 249.4 / 245.9 (-1.4%) |
| 4 | reject | 193.4 / 578.2 (+198.9%) | 574.1 / 1,445 (+151.8%) | 35.9 / 256.7 (+615.1%) | 48.6 / 307.9 (+533.4%) | 20.3 / 218.6 (+976.7%) | 241.4 / 926.8 (+284.0%) | 483.7 / 387.2 (-20.0%) |
| 6 | reject | 467.8 / 731.9 (+56.4%) | 1,687 / 2,026 (+20.1%) | 62.6 / 249.7 (+298.7%) | 79.9 / 285.8 (+257.6%) | 22.5 / 197.7 (+778.7%) | 342.1 / 1,467 (+328.7%) | 700.9 / 512.4 (-26.9%) |
| 8 | reject | 3,703 / 1,556 (-58.0%) | 6,323 / 6,322 (-0.0%) | 76.4 / 242.8 (+217.7%) | 80.5 / 307.3 (+281.6%) | 23.3 / 184.0 (+689.8%) | 405.4 / 1,518 (+274.4%) | 746.3 / 592.9 (-20.6%) |
| 10 | reject | 5,108 / 11,737 (+129.8%) | 10,598 / 19,454 (+83.6%) | 72.9 / 146.5 (+100.9%) | 77.7 / 146.5 (+88.4%) | 23.3 / 146.3 (+529.0%) | 405.7 / 147.1 (-63.7%) | 768.1 / 668.9 (-12.9%) |
| inf | reject | 13,207 / 19,515 (+47.8%) | 28,599 / 19,515 (-31.8%) | 65.7 / 69.1 (+5.1%) | 70.9 / 69.1 (-2.6%) | 23.3 / 69.1 (+195.9%) | 405.7 / 69.1 (-83.0%) | 818.3 / 905.0 (+10.6%) |

### After: iteration engine, vLLM admission limits (`engine`), reference batch 1

| rate | sim status | TTFT p50 real / sim (err) | TTFT p99 real / sim (err) | TPOT p50 real / sim (err) | TPOT p99 real / sim (err) | ITL p50 real / sim (err) | ITL p99 real / sim (err) | out tok/s real / sim (err) |
|---|---|---|---|---|---|---|---|---|
| 1 | ok | 162.3 / 108.4 (-33.2%) | 277.9 / 211.7 (-23.8%) | 21.8 / 21.4 (-1.6%) | 27.1 / 24.6 (-9.0%) | 19.4 / 19.6 (+1.0%) | 126.6 / 98.2 (-22.5%) | 126.3 / 135.9 (+7.6%) |
| 2 | ok | 168.0 / 111.8 (-33.5%) | 384.8 / 284.6 (-26.0%) | 24.7 / 23.5 (-4.9%) | 32.7 / 28.9 (-11.8%) | 19.8 / 19.7 (-0.2%) | 129.5 / 99.3 (-23.3%) | 249.4 / 268.2 (+7.5%) |
| 4 | ok | 193.4 / 118.0 (-39.0%) | 574.1 / 403.2 (-29.8%) | 35.9 / 30.9 (-14.0%) | 48.6 / 39.4 (-18.9%) | 20.3 / 20.2 (-0.5%) | 241.4 / 198.9 (-17.6%) | 483.7 / 521.6 (+7.8%) |
| 6 | ok | 467.8 / 213.2 (-54.4%) | 1,687 / 749.2 (-55.6%) | 62.6 / 42.5 (-32.1%) | 79.9 / 66.4 (-17.0%) | 22.5 / 21.1 (-6.0%) | 342.1 / 302.8 (-11.5%) | 700.9 / 758.3 (+8.2%) |
| 8 | ok | 3,703 / 2,411 (-34.9%) | 6,323 / 3,979 (-37.1%) | 76.4 / 65.1 (-14.8%) | 80.5 / 68.0 (-15.6%) | 23.3 / 22.1 (-5.0%) | 405.4 / 389.0 (-4.0%) | 746.3 / 866.6 (+16.1%) |
| 10 | ok | 5,108 / 4,474 (-12.4%) | 10,598 / 8,642 (-18.5%) | 72.9 / 65.3 (-10.4%) | 77.7 / 68.6 (-11.8%) | 23.3 / 22.1 (-4.9%) | 405.7 / 389.1 (-4.1%) | 768.1 / 860.6 (+12.0%) |
| inf | ok | 13,207 / 12,593 (-4.6%) | 28,599 / 27,329 (-4.4%) | 65.7 / 65.5 (-0.4%) | 70.9 / 67.5 (-4.9%) | 23.3 / 22.1 (-5.2%) | 405.7 / 389.1 (-4.1%) | 818.3 / 857.5 (+4.8%) |

"Before" with vLLM's real admission limits (`engine`, the harness default
then) rejected every rate >= 4 req/s (`decode capacity exceeded`, KV
residency and KV block capacity exceeded), so the before table uses the
`uncapped` run, the only one that reported latencies at every rate; its
`sim status` column shows what the `engine` run returned. Sources:
`validation-serving-pp1-profile-littles-law{,-uncapped}.json` (before),
`validation-serving-pp1-profile-iteration-engine.json` (after).

Mean |error| across the 7 rates:

| metric | before (uncapped) | after |
|---|---:|---:|
| TTFT p50 | 101.3% | 30.3% |
| TTFT p99 | 82.3% | 27.9% |
| TPOT p50 | 354.1% | 11.2% |
| TPOT p99 | 364.9% | 12.7% |
| ITL p50 | 650.6% | 3.3% |
| ITL p99 | 216.5% | 12.4% |
| E2EL p50 | 341.3% | 13.1% |
| E2EL p99 | 342.8% | 13.8% |
| output tok/s | 14.2% | 9.2% |

## Targets

| target | result |
|---|---|
| TPOT p50 within ~15% at 1, 2, 4 req/s | met: -1.6%, -4.9%, -14.0% |
| ITL p50 within ~15% at 1, 2, 4 req/s | met: +1.0%, -0.2%, -0.5% |
| TTFT p50 within ~15% at 1, 2, 4 req/s | **not met**: -33.2%, -33.5%, -39.0% (see below) |
| TPOT rises while ITL stays flat | reproduced: sim TPOT p50 21.4 -> 65.5 ms (real 21.8 -> 65.7), sim ITL p50 19.6 -> 22.1 ms (real 19.4 -> 23.3) |
| saturation throughput within ~15% of ~800 tok/s | met: 857.5 vs 818.3 at `inf` (+4.8%); 860.6 vs 768.1 at 10 req/s (+12.0%) |
| TTFT grows into seconds past saturation | reproduced: 2.4 s at 8 req/s, 4.5 s at 10, 12.6 s at `inf` (real 3.7, 5.1, 13.2 s) |
| no candidate rejected under vLLM's limits | met: every rate `ok`, 200/200 requests complete, peak running sequences <= 64 |

## Per-step overhead (static-batch data only)

The step model is `max(compute, memory) + scheduler_overhead_us`. Fitting that
constant on the 10 static-batch points of
`lab-runs/2026-09-28-static-batch/calibration_profile-pp1.toml` (5 prefill
latencies, 5 decode steps = decode_ms / 128) with the profile's two scalars
held fixed gives **0.04 ms** (least squares on relative error; the median
residual is 0.0 ms). The memory-bound decode steps already match within 1%
and pin the constant to zero, so no per-step overhead is applied
(`scheduler_overhead_us` stays 0). The static residuals are not a constant:
+17.5 ms at the 1x512 prefill, ~0 at 1x2048 and 8x512, and -90 to -99 ms at
the 16,384-token prefills.

## Remaining gap: TTFT at low load

At 1 req/s the simulated TTFT p50 is 108.4 ms: the step that carries the
512-token prompt (97.5 ms predicted) plus waiting for the in-flight decode
step (~10 ms on average). The measured 162.3 ms breaks down as:

- the 512-token forward pass itself is ~17 ms slower than the profile
  predicts: the static-batch 1x512 prefill measures 114.7 ms vs 97.2 ms
  predicted (-15%), because the median-ratio `compute_efficiency` fits the
  >= 2048-token shapes. The measured serving ITL p99 at 1 req/s (126.6 ms,
  the gap a decoding request sees when a prompt shares its step) vs the
  simulated 98.2 ms shows the same ~20-28 ms miss;
- ~35 ms not in the engine model: API-server tokenization, detokenization
  and HTTP streaming, and possibly the engine scheduling a new arrival one
  step later than the next step boundary.

Even a perfect forward-pass model would put TTFT p50 at ~125 ms (-23%), so
reaching the 15% target needs a measured per-request frontend latency, which
has to come from a separate measurement, not from fitting this serving data.
The same two effects also explain the 6 req/s point: TTFT -54% and TPOT -32%.
At 6 req/s the real server is close to saturation (~700 tok/s), and
under-pricing the ~512-token prompt steps by ~15% leaves the simulated server
further from saturation, with much less queueing.

## Runtime

- Before: about 1 minute per 7-rate sweep with the Little's-law reference
  batch (as reported when the gap was found; not re-measured here). The
  reference-batch-64 run did not finish 7 rates within 10 minutes.
- After: 3.05 s wall for the whole 7-rate `lab.py validate` sweep, including
  the static-score simulations. One rate through the release binary takes
  0.11 s with the text report and 0.66 s with `--json`, which is mostly JSON
  serialization of per-request token timelines and per-step operations. The
  in-process 200-request, 128-token simulation takes 0.10 s in a release
  build
  (`serving::engine::tests::serving::two_hundred_request_serving_run_finishes_quickly`).

## Reproduce

```sh
cargo build --release
cd tools/lab
uv run lab.py validate specs/rtx3090_qwen7b_serving_pp1.toml \
    --run-dir ../../lab-runs/2026-09-30-serving-baseline \
    --profile ../../lab-runs/2026-09-28-static-batch/calibration_profile-pp1.toml \
    --tag serving-pp1-profile-iteration-engine
```

The spec now uses `sim_reference_batch = 1`. A run with the Little's-law
reference batch gives identical serving metrics because the engine does not
use the reference batch.
