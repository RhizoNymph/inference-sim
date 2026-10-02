# Validation ledger

Measured accuracy of inference-sim against real hardware, one entry per
regime. Every number here is reproducible from a `lab-runs/` directory with
`tools/lab/lab.py` (see `docs/features/lab_harness.md`). Errors are
`(sim - measured) / measured`; tables report mean |error| over the listed
shapes or rates. "Fitted" means the calibration was fitted on the same data;
"LOO" means leave-one-shape-out (each shape predicted by scalars fitted
without it); "transfer" means calibration fitted on a different regime.

When adding an entry: record the simulator git sha from the run's
`manifest.json` (the two 2026-09 runs predate the harness and have none), the
calibration used, and every caveat; never delete superseded entries, mark
them superseded.

## Summary

| # | date | hardware | model | backend | parallelism | mode | calibration | prefill | decode step | end-to-end | status |
|---|---|---|---|---|---|---|---|---:|---:|---:|---|
| 1 | 2026-09-28 | 1x RTX 3090 24 GB | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0 | TP1 PP1 | static batch | simulator defaults | 138.5% | 15.9% | 37.3% | baseline |
| 2 | 2026-09-28 | 1x RTX 3090 24 GB | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0 | TP1 PP1 | static batch | fitted scalars, LOO | 5.2% | 0.8% | **1.3%** | validated |
| 3 | 2026-09-28 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, Ray | TP1 PP2 | static batch | simulator defaults | 138.1% | 19.4% | 33.7% | baseline |
| 4 | 2026-09-28 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, Ray | TP1 PP2 | static batch | PP1 profile (transfer) | 14.1% | 4.2% | **1.1%** | validated, with caveats |
| 5 | 2026-09-30 | 1x RTX 3090 24 GB | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0 | TP1 PP1 | serving, Poisson 1-10 req/s + inf | PP1 profile (transfer) | see entry | see entry | see entry | **not accurate**; superseded by 10 |
| 6 | 2026-09-30 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, Ray | TP2 PP1 | static batch | PP1 profile, pre-fix simulator | 90.8% | 60.4% | 74.8% | baseline |
| 7 | 2026-09-30 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, Ray | TP2 PP1 | static batch | PP1 profile + measured NIC + `collective_latency_scale` fitted on 1 shape (held-out 4) | 7.5% | 14.5% | **9.6%** | superseded by 12 |
| 8 | 2026-09-30 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, Ray | TP1 PP2 | static batch | as row 7 (network fit transfer) | 14.5% | 3.8% | **1.6%** | validated |
| 9 | 2026-09-30 | 1x RTX 3090 24 GB (node1, native venv) | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0 | TP1 PP1 | static batch, 4k-16k prompts | PP1 profile (transfer to 8x context) | 0.6% | 2.5% | **0.8%** | validated |
| 10 | 2026-09-30 | 1x RTX 3090 24 GB | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0 | TP1 PP1 | serving, iteration engine, Poisson 1-10 req/s + inf | PP1 profile (transfer) | see entry | see entry | see entry | superseded by 15 |
| 11 | 2026-09-30 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-14B-Instruct bf16 | vLLM 0.29.0, Ray | TP1 PP2 | static batch | recalibrated 7B constants (model transfer) | 13.4% | 7.2% | **4.9%** | validated, with caveats |
| 12 | 2026-09-30 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, Ray | TP2 PP1 | static batch | PP1 profile + measured NCCL curves, no network fit | 12.5% | 7.2% | **4.0%** (all held out) | validated, with caveats |
| 13 | 2026-09-30 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, Ray | TP1 PP2 | static batch | PP1 profile + measured NCCL curves | 15.3% | 3.5% | 2.4% | validated, with caveats; entry 8 (alpha-beta) is 1.6% |
| 14 | 2026-10-01 | 1x RTX 3090 (node0; node1 for long context and decode sweep) + 2-node PP=2 | Qwen2.5-7B / 14B bf16 | vLLM 0.29.0 | TP1 PP1, TP1 PP2 | static batch, prefill token sweep 16-4096 + all static regimes | token-dependent efficiency curve (fitted on the sweep) | 0.8-10.4% by regime | unchanged | 0.9-5.1% by regime | validated; see entry |
| 15 | 2026-10-01 | 1x RTX 3090 24 GB | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0 | TP1 PP1 | serving, iteration engine, Poisson 1-10 req/s + inf | curve + measured frontend latency (transfer) | see entry | see entry | see entry | validated; low-load TTFT -15% to -19% |
| 16 | 2026-10-01 | 1x RTX 3090 24 GB (node1) | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0 | TP1 PP1 | static decode batch sweep 1-64 x 512 | recalibrated scalars / curve | 6.0% | 2.7% (batch 1-32) | 3.0% | validated to batch 32; KV exhaustion beyond |
| 17 | 2026-10-01 | 1x RTX 3090 (node2), graphics clock locked at 1200 MHz | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0 | TP1 PP1 | static batch, degraded GPU | recalibrated constants, peak scaled to the locked clock | 13.8% | 22.8% | 9.6% (offsetting errors) | **not accurate**: see entry |
| 18 | 2026-10-01 | 2x RTX 3090: prefill node0, decode node1, NIXL over UCX TCP on 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, NixlConnector | 1P1D disaggregated | serving, Poisson 1-8 req/s + burst | curve + frontend profile (blind prediction) | see entry | see entry | see entry | validated below saturation; burst not accurate |

## 1-2. RTX 3090, Qwen2.5-7B, static batch, PP=1

- Run: [`lab-runs/2026-09-28-static-batch/`](../lab-runs/2026-09-28-static-batch/)
  (`real_pp1.jsonl`, measured on node0; report in `report.md`,
  numbers in `calibration-pp1.json`).
- Spec: `tools/lab/specs/rtx3090_qwen7b_static_pp1.toml`.
- Shapes (batch x prompt): 1x512, 1x2048, 8x512, 8x2048, 32x512; 128 decode
  tokens; medians of 5 iterations after 2 warmups; prefix caching off; CUDA
  graphs on.
- Calibration fitted: `compute_efficiency = 0.8507` (0.35 x median sim/real
  prefill 2.431), `decode_memory_bandwidth_scale = 0.8383`; profile
  `lab-runs/2026-09-28-static-batch/calibration_profile-pp1.toml`, verified
  to load (`within_valid_shape`) and reproduce the scalar predictions exactly.
- Results (mean |error|): defaults 138.5% prefill / 15.9% decode step /
  37.3% end-to-end; fitted in-sample 4.4% / 0.5% / 1.4%; LOO 5.2% / 0.8% /
  1.3% (worst shape 2.0% end-to-end).
- Caveats: the fit has 5 shapes and one decode length, so `valid_shape` is
  batch 1-32, prompt 512-2048, decode 128. The largest prefill miss is 1x512
  (-16.5% LOO): small prefills carry fixed launch/scheduler overhead the
  roofline does not model. The measured prefill includes one sampled token.

## 3-4. RTX 3090 x2, Qwen2.5-7B, static batch, PP=2 over 2x10GbE

- Run: same directory, `real_pp2.jsonl` (node0 + node1, Ray); numbers in
  `validation-pp2-default.json` and `validation-pp2-pp1-profile.json`.
- Spec: `tools/lab/specs/rtx3090_qwen7b_static_pp2.toml`.
- Calibration: the PP=1 profile unchanged (no PP=2 fit), so this is a
  transfer test of the pipeline-parallel model.
- Results: defaults 138.1% / 19.4% / 33.7%; with the PP=1 profile 14.1%
  prefill, 4.2% decode step, **1.1% end-to-end** (worst shape 3.7%).
- Caveats: requires the pipeline-parallel roofline fix in this worktree
  (without it the simulator predicted PP=2 at roughly half the PP=1 latency;
  measured PP=2 is no faster than PP=1). End-to-end accuracy benefits from
  error cancellation: prefill is over-predicted at 8x512 (+20.8%) and
  8x2048 (+13.9%) while decode steps at batch >= 8 are under-predicted by
  3-11%, which is where inter-node activation transfer and bubbles show up.
  Measured PP=2 spreads are large (e2e spread up to 627 ms at 8x2048), so
  single-shape errors under ~5% are within run-to-run noise.

## 5. RTX 3090, Qwen2.5-7B, online serving (vllm bench serve), PP=1 (superseded by 10)

Superseded: this entry measured the phase-pipeline serving scheduler, which
was replaced for colocated continuous-batching serving by the iteration
engine (entry 10). Kept for the record.

- Run: [`lab-runs/2026-09-30-serving-baseline/`](../lab-runs/2026-09-30-serving-baseline/)
  (`rate_<R>.json`, server in the node0 container, client from the node0
  venv; `run_serving.sh` is the exact script). Numbers in
  `validation-serving-pp1-profile-*.json`, tables in `report.md`.
- Spec: `tools/lab/specs/rtx3090_qwen7b_serving_pp1.toml`. Workload: random
  dataset, 512 input / 128 output tokens fixed, 200 prompts, Poisson arrivals
  at 1, 2, 4, 6, 8, 10 req/s and `inf`; `max_num_batched_tokens = 2048`
  (chunked prefill), `max_num_seqs = 64`, KV cache 82,864 tokens (from
  `server.log`).
- Calibration: the static PP=1 profile (entry 2), unchanged.
- Result: the serving model is **not accurate**. Output-token throughput is
  within ~20% at every rate, but per-token latencies are off by 5-10x at low
  load, and the simulator rejects every rate >= 4 req/s under vLLM's real
  admission limits.

Simulator vs measured, Little's-law reference batch (the harness default),
admission limits lifted (`--admission uncapped`) so every rate reports
latencies (with vLLM's limits, rates >= 4 are rejected with `decode capacity
exceeded`). Times in ms, throughput in output tokens/s:

| rate | ref batch | TTFT p50 real / sim | TTFT p99 real / sim | TPOT p50 real / sim | TPOT p99 real / sim | ITL p50 real / sim | ITL p99 real / sim | tok/s real / sim |
|---|---:|---|---|---|---|---|---|---|
| 1 | 3 | 162 / 303 | 278 / 707 | 21.8 / 144 | 27.1 / 242 | 19.4 / 143 | 127 / 389 | 126 / 135 |
| 2 | 6 | 168 / 389 | 385 / 901 | 24.7 / 192 | 32.7 / 229 | 19.8 / 167 | 130 / 486 | 249 / 246 |
| 4 | 11 | 193 / 578 | 574 / 1,446 | 35.9 / 257 | 48.6 / 308 | 20.3 / 219 | 241 / 927 | 484 / 387 |
| 6 | 17 | 468 / 732 | 1,687 / 2,026 | 62.6 / 250 | 79.9 / 286 | 22.5 / 198 | 342 / 1,467 | 701 / 512 |
| 8 | 22 | 3,703 / 1,556 | 6,323 / 6,322 | 76.4 / 243 | 80.5 / 307 | 23.3 / 184 | 405 / 1,518 | 746 / 593 |
| 10 | 28 | 5,108 / 11,737 | 10,598 / 19,454 | 72.9 / 147 | 77.7 / 147 | 23.3 / 146 | 406 / 147 | 768 / 669 |
| inf | 64 | 13,207 / 19,515 | 28,599 / 19,515 | 65.7 / 69.1 | 70.9 / 69.1 | 23.3 / 69.1 | 406 / 69.1 | 818 / 905 |

Diagnosis (for the serving-loop fix):

- `src/serving/scheduling/decode.rs::schedule_continuous_decodes` prices a
  decode iteration over k sequences as `decode_one(B) * k / B` - linear in k
  through the origin, anchored at the workload's `[request].batch_size = B`.
  The static benchmark shows the real step is nearly flat in k (19.3 ms at
  k = 1, 20.8 ms at k = 32), because decode reads the weights once per step.
  With B = 1 (`validation-serving-pp1-profile-ref1-uncapped.json`) the
  simulator's throughput is capped at ~49 tok/s at every rate (TPOT ~3.9 s);
  with B = 64 (`validation-serving-pp1-profile-ref64-uncapped.json`, ~37
  CPU-minutes for 7 rates) median TPOT is 0.35 ms at 1-2 req/s (~60x too
  cheap) and TTFT median 98 ms vs 162-168 ms measured, while throughput is
  within 10-20% at every rate. No single B fits a rate sweep;
  Little's law is the least-bad choice.
- The same linear scaling makes simulated concurrency run away (at 4 req/s
  the simulation peaks at 120 concurrent sequences; vLLM peaked at 37),
  which trips decode-capacity and KV rejections. vLLM instead queues (TTFT grows, as measured at 8-10 req/s);
  the simulator has no admission queue for decode capacity, so it rejects the
  candidate.
- Median ITL in vLLM stays ~19-23 ms at every load (steps are nearly flat in
  batch) while p99 ITL jumps to 130-400 ms when a chunked prefill shares the
  step; the simulator's ITL tracks its inflated TPOT instead.

## 6-8. RTX 3090 x2, Qwen2.5-7B, static batch, TP=2 over 2x10GbE

Data: `lab-runs/2026-09-30-tp2/` (real_tp2.jsonl, sim sweeps, fitted_network.json).

Real TP=2 across node0+node1 makes prefill about 5x slower than one GPU
(8x2048: 3,096 -> 17,624 ms) and decode slower as batch grows (32x512: 20.8 ->
51.3 ms/step). The pre-fix simulator predicted about 0.5x for everything
(row 6). Fixes (uncommitted, staging): two all-reduces per layer, full
activation per rank, collective latency paid per token crossing, and
tensor-parallel collectives blocking the next layer by default
(`allow_compute_comm_overlap = false`).

Row 7 also sets node0's NIC to the measured single-stream node0->node1 rate
(3.61 Gb/s; the extra prefill time implies 3.62 Gb/s) and fits
`collective_latency_scale = 1.824` on 1x512 decode only; the other four shapes
are held out. Caveat: batch-8 decode is 18-25% under. The measured NCCL curve
(`lab-runs/2026-09-30-nccl-curve/`) shows a protocol step between 32 KB and
64 KB all-reduces, and batch-8 decode sends 57 KB, so an alpha-beta model
cannot capture it; measured collective curves are the planned fix.

Row 8 re-runs PP=2 with the same network settings, an independent check of the
network fit: 1.6% end-to-end (was 1.1% before the network change).

## 9. RTX 3090, Qwen2.5-7B, static batch, 4k-16k prompts (node1)

Data: `lab-runs/2026-09-30-qwen7b-static-longctx-node1/` (first run of the lab
harness on real hardware) and `lab-runs/2026-09-30-longctx-sim/`.

Same 3090 constants as row 2 (fitted on 512-2048 prompts on node0 in Docker),
applied to 4k-16k prompts on node1 running natively: prefill 0.6%, decode step
2.5%, end-to-end 0.8%. Caveat: decode error trends from -0.9% (1x4096) to
-4.4% (4x4096) as total KV grows, so the KV-read term is slightly undercounted
at large cache sizes.

## 10. RTX 3090, Qwen2.5-7B, online serving on the iteration engine, PP=1

- Run: same measurements as entry 5
  ([`lab-runs/2026-09-30-serving-baseline/`](../lab-runs/2026-09-30-serving-baseline/));
  numbers in `validation-serving-pp1-profile-iteration-engine.json`, full
  before/after tables and analysis in `report-iteration-engine.md`.
- Simulator: the iteration-level serving engine
  (`docs/features/serving_iteration_engine.md`), built from the working tree
  of `feat/serving-iteration-loop` on 2026-09-30 (uncommitted, so no sha).
- Spec: `tools/lab/specs/rtx3090_qwen7b_serving_pp1.toml` with vLLM's real
  admission limits (`sim_admission = "engine"`) and `sim_reference_batch =
  1`. The engine does not use the reference batch: a Little's-law run gives
  identical metrics.
- Calibration: the static PP=1 profile (entry 2; built before the entry 11 recalibration, which leaves 7B predictions unchanged), unchanged; nothing fitted
  to serving data. A per-step overhead fitted on the static-batch data alone
  comes out at 0.04 ms, so none is applied.
- Result (mean |error| over 7 rates): TTFT p50 30.3%, TPOT p50 11.2%, ITL
  p50 3.3%, E2EL p50 13.1%, output throughput 9.2% (entry 5: 101%, 354%,
  651%, 341%, 14%). Every rate is feasible with vLLM's limits; requests queue
  instead of being rejected.

| rate | TTFT p50 real / sim | TTFT p99 real / sim | TPOT p50 real / sim | TPOT p99 real / sim | ITL p50 real / sim | ITL p99 real / sim | tok/s real / sim |
|---|---|---|---|---|---|---|---|
| 1 | 162 / 108 (-33%) | 278 / 212 | 21.8 / 21.4 (-2%) | 27.1 / 24.6 | 19.4 / 19.6 (+1%) | 127 / 98 | 126 / 136 |
| 2 | 168 / 112 (-34%) | 385 / 285 | 24.7 / 23.5 (-5%) | 32.7 / 28.9 | 19.8 / 19.7 (-0%) | 130 / 99 | 249 / 268 |
| 4 | 193 / 118 (-39%) | 574 / 403 | 35.9 / 30.9 (-14%) | 48.6 / 39.4 | 20.3 / 20.2 (-1%) | 241 / 199 | 484 / 522 |
| 6 | 468 / 213 (-54%) | 1,687 / 749 | 62.6 / 42.5 (-32%) | 79.9 / 66.4 | 22.5 / 21.1 (-6%) | 342 / 303 | 701 / 758 |
| 8 | 3,703 / 2,411 (-35%) | 6,323 / 3,979 | 76.4 / 65.1 (-15%) | 80.5 / 68.0 | 23.3 / 22.1 (-5%) | 405 / 389 | 746 / 867 |
| 10 | 5,108 / 4,474 (-12%) | 10,598 / 8,642 | 72.9 / 65.3 (-10%) | 77.7 / 68.6 | 23.3 / 22.1 (-5%) | 406 / 389 | 768 / 861 |
| inf | 13,207 / 12,593 (-5%) | 28,599 / 27,329 | 65.7 / 65.5 (-0%) | 70.9 / 67.5 | 23.3 / 22.1 (-5%) | 406 / 389 | 818 / 858 |

- Reproduced: TPOT rises with load while median ITL stays at the decode
  step (19.6 -> 22.1 ms simulated, 19.4 -> 23.3 ms real); saturation
  throughput 858 vs 818 tok/s (+4.8%); TTFT grows into seconds past
  saturation; ITL p99 at and past saturation (389 vs 405 ms) is the step
  carrying a full 2048-token prefill chunk.
- Caveats: low-load TTFT is under-predicted by 33-39% at 1-4 req/s. About
  17 ms comes from the static calibration under-predicting the 512-token
  forward pass (static 1x512 prefill: 114.7 measured vs 97.2 ms predicted).
  The remaining ~35 ms is frontend and scheduling latency the engine does
  not model (`iteration_engine_no_frontend_overhead`). Under-pricing
  512-token prompt steps also leaves the 6 req/s point, near saturation,
  short on queueing (TTFT -54%, TPOT -32%). The engine reserves KV for whole
  sequences and never preempts; this run never approaches KV exhaustion
  (64 x 640 = 40,960 tokens vs 82,864 available). The simulated Poisson
  sample path differs from vLLM's.

## 11. RTX 3090 x2, Qwen2.5-14B, static batch, PP=2 (model transfer + recalibration)

Data: `lab-runs/2026-09-30-qwen14b-static-pp2/` (measured via the lab
harness), `lab-runs/2026-09-30-recalibration/` (recalibrate.py,
recalibration.json, clock-adjusted cluster configs).

First attempt with the 7B constants: 22.8% end-to-end, prefill 69.9%. Cause:
the specs gave `parameters_gb` but not `ffn_hidden_size`, so the simulator
derived FLOPs from a 4x-hidden MLP default: 6.23B parameters for 7B (real
7.62B, FLOPs undercounted 18%) and 19.68B for 14B (real 14.77B, overcounted
33%). The fitted `compute_efficiency` had silently absorbed the 7B error.

Recalibration: real MLP widths in every spec (7B 18944, 14B 13824), and the
cluster peak set to 88 TFLOPs (datasheet 71 at 1695 MHz scaled to the
measured 2100 MHz max SM clock), so efficiency stays below 1. Refitted on the
7B PP=1 shapes: `compute_efficiency = 0.8494` (about 75 TFLOPs achieved),
`decode_memory_bandwidth_scale = 0.8383`; `collective_latency_scale = 1.824`
unchanged.

| regime | prefill | decode step | end-to-end |
|---|---|---|---|
| 7B PP=1 leave-one-out | | | 1.3% |
| 7B long context 4k-16k (node1) | 1.4% | 2.5% | 1.8% |
| 7B PP=2 | 14.6% | 3.8% | 1.6% |
| 7B TP=2 | 7.7% | 11.6% | 8.1% |
| **14B PP=2 (transfer)** | 13.4% | 7.2% | **4.9%** |

Caveats: 14B prefill is over-predicted at large batches, consistent with
vLLM overlapping pipeline stages across requests inside one batch (also seen
for 7B PP=2), which the simulator does not model. Lesson for every future
model: give the real MLP width; a wrong default is absorbed by a fitted
constant and only surfaces when the model changes.

## 12-13. RTX 3090 x2, Qwen2.5-7B, TP=2 and PP=2 with measured collective curves

- Runs: TP=2 [`lab-runs/2026-09-30-tp2/`](../lab-runs/2026-09-30-tp2/)
  (`real_tp2.jsonl`); PP=2 `lab-runs/2026-09-28-static-batch/real_pp2.jsonl`;
  NCCL sweep [`lab-runs/2026-09-30-nccl-curve/`](../lab-runs/2026-09-30-nccl-curve/)
  (`collective_curve.jsonl`, torch.distributed bf16 over bond0, 1 KiB-256 MiB).
  Comparison: `lab-runs/2026-09-30-tp2/report-collective-curves.md` and
  `comparison-collective-curves.json`, reproduced by
  `python3 lab-runs/2026-09-30-tp2/compare_collective_curves.py`. No
  manifest (pre-harness runs); simulator from the uncommitted
  `feat/measured-collective-curves` worktree.
- Cluster: `rtx3090_lab_cluster_measured_curves.toml` = the lab cluster with
  node0's NIC egress capped at 3.61 Gb/s (one-way link) plus
  `[[collective_curves]]` generated by `tools/lab/lab.py curves` (all_reduce
  over nodes 0+1; send_recv 1->0 measured; send_recv 0->1 measured at
  64-256 MiB and derived below, because the legacy benchmark timed node0's
  sends on the sender). Compute constants are the PP=1 fit (entry 2; pre-recalibration, equivalent for 7B); no
  network parameter is fitted.
- Entry 7 (the previous best) fitted `collective_latency_scale` on one TP=2
  shape; it missed batch-8 decode by -24.6% / -18.2% because the 57 KiB
  decode all-reduce sits just past NCCL's 32-64 KiB protocol step.
- Entry 12 also includes the TP trace fix that adds vLLM's vocab-parallel
  embedding all-reduce and LM-head logits all-gather (curves alone, old
  trace: 7.0% e2e). The all-gather has no measured curve in the legacy sweep,
  so it is priced by alpha-beta (`collective_curve_absent`).
- Caveats: batch-1 decode is +11.8% (the isolated 7 KiB all-reduce carries
  eager-launch overhead vLLM's CUDA-graph decode avoids); prefill is
  under-predicted 4-25% (in-situ 3.7 MB all-reduce ~1.4x the isolated
  benchmark). PP=2 gets slightly worse with curves (2.4% vs 1.6%
  asymmetric alpha-beta, 1.1% symmetric) because PP=2 prefill timings are
  inconsistent with a serial 0.35 GB/s node0->node1 transfer (PP=2 prefill is
  faster than PP=1 at batch 8), pointing at overlap/pipelining in vLLM's PP
  path that the simulator does not model.

## 14. Token-dependent compute efficiency (prefill token sweep) and static re-validation

- Data: [`lab-runs/2026-10-01-qwen7b-prefill-token-sweep/`](../lab-runs/2026-10-01-qwen7b-prefill-token-sweep/)
  (node0, Docker, batch-1 prefills at 16-4096 prompt tokens, manifest sha in
  `manifest.json`). Re-validation of every static regime:
  `lab-runs/2026-10-01-structural-calibration/` (`revalidate.py`,
  `revalidation.json`, `summary.md`; per-regime `validation-sc-<regime>-<variant>.json`
  in each measured run dir).
- Simulator: `feat/structural-calibration-gaps` at ec809ed. Changes:
  prefill bounded below by one weight read; optional
  `compute_efficiency_curve` evaluated at tokens per forward pass
  (`batch x prompt` for prefill, `batch` for a decode step, step tokens in
  the serving engine).
- Curve, fitted by `lab.py fit-curve` (tag `curve-only`) with the
  recalibrated scalars as base (88 TFLOPs peak, `decode_memory_bandwidth_scale
  = 0.8383`): 128: 0.654, 256: 0.618, 512: 0.724, 1024: 0.838, 2048: 0.847,
  4096: 0.874. 1x16, 1x32, 1x64 are weight-read bound (measured within 25%
  of the simulated 19.4 ms floor) and are not curve points. Efficiencies are
  relative to the simulator's FLOP count, which omits the untied LM head over
  prompt tokens, so they differ from a hand count.
- Variants: **before** = base-commit binary (941339e) + recalibrated scalar
  profile; **floor** = new binary + same scalars (weight-read floor only);
  **curve** = new binary + curve profile. PP=2 rows use the recalibration
  cluster (node0 NIC 3.61 Gb/s) and `collective_latency_scale = 1.824`.

| regime | variant | prefill | decode step | end-to-end |
|---|---|---:|---:|---:|
| prefill token sweep 1x16-1x4096 (in-sample for the curve) | before | 30.1% | 0.7% | 1.1% |
| | floor | 10.9% | 0.7% | 1.2% |
| | curve | **3.2%** | 0.7% | 1.4% |
| 7B PP=1, 5 shapes (held out) | before | 4.3% | 0.5% | 1.5% |
| | curve | **0.8%** | 0.5% | **0.9%** |
| 7B long context 4k-16k (node1, held out) | before | 1.4% | 2.5% | 1.8% |
| | curve | 4.2% | 2.5% | 3.1% |
| 7B decode sweep prefills 1-32 x 512 (node1, held out) | before | 5.7% | 2.7% | 2.4% |
| | curve | 6.0% | 2.7% | 3.0% |
| 7B PP=2 | before | 14.6% | 3.8% | 1.6% |
| | curve | **10.4%** | 3.8% | **1.3%** |
| 14B PP=2 (model transfer) | before | 13.4% | 7.2% | 4.9% |
| | curve | **8.7%** | 7.2% | 5.1% |

(The floor variant equals before on every regime except the sweep: only
prompts below ~100 tokens are weight-read bound.)

- Fixed: short-prompt prefill. 1x512 -14.9% -> -0.2% (PP=1), 1x16 -86% ->
  -11%, 14B 1x512 -14.2% -> -0.2%. Decode steps are unchanged everywhere
  (memory-bound; the clamped low-token efficiency only affects compute
  terms far below the weight read).
- Caveats: node1 runs large prefills ~3% slower than node0 (1x512 117.6 vs
  114.7 ms; 8x512 813 vs 781 ms), so the node0-fitted curve under-predicts
  node1 prefills by ~3% (long context -3.3% at 4k to -5.8% at 16k; the
  scalar happened to sit between the nodes). The long-context trend with
  context length predates the curve (-0.5% -> -3.0% before) and points at
  attention running below GEMM efficiency at 8k-16k. Batched small prefills
  on node1 (2x512, 3x512, 4x512) take nearly per-sequence time (2x512 =
  235 ms = 2 x 1x512) and stay 8-16% under with or without the curve; there
  is no node0 measurement of those shapes. 14B PP=2 e2e moves 4.9% -> 5.1%
  (decode-step errors, untouched here, dominate).

## 15. RTX 3090, Qwen2.5-7B, online serving with the efficiency curve and frontend latency

- Run: the entry 5/10 measurements
  ([`lab-runs/2026-09-30-serving-baseline/`](../lab-runs/2026-09-30-serving-baseline/));
  numbers in `validation-sc-serving-<variant>.json`.
- Simulator ec809ed. Calibration: entry 14's curve profile plus the
  frontend latency fitted by `lab.py fit-curve --frontend-dir
  lab-runs/2026-09-30-frontend` (`calibration_profile-curve.toml` in the
  prefill sweep dir): `frontend_latency_us = 5081`,
  `frontend_latency_per_prompt_token_us = 14.44` (isolated one-token requests
  at 0.5 req/s: TTFT 27.05 ms at 16 tokens and 126.88 ms at 512, minus the
  batch-1 static prefill 21.73 / 114.37 ms = 5.3 / 12.5 ms). Nothing is
  fitted to the serving data.

| variant | TTFT p50 | TPOT p50 | ITL p50 | E2EL p50 | tok/s | TTFT p50 at 1 / 2 / 4 req/s |
|---|---:|---:|---:|---:|---:|---|
| before (entry 10, recalibrated scalars) | 29.8% | 10.8% | 3.3% | 12.7% | 9.0% | -33% / -35% / -39% |
| curve | 21.3% | 6.5% | 2.9% | 7.3% | 8.5% | -23% / -25% / -26% |
| **curve + frontend** | **17.7%** | **6.5%** | **2.9%** | **7.1%** | **8.4%** | **-15% / -17% / -19%** |

| rate | TTFT p50 real / sim | TPOT p50 real / sim | ITL p50 real / sim | E2EL p50 real / sim | tok/s real / sim |
|---|---|---|---|---|---|
| 1 | 162.3 / 137.2 (-15%) | 21.8 / 21.8 (+0%) | 19.4 / 19.6 (+1%) | 2,911 / 2,909 (-0%) | 126 / 136 |
| 2 | 168.0 / 139.0 (-17%) | 24.7 / 24.4 (-1%) | 19.8 / 19.7 (-0%) | 3,309 / 3,240 (-2%) | 249 / 268 |
| 4 | 193.4 / 155.9 (-19%) | 35.9 / 33.7 (-6%) | 20.3 / 20.3 (+0%) | 4,783 / 4,528 (-5%) | 484 / 521 |
| 6 | 467.8 / 298.9 (-36%) | 62.6 / 52.7 (-16%) | 22.5 / 21.6 (-4%) | 8,635 / 7,327 (-15%) | 701 / 757 |
| 8 | 3,703 / 2,805 (-24%) | 76.4 / 66.7 (-13%) | 23.3 / 22.1 (-5%) | 11,688 / 9,535 (-18%) | 746 / 847 |
| 10 | 5,108 / 4,709 (-8%) | 72.9 / 66.2 (-9%) | 23.3 / 22.1 (-5%) | 13,537 / 12,715 (-6%) | 768 / 849 |
| inf | 13,207 / 12,692 (-4%) | 65.7 / 65.9 (+0%) | 23.3 / 22.1 (-5%) | 21,661 / 21,129 (-2%) | 818 / 853 |

- The curve closes ~17 ms of the low-load gap (512-token prefill step now
  priced at the measured 114 ms) and raises the price of every mixed step,
  which is what cut TPOT error at 4-8 req/s. The frontend adds 12.5 ms per
  512-token request.
- Remaining low-load gap: ~25 ms at 1 req/s. It is in-load interference
  (arrivals waiting for an in-flight decode step, mixed-step slowdown) the
  engine does not capture; near saturation (6 req/s) TTFT is still -36%
  because the simulated server is ~8% faster (throughput 757 vs 701 tok/s).

## 16. RTX 3090, Qwen2.5-7B, decode batch sweep: CUDA-graph capture sizes and KV exhaustion

- Data: `lab-runs/2026-10-01-qwen7b-decode-batch-sweep/` (node1, batch 1-40)
  and `-tail/` (32-64), 512-token prompts, 128 decode tokens,
  `max_num_batched_tokens = 32768`, utilization 0.85; validation on batch
  1-32 (`measured-b1-32.jsonl`, entry 14's decode row).
- CUDA-graph capture sizes (vLLM 0.29 captures 1, 2, 4, 8, 16, 24, 32, 40,
  ...; a batch is padded up to the next size): measured ms/step 1: 19.58, 2:
  19.25, 3: 19.75, 4: 19.62, 6: 20.06, 8: 20.08, 12: 20.64, 16: 20.97, 20:
  20.79, 24: 20.95, 28: 21.89, 32: 22.21 (repeat 21.63). Padded batches cost
  the same as the next captured size within noise (3 vs 4, 6 vs 8, 12 vs 16,
  20 vs 24, 28 vs 32 differ by < 0.4 ms), and there is no step at any capture
  boundary, because decode steps are weight-read bound and padding only adds
  compute far below the read. **Negligible on this hardware; not modeled.**
  It may matter on GPUs or models where decode is closer to compute-bound
  (large batches on H100 with small models).
- Decode step accuracy (batch 1-32): 2.7% mean |error|, but trending from
  -0.7% (batch 1) to -6.5% (batch 32): the real step grows 2.6 ms from batch
  1 to 32 while the simulated KV read adds 1.35 ms, so paged KV reads run at
  roughly half the weight-read bandwidth. Same signature as entry 9's
  long-context decode drift. Candidate fix: a separate KV-read bandwidth
  scale (not done here).
- KV exhaustion, not padding, causes the jump at batch >= 40 (30.4 ms at 40,
  ~53 ms at 48-64, 2.4x): vLLM logged a KV cache of 24,896 tokens for these
  settings (the 32,768-token budget reserves 4.28 GiB of peak activation),
  and 40 x 641 = 25,640 tokens exceeds it, so vLLM preempts and recomputes.
  **Gap: the serving engine never preempts** (it reserves KV at admission
  and queues), so it cannot reproduce this regime.
- KV capacity estimate: the harness's fallback (used without a server.log)
  ignored vLLM's activation reserve and was 9% to 262% high. The new
  `kv_estimate.py` model is within 10% of every capacity vLLM logged:

| run | max_num_batched_tokens | vLLM logged | new estimate | old estimate |
|---|---:|---:|---:|---:|
| serving baseline (util 0.85) | 2,048 | 82,864 | 85,565 (+3.3%) | 90,157 (+8.8%) |
| decode batch sweep | 32,768 | 24,896 | 27,356 (+9.9%) | 90,157 (+262%) |
| prefill token sweep | 16,384 | 63,552 | 58,401 (-8.1%) | 90,157 (+42%) |
| long context node1 (util 0.90) | 16,384 | 85,568 | 80,459 (-6.0%) | 111,083 (+30%) |
| 14B PP=2 (per stage) | 16,384 | 43,440 | 42,547 (-2.1%) | n/a |

  The simulator itself has no KV-capacity model (serving capacity comes from
  `max_resident_tokens` / `max_kv_blocks`); every recorded validation uses
  vLLM's logged capacity, so none of their numbers change.

## 17. RTX 3090, Qwen2.5-7B, static batch with the GPU clock locked at 1200 MHz

Data: `lab-runs/2026-10-01-qwen7b-static-node2-clock1200/`, predictions in
`lab-runs/2026-10-01-degraded-sim/` (cluster with `peak_f16_tflops` scaled from
88 to 50.29 = 88 x 1200/2100; memory bandwidth unchanged; recalibrated
constants; nothing refitted). The clock was locked with `nvidia-smi -lgc`
through `tools/lab/remote/run_with_gpu_clock.sh`, which always resets it.

| shape | prefill slowdown real / sim | decode/step full clock -> 1200 MHz real (sim) |
|---|---|---|
| 1x512 | 1.55x / 1.49x | 19.26 -> 24.30 ms (19.45) |
| 1x2048 | 1.54x / 1.75x | 19.34 -> 24.34 ms (19.56) |
| 8x512 | 1.52x / 1.75x | 19.75 -> 26.47 ms (19.75) |
| 8x2048 | 1.53x / 1.75x | 20.76 -> 27.80 ms (20.65) |
| 32x512 | 1.52x / 1.75x | 20.77 -> 27.07 ms (20.76) |

Mean |error|: prefill 13.8%, decode step 22.8%, end-to-end 9.6% (the two
errors partly offset). Findings: (1) under sustained prefill the unlocked GPU
does not run at its 2100 MHz maximum: the 1.53x slowdown implies ~1835 MHz
sustained, so clock-scaled peaks must use the sustained clock, not the
maximum; (2) decode slows 1.26x with memory clocks untouched, because the
achievable memory bandwidth depends on the SM clock. A degraded-GPU state
therefore needs both a compute factor (relative to the sustained clock) and a
bandwidth factor; scaling peak FLOPs alone misses decode entirely.

## 18. RTX 3090 x2, Qwen2.5-7B, disaggregated prefill/decode serving (NIXL)

Data: `lab-runs/2026-10-01-disagg-1p1d/` (rate_*.json from `vllm bench serve`
through the proxy; prefill/decode/proxy logs, including vLLM's per-transfer
KV metrics). Blind predictions: `lab-runs/2026-10-01-disagg-sim/` (made
before the measurement was read). Setup: prefill on node0, decode on node1,
vLLM 0.29.0 with NixlConnector (nixl-cu13 1.4.1, `UCX_TLS=tcp,cuda_copy,self,sm`
on bond0), the toy-proxy pattern (prefill max_tokens=1 non-streaming, then
decode streaming with the returned kv_transfer_params), the colocated
baseline's traffic (512 in / 128 out, 200 requests). Scripts:
`tools/lab/remote/run_disagg.sh`, `tools/lab/remote/disagg_proxy.py`.

| rate | TTFT p50 real / scalar / curve+frontend | TPOT p50 real / sim | out tok/s real / sim | colocated TTFT / TPOT |
|---|---|---|---|---|
| 1 | 317 / 226 (-29%) / 256 (-19%) | 19.65 / 19.56 | 126 / 136 | 162 / 21.8 |
| 2 | 331 / 227 (-31%) / 258 (-22%) | 21.23 / 19.66 | 249 / 268 | 168 / 24.7 |
| 4 | 337 / 234 (-31%) / 274 (-19%) | 21.68 / 19.91 | 480 / 520 | 193 / 35.9 |
| 6 | 423 / 311 (-26%) / 391 (-7%) | 20.95 / 20.14 | 692 / 754 | 468 / 62.6 |
| 8 | 647 / 589 (-9%) / 639 (-1%) | 21.01 / 20.34 | 890 / 970 | 3,703 / 76.4 |
| burst | 19,777 / 10,256 / 10,309 (-48%) | 21.75 / 20.54 | 795 / 1,139 | 13,207 / 65.7 |

- Reproduced: decode TPOT stays flat at ~21 ms (within 0-8%) while colocated
  TPOT degraded to 76 ms, i.e. the simulator captures that disaggregation
  removes prefill/decode interference; TTFT within 1-22% below saturation with
  the curve + frontend profile (9-31% with scalars only).
- KV transfer per request: vLLM logs ~95-115 ms average (P90 ~120-135 ms) for
  28 MB at low load; the simulator prices 81 ms from the node0->node1 send
  curve's derived (unmeasured, sender-timed) region below 64 MiB. Next: the
  receiver-timed collective bench.
- Burst: per-transfer time balloons (290 ms, then 2.8 s average, P90 4.8 s)
  while aggregate KV throughput stays near the link (~300 MB/s), i.e. the link
  is fair-shared across many concurrent pulls; the simulator's FIFO link
  queues get aggregate throughput roughly right but serialize transfers, so
  burst TTFT and throughput are off (-48% / +43%). Fair-share link modeling
  is the fix.
- Remaining low-load TTFT gap (~60 ms) is consistent with the proxy's two
  HTTP hops, which the simulator does not model
  (`disaggregated_proxy_hop_not_modeled`).

## Untested regimes

Nothing below has been measured; treat simulator output there as
unvalidated.

- Tensor parallelism (TP >= 2), within a node (no NVLink/PCIe P2P lab yet) or
  across nodes.
- Pipeline parallelism beyond 2 stages, and PP=2 with a PP=2-fitted profile.
- Expert parallelism / MoE models; data parallelism / multiple replicas.
- Models other than Qwen2.5-7B (larger dense models, GQA ratios, FFN sizes;
  the fit leaves `ffn_hidden_size` at the simulator default).
- Prompt lengths outside 512-2048, batch sizes above 32, decode lengths other
  than 128 (static), and any input/output mix other than 512/128 (serving).
- Prefix caching on, speculative decoding, quantized weights or KV cache
  (fp8/int8), eager mode (CUDA graphs off).
- Serving with TP/PP (the iteration engine prices TP all-reduces and PP
  stages per step but is unmeasured there), disaggregated prefill/decode
  (now on the iteration engine with decode-initiated KV pulls; predictions
  for the lab case are in docs/features/disaggregated_serving_engine.md),
  KV transfer between nodes,
  heterogeneous nodes, trace-driven or bursty arrivals, SLO metrics.
- Other GPUs (A100, H100, L40S, consumer cards other than the 3090) and
  other stacks (SGLang, TensorRT-LLM) or vLLM versions other than 0.29.0.
- Tensor parallelism within a node (no NVLink/PCIe P2P lab yet), TP > 2, and
  TP=2 across nodes beyond the five static shapes of entries 7 and 12.
- Network-bound regimes beyond two nodes: all_gather, reduce_scatter and
  all_to_all curves, more than two ranks, and receiver-timed node0->node1
  sends are unmeasured (the collective-bench mode in `tools/lab` measures
  them).
