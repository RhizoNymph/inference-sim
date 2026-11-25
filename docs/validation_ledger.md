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
| 5 | 2026-09-30 | 1x RTX 3090 24 GB | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0 | TP1 PP1 | serving, Poisson 1-10 req/s + inf | PP1 profile (transfer) | see entry | see entry | see entry | **not accurate** |
| 6 | 2026-09-30 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, Ray | TP2 PP1 | static batch | PP1 profile + alpha-beta, node0 NIC 3.61 Gb/s, `collective_latency_scale` 1.8238 fitted on 1x512 | 7.8% | 11.6% | 8.1% (9.6% held out) | superseded by 7 |
| 7 | 2026-09-30 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, Ray | TP2 PP1 | static batch | PP1 profile + measured NCCL curves, no network fit | 12.5% | 7.2% | **4.0%** (all held out) | validated, with caveats |
| 8 | 2026-09-30 | 2x RTX 3090, 2 nodes, 2x10GbE | Qwen2.5-7B-Instruct bf16 | vLLM 0.29.0, Ray | TP1 PP2 | static batch | PP1 profile + measured NCCL curves | 15.3% | 3.5% | 2.4% | validated, with caveats; entry 4 (symmetric alpha-beta) is 1.1% |

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

## 5. RTX 3090, Qwen2.5-7B, online serving (vllm bench serve), PP=1

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
  with B = 64 a sequence costs decode_one(64)/64 ~ 0.35 ms, so low-load steps
  are ~50x too cheap (that run also did not finish 7 rates within 10 minutes:
  cheap iterations mean very many of them). No single B fits a rate sweep;
  Little's law is the least-bad choice.
- The same linear scaling makes simulated concurrency run away (at 4 req/s
  the simulation peaks at 120 concurrent sequences; vLLM peaked at 37),
  which trips decode-capacity and KV rejections. vLLM instead queues (TTFT grows, as measured at 8-10 req/s);
  the simulator has no admission queue for decode capacity, so it rejects the
  candidate.
- Median ITL in vLLM stays ~19-23 ms at every load (steps are nearly flat in
  batch) while p99 ITL jumps to 130-400 ms when a chunked prefill shares the
  step; the simulator's ITL tracks its inflated TPOT instead.

## 6-8. RTX 3090 x2, Qwen2.5-7B, TP=2 and PP=2 with measured collective curves

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
  sends on the sender). Compute constants are the PP=1 fit (entry 2); no
  network parameter is fitted.
- Entry 6 (the previous best) fitted `collective_latency_scale` on one TP=2
  shape; it missed batch-8 decode by -24.6% / -18.2% because the 57 KiB
  decode all-reduce sits just past NCCL's 32-64 KiB protocol step.
- Entry 7 also includes the TP trace fix that adds vLLM's vocab-parallel
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

## Untested regimes

Nothing below has been measured; treat simulator output there as
unvalidated.

- Tensor parallelism within a node (no NVLink/PCIe P2P lab yet), TP > 2, and
  TP=2 across nodes beyond the five static shapes of entries 6-7.
- Pipeline parallelism beyond 2 stages, and PP=2 with a PP=2-fitted profile.
- Expert parallelism / MoE models; data parallelism / multiple replicas.
- Models other than Qwen2.5-7B (larger dense models, GQA ratios, FFN sizes;
  the fit leaves `ffn_hidden_size` at the simulator default).
- Prompt lengths outside 512-2048, batch sizes above 32, decode lengths other
  than 128 (static), and any input/output mix other than 512/128 (serving).
- Prefix caching on, speculative decoding, quantized weights or KV cache
  (fp8/int8), eager mode (CUDA graphs off).
- Serving with TP/PP, disaggregated prefill/decode, KV transfer between
  nodes, heterogeneous nodes, trace-driven or bursty arrivals, SLO metrics.
- Other GPUs (A100, H100, L40S, consumer cards other than the 3090) and
  other stacks (SGLang, TensorRT-LLM) or vLLM versions other than 0.29.0.
- Network-bound regimes beyond two nodes: all_gather, reduce_scatter and
  all_to_all curves, more than two ranks, and receiver-timed node0->node1
  sends are unmeasured (the collective-bench mode in `tools/lab` measures
  them).
