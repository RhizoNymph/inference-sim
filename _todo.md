# Accuracy roadmap: 3090 lab now, H100s later

Goal: remove every error that does not depend on the GPU while the 3090 lab is
cheap, and make measurement-to-profile turn-key, so H100 time goes only to what
needs H100s (NVLink, InfiniBand, large/MoE models, high-throughput regimes).

"Ready for H100s" means: after calibration on the 3090s, every testable regime
is under ~10% error, and residual errors show no trend with batch size,
context length, or node count. Trending errors are model bugs; flat errors are
constants an H100 run can fit.

## Current state (2026-09-29)

- Single node, static batches, Qwen2.5-7B bf16, vLLM 0.29: 1.3% end-to-end
  error on held-out shapes after fitting `compute_efficiency = 0.85` and
  `decode_memory_bandwidth_scale = 0.84` (uncalibrated: 37%).
- Two-node pipeline parallel over 10GbE: 1.1% end-to-end after the pipeline
  fix (was 50%). Prefill still 14% off; network cost barely mattered in this
  test.
- Untested: serving under load, tensor parallelism, disaggregation, MoE, long
  contexts, other GPUs, heterogeneous/degraded topology, the AISimulate-derived
  H100 profile against real H100s.

## Before H100s (3090 lab)

- [x] 1. [done 2026-09-30, uncommitted in staging: iteration engine; low-load
      TTFT still 33-39% under, see progress log] Serving loop: run `vllm bench serve` at several request rates;
      compare TTFT, TPOT, ITL and throughput with the simulator's serving mode.
      Expected gap: the simulator schedules all prefill before all decode,
      while vLLM interleaves chunked prefill with decode steps. Likely fix is
      the authoritative event loop from goals.md.
- [x] 2. [done 2026-09-30, uncommitted in staging: TP trace fixes +
      measured collective curves + one-way-asymmetric links, 4.0% e2e all
      held out, validation ledger entry 12] Tensor parallelism across nodes
      over Ethernet
      (TP=2, node0+node1):
      the real test of the collective and topology model, since TP crosses
      the network every layer. Expected fixes: one-way-slower links
      (node0->node1 is 3.6 Gb/s vs 9.4 Gb/s back) and a small-message regime
      in the collective cost model.
- [ ] 3. [simulator side done 2026-10-01 on feat/disaggregated-serving-engine:
      disaggregated pools run on the iteration engine with NIXL-style
      decode-initiated KV pulls; reference workload
      examples/rtx3090_qwen7b_disaggregated_workload.toml, predictions in
      docs/features/disaggregated_serving_engine.md; measurement pending]
      Disaggregated prefill/decode across nodes (1 prefill + 1-2 decode)
      with a vLLM KV connector, to validate the KV-transfer model for the
      first time. Needs ~30 GB freed on node2 for the 3-node variant.
- [x] 4. [done 2026-09-30, uncommitted in staging: tools/lab, first real
      runs succeeded] Turn-key measurement pipeline: unattended benchmark harness, a fitter
      that emits a complete calibration TOML with holdout stats and feature
      ranges, and a committed validation ledger recording measured accuracy
      per regime.
- [x] 5. [done 2026-09-30: long context 1.8%, 14B model 4.9% after fixing
      model FLOP inputs] Calibration transfer: keep the 3090 constants and test a different
      model (Qwen2.5-14B across 2 nodes), long contexts (8k-16k) and larger
      batches. Tells us which constants are per-GPU vs per-model, which sizes
      the H100 plan.
- [ ] 6. Small structural gaps already visible: per-iteration overhead floor
      (short prefill 16% under), pipeline microbatching inside one
      multi-request batch (0.85x measured vs 1.03x simulated), decode batch
      non-linearity around CUDA-graph capture sizes.
- [ ] 7. Degraded states: throttled NIC, lowered GPU clocks, node removed
      mid-run. Blocked: `tc` and `nvidia-smi -lgc` need root on the nodes and
      there is no passwordless sudo.
- [ ] Write the H100 plan before getting hardware: fixed matrix of models x
      parallelism x shapes x rates, unattended scripts, time budget.

## Progress log

- 2026-09-30 serving baseline (node0, Qwen2.5-7B, 512 in / 128 out, rates
  1-10 req/s + burst): saturates at ~6 req/s (~800 output tok/s). TPOT rises
  21.8 -> 76 ms while median ITL only 19.4 -> 23.4 ms: decode steps that carry
  a prefill chunk are slow and inflate TPOT. This is the interleaving the
  simulator's prefill-first schedule cannot express. Data:
  lab-harness/lab-runs/2026-09-30-serving-baseline/.
- 2026-09-30 TP=2 over 10GbE: real prefill is ~5x slower than one GPU;
  simulator predicted 0.5x (75% e2e error). Fixed: two all-reduces per layer,
  full activation per rank, collective latency paid per token crossing, TP
  collectives block the next layer by default
  (`allow_compute_comm_overlap = false`). With node0's NIC set to the measured
  3.61 Gb/s and `collective_latency_scale = 1.824` fitted on one shape:
  held-out e2e error 9.6% (prefill 7.5%, decode 14.5%). PP=1 1.4% and PP=2
  1.6% e2e with the same network settings (no regression; PP is an
  independent check of the network fit). Data:
  lab-harness/lab-runs/2026-09-30-tp2/.
- [resolved by measured collective curves, see 2026-09-30 entry below]
  Batch-8 TP decode was 18-25% under (57 KB all-reduces past NCCL's 32-64 KB
  protocol step).
- 2026-09-30 calibration transfer, long contexts: the same 3090 constants
  (fitted at 512-2048 prompts, node0, Docker) predict 4k-16k prompts on node1
  (native venv) at 0.8% e2e, 0.6% prefill. Decode drifts -0.9% -> -4.4% as
  total KV grows: KV-read slightly undercounted at large caches (watch item).
  First real run of the lab harness; found and fixed a logging crash
  (reserved LogRecord key) that dry runs could not hit.
- 2026-09-30 serving iteration engine (workstream A, merged into staging):
  colocated continuous batching now runs on a vLLM-V1-style step loop.
  Mean error over 7 rates: TTFT p50 30%, TPOT p50 11%, ITL p50 3%,
  throughput 9% (was 101% / 354% / 651% / 14%). Requests queue instead of
  being rejected; 200-request run takes 0.1 s. Remaining: low-load TTFT
  33-39% under (about 17 ms from 512-token prefill under-prediction, about
  35 ms assumed frontend latency, to be measured). Disaggregated pools still
  use the old phase-pipeline scheduler.
- 2026-09-30 model transfer and recalibration: Qwen2.5-14B PP=2 first came out
  at 22.8% e2e (prefill 70%). Root cause: specs gave parameters_gb but not
  ffn_hidden_size, so FLOPs came from a 4x-hidden default (7B undercounted
  18%, 14B overcounted 33%); the fitted compute_efficiency had absorbed the
  7B error. Fixed inputs (real MLP widths) and set the 3090 peak to 88
  TFLOPs (2100 MHz max clock). Refit: compute_efficiency 0.8494, 14B e2e
  4.9%, 7B results unchanged. Harness fitter reproduces the refit exactly.
- 2026-09-30 frontend latency measured (lab-runs/2026-09-30-frontend/,
  isolated requests at 0.5 req/s): TTFT 126.9 ms for a 512-token prompt vs
  114.7 ms engine-only prefill, and 27.0 ms for 16 tokens vs ~19 ms one
  forward, so API-server overhead is ~8-12 ms, not the ~35 ms assumed.
  The serving engine's low-load TTFT gap (~53 ms at 1 req/s) therefore
  splits ~17 ms prefill under-prediction (short-prompt GEMM efficiency),
  ~10 ms frontend (add as a measured constant), and ~25 ms in-load
  interference the engine does not yet capture (arrival waiting on in-flight
  decode steps / mixed-step slowdown). Next: model the frontend constant,
  then compare per-request TTFT decomposition at 1 req/s.
- TODO (small, recommended): warn when a derived parameter count disagrees
  with parameters_gb (e.g. >10%), since a wrong ffn default is otherwise
  silently absorbed by fitted constants.
- TODO: short-prompt prefill is 16% under (1x512) and is not a per-step
  overhead (fitted overhead is 0.04 ms). Likely GEMM efficiency falling at
  small token counts; a token-count-dependent efficiency curve (or
  AISimulate-style GEMM tables) would capture it.
- [resolved: per-direction NIC/link bandwidth] The simulator could not
  represent a link slower in one direction.
- 2026-09-30 measured collective curves (workstream B, merged into
  staging): cluster TOML [[collective_curves]] (log-log interpolation per
  op/scope/rank count, directed point-to-point), per-direction NIC/link
  bandwidth, and vLLM's vocab-parallel embedding all-reduce + logits
  all-gather in the TP trace. TP=2 4.0% e2e with no fitted network scalar
  (batch-8 decode fixed); PP=2 2.4% (was 1.6%; within run spread).
- Open from collectives: re-run collective-bench with receiver-timed
  node0->node1 sends and all_gather/reduce_scatter/all_to_all curves; TP
  prefill all-reduce runs ~1.4x the isolated benchmark in situ; batch-1
  decode all-reduce carries eager overhead CUDA graphs avoid.
- [resolved 2026-09-30] The serving engine's per-step TP pricing now
  includes the embedding all-reduce and the logits all-gather (sized by
  sampling sequences, i.e. decodes plus prompt-completing chunks), so an
  engine decode step again equals the static one-token decode under TP.
- Open: vLLM overlaps pipeline stages across requests inside one batch
  (7B and 14B PP=2 prefill faster than serial); not modeled.

## After H100s (ordered by information per H100-hour)

- [ ] 1. Validate the AISimulate-derived H100 profile
      (`examples/calibration_h100_vllm_llama31_70b.toml`) against real
      silicon. Decides whether we need to measure most other GPUs ourselves.
- [ ] 2. Intra-node tensor parallelism over NVLink/NVSwitch (TP 2/4/8), and
      FP8.
- [ ] 3. Multi-node over InfiniBand with GPUDirect: rails, NIC affinity,
      NUMA. Fit allreduce curves per fabric with `nccl-tests`.
- [ ] 4. Large and MoE models (70B, Mixtral/DeepSeek-class): expert
      parallelism and MoE all-to-all have no test today.
- [ ] 5. High-throughput regimes: batch 128-256, 32k+ contexts, real request
      traces.
- [ ] 6. Heterogeneous pools if the 3090s can join (e.g. H100 prefill with
      3090 decode).

## Lab notes

- Nodes: node0/node1/node2, 1x RTX 3090 each, 2x10GbE LACP bond (`bond0`).
- node0: snap Docker, use `--runtime=nvidia`; shares its GPU with the
  `tei-nomic` embedding container. node1/node2: no NVIDIA container runtime,
  run natively from `~/inference-sim-lab/.venv` (vllm 0.29.0, torch 2.13.0,
  ray 2.58.0).
- Override `HF_HOME=$HOME/.cache/huggingface` per process (node0's default
  points at an unmounted NAS path; node1's at `~/Models`).
- node2 has ~9 GB free disk.
