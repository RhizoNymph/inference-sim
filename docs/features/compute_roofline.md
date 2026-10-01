# Compute Roofline

## Scope

The analytical per-phase compute/memory latency model used to price prefill
and decode operations before scheduling. It covers:

- Dense parameter FLOPs: `2 * parameter_count * active_tokens`.
- Causal attention FLOPs (QK^T and AV bilinear work, which is not part of
  the parameter-count term): `2 * layers * hidden * prompt_tokens^2 * batch`
  for prefill; `4 * layers * hidden * avg_context * decode_tokens * batch`
  for decode, where `avg_context = prompt + (decode + 1) / 2`.
- Decode HBM bandwidth: per-step weight reads
  (`parameter_bytes_per_rank * decode_tokens / bandwidth`) plus KV-cache
  reads (`2 * layers * kv_heads * head_dim * kv_bytes * avg_context *
  decode_tokens * batch`, sharded, over bandwidth). Decode latency is
  `max(flop_term, memory_term) * decode_compute_scale`.
- Sharding: phase latency is the time for one batch to pass through every
  pipeline stage, so latency divides dense FLOPs and weight bytes by
  `tensor * expert` ranks and attention FLOPs and KV reads by `tensor` ranks
  only. Pipeline ranks split a batch's layers across stages that run one
  after another, so they shard weight and KV memory but not one batch's
  latency. Expert ranks partition MLP expert weights, not attention heads
  or KV cache.
- Effective peaks: minimum per-placement GPU peak TFLOPs (dtype-selected)
  times `calibration.compute_efficiency`; minimum HBM bandwidth times
  `calibration.decode_memory_bandwidth_scale`.

## Non-scope

Layer-aware kernel decomposition (attention vs MLP vs logits), tensor-core
utilization curves, batch-roofline nonlinearity beyond the two-term max,
paged-attention block effects, speculative decoding, prefix-cache-aware
decode reads, backend microbatch scheduling inside one pipelined batch,
and any backend-specific behavior. Fitted
calibration models override these baselines when a profile matches.

## Data/control flow

1. `Solver::score_config_with_options` (src/solver.rs) validates a
   `ParallelismConfig`, places ranks, then calls
   `Solver::estimate_compute_latency_s`.
2. Per `InferencePhase`:
   - Prefill: `Solver::prefill_baseline_s` = (dense FLOPs / shard +
     `prefill_attention_flops` / attention shard) / effective peak FLOPs,
     scaled by `prefill_compute_scale`.
   - Decode: `Solver::decode_compute_latency_s` computes the FLOP term
     (dense + `decode_attention_flops`) and the memory term (weight reads +
     `decode_kv_read_bytes` / attention shard / bandwidth) and takes the max.
   - EndToEnd: prefill + decode.
3. If a calibration profile supplies a matching phase fit
   (`Solver::fitted_phase_latency`, src/solver/calibration_fits.rs), the
   fitted value replaces the baseline and the application is recorded.
4. The scalar phase latency is split evenly across layers into scheduled
   operations (src/solver/operations.rs) and interleaved with collective
   costs (src/solver/network_cost.rs). Layers map to contiguous pipeline
   stages (`layer_stage`); each layer occupies only its stage's GPU
   resources, tensor/expert collectives run only for groups inside that
   stage, and a `pipeline sendrecv edge N` operation sits between the last
   layer of stage N and the first layer of stage N+1. The boundary is
   crossed once for the prompt and once per generated token, so its latency
   term repeats per crossing. Because stages hold separate resources,
   concurrent serving batches can overlap across stages. Each tensor-parallel
   layer issues two all-reduces (after attention and after the MLP), each
   moving the full per-rank activation (`batch * hidden * dtype` per crossing
   token) with the same per-crossing repetition; expert all-to-alls use
   `activation * top_k / expert_ranks` per crossing token
   (`push_repeated_collective_operation`, `activation_crossings`). With
   tensor parallelism the trace also carries the two vocab-parallel
   collectives vLLM issues once per forward pass: an `embedding tp
   all-reduce` (activation-sized, before layer 0 on the first stage, which
   layer 0 depends on) and an `lm_head logits all-gather` after the last
   layer on the last stage, contributing each rank's logits shard
   (`batch * ceil(vocab / tensor) * dtype`) once for the prompt pass and
   once per generated token (`sampled_logit_crossings`,
   `logits_shard_bytes`). Collective times come from measured curves when the
   cluster lists one (docs/features/collective_curves.md). By default
   the next layer waits for the layer's last collective, because tensor and
   expert parallel results are data dependencies;
   `calibration.allow_compute_comm_overlap = true` lets compute overlap them,
   modeling engines that pipeline communication with compute.
5. The serving iteration engine prices each engine step with
   `IterationCostModel` (src/solver/step_cost.rs), the same primitives
   applied to a step's composition: dense FLOPs for every token, causal
   attention per prefill chunk over its context
   (`2 L H ((p + c)^2 - p^2)`), decode attention over each sequence's
   context, and one weight read plus every active sequence's KV read, with
   per-step TP all-reduces and PP send/recvs sized by the step's tokens. A
   pure-decode step at context `prompt + 1` equals the `decode_tokens = 1`
   score; a compute-bound pure-prefill step equals the prefill score (see
   docs/features/serving_iteration_engine.md).
6. The phase-pipeline serving scheduler derives per-token decode cost from a
   `decode_tokens = 1` score and a whole-decode score (`decode_tail_scale`,
   src/serving/scheduling.rs), so the context-dependent KV-read term makes
   later decode iterations costlier than the first.

## Related files

- `src/solver.rs` — `estimate_compute_latency_s`, `prefill_baseline_s`,
  `decode_compute_latency_s`, `flop_latency_s`, `latency_shard_factor`,
  `attention_shard_factor`, `kv_cache_bytes`,
  `prefill_attention_flops`, `decode_attention_flops`,
  `decode_kv_read_bytes`, `average_decode_context_tokens`,
  `effective_peak_flops`, `effective_hbm_bandwidth`.
- `src/workload.rs` — `ModelSpec` (layers, hidden size, heads, kv_heads,
  dtypes, parameter count) and `InferenceRequest` shapes.
- `src/calibration.rs` — `SimulationCalibration` scales applied to the
  roofline.
- `src/solver/calibration_fits.rs` — fitted-model override path.
- `src/solver/operations.rs` — per-layer operation trace built from the
  scalar phase latency: `build_operation_trace`, `layer_stage`,
  `PipelineStagePlan` (per-stage ranks and compute resources), and
  `push_pipeline_boundary_operation` (per-crossing stage sends).
- `src/serving/memory.rs` — serving memory components; `kv_cache_gb`
  divides KV by tensor and pipeline ranks.
- `src/solver/step_cost.rs` — per-engine-step pricing for serving:
  `IterationCostModel`, `StepWork`, `StepLatency`, `StepCostError`.

## Invariants and constraints

- Attention terms use the mean decode context so total decode work stays
  closed-form; they never depend on scheduler state.
- Expert ranks must never shard attention FLOPs or KV reads.
- Pipeline ranks must never shorten one batch's latency; they divide weight
  and KV memory per GPU (`kv_cache_bytes`, serving `kv_cache_gb`) only.
- Calibration fits are whole-model per-batch latencies, so they apply
  unchanged at any pipeline degree.
- Collective latency is paid once per activation crossing (prompt, then each
  generated token), never once per phase.
- `allow_compute_comm_overlap` defaults to false: tensor/expert collectives
  sit on the critical path unless a profile opts in.
- KV-read bytes use `kv_dtype()` (falls back to model dtype) and `kv_heads`,
  so GQA/MQA and quantized KV reduce the memory term.
- Calibration fits, when applicable, fully replace the baseline for that
  phase; baselines are still recorded as fit `baseline_s` evidence.
- `kv_cache_bytes` (capacity feasibility) remains sized by
  `max_sequence_tokens`; the bandwidth term is sized by actual
  prompt/decode tokens. These are intentionally different denominators.
