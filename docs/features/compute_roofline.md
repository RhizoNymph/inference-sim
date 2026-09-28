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
- Sharding: dense FLOPs and weight bytes divide by
  `tensor * pipeline * expert` ranks; attention FLOPs and KV reads divide by
  `tensor * pipeline` only, because expert ranks partition MLP expert
  weights, not attention heads or KV cache.
- Effective peaks: minimum per-placement GPU peak TFLOPs (dtype-selected)
  times `calibration.compute_efficiency`; minimum HBM bandwidth times
  `calibration.decode_memory_bandwidth_scale`.

## Non-scope

Layer-aware kernel decomposition (attention vs MLP vs logits), tensor-core
utilization curves, batch-roofline nonlinearity beyond the two-term max,
paged-attention block effects, speculative decoding, prefix-cache-aware
decode reads, pipeline bubbles, and any backend-specific behavior. Fitted
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
   costs (src/solver/network_cost.rs).
5. Serving derives per-token decode cost from a `decode_tokens = 1` score
   and a whole-decode score (`decode_tail_scale`, src/serving/scheduling.rs),
   so the context-dependent KV-read term makes later decode iterations
   costlier than the first.

## Related files

- `src/solver.rs` — `estimate_compute_latency_s`, `prefill_baseline_s`,
  `decode_compute_latency_s`, `flop_latency_s`, `attention_shard_factor`,
  `prefill_attention_flops`, `decode_attention_flops`,
  `decode_kv_read_bytes`, `average_decode_context_tokens`,
  `effective_peak_flops`, `effective_hbm_bandwidth`.
- `src/workload.rs` — `ModelSpec` (layers, hidden size, heads, kv_heads,
  dtypes, parameter count) and `InferenceRequest` shapes.
- `src/calibration.rs` — `SimulationCalibration` scales applied to the
  roofline.
- `src/solver/calibration_fits.rs` — fitted-model override path.
- `src/solver/operations.rs` — per-layer operation trace built from the
  scalar phase latency.

## Invariants and constraints

- Attention terms use the mean decode context so total decode work stays
  closed-form; they never depend on scheduler state.
- Expert ranks must never shard attention FLOPs or KV reads.
- KV-read bytes use `kv_dtype()` (falls back to model dtype) and `kv_heads`,
  so GQA/MQA and quantized KV reduce the memory term.
- Calibration fits, when applicable, fully replace the baseline for that
  phase; baselines are still recorded as fit `baseline_s` evidence.
- `kv_cache_bytes` (capacity feasibility) remains sized by
  `max_sequence_tokens`; the bandwidth term is sized by actual
  prompt/decode tokens. These are intentionally different denominators.
