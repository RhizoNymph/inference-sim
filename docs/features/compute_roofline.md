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
- Prefill weight-read floor: one prefill pass still reads every weight once,
  so prefill latency is `max(compute, weight_bytes_per_rank / bandwidth *
  decode_compute_scale)`. Short prompts (below ~100 tokens for a 7B model on
  a 3090) are weight-read bound, exactly like the serving step cost.
- Effective peaks: minimum per-placement GPU peak TFLOPs (dtype-selected)
  times the compute efficiency for the pass; minimum HBM bandwidth times
  `calibration.decode_memory_bandwidth_scale`.
- Token-dependent compute efficiency: `SimulationCalibration::
  compute_efficiency_at(tokens_per_pass)` returns the optional
  `calibration.compute_efficiency_curve` interpolated at the pass's token
  count, or the scalar `compute_efficiency` when no curve is set. Tokens per
  pass are `batch x prompt` for a static prefill, `batch` for a decode step,
  and the step's total tokens (prefill chunk tokens plus decodes) in the
  serving engine. Tensor parallelism does not change the token count (each
  rank's GEMMs keep the full token dimension).
- Parameter-count consistency: when the workload gives `parameters_gb` but
  neither `ffn_hidden_size` nor `parameter_count_billion`, FLOPs come from a
  default MLP width of 4 x hidden. If that derived count differs from
  `parameters_gb / dtype_bytes` by more than 10%, every scored config carries
  the approximation `model_parameter_count_mismatch` (phase `model`, category
  `model`) naming both counts and the fix.

### Compute-efficiency curve

```toml
[calibration]
compute_efficiency = 0.8494                  # still the default without a curve
compute_efficiency_curve = [[128, 0.654], [256, 0.618], [512, 0.724],
                            [1024, 0.838], [2048, 0.847], [4096, 0.874]]
```

Points are `[tokens per forward pass, efficiency]`. Validation (config
error otherwise): at least 2 and at most 32 points, strictly increasing
positive token counts, efficiencies finite in `(0, 1]`. Evaluation is
piecewise linear in `ln(tokens)` and clamps to the first point's efficiency
below the range and the last point's above it. The curve is accepted in a
workload `[calibration]`, a calibration profile `[calibration]` (a workload
curve overrides the profile's), and a run scenario's `[scenarios.calibration]`.
The JSON `calibration` block reports `compute_efficiency_curve` (or null).
`tools/lab/lab.py fit-curve` fits it from a batch-1 prefill token sweep
(docs/features/lab_harness.md).

## Non-scope

Layer-aware kernel decomposition (attention vs MLP vs logits; the
efficiency curve applies one efficiency to dense and attention FLOPs alike),
per-kernel tensor-core utilization tables (the curve is one aggregate
efficiency per token count, not a GEMM-shape table), batch-roofline
nonlinearity beyond the two-term max, CUDA-graph capture-size padding of
decode batches (measured negligible on the 3090: decode steps are
memory-bound, see validation ledger entry 16), a separate effective bandwidth
for paged KV reads,
paged-attention block effects, speculative decoding, prefix-cache-aware
decode reads, backend microbatch scheduling inside one pipelined batch,
and any backend-specific behavior. Fitted
calibration models override these baselines when a profile matches.

## Data/control flow

1. `Solver::score_config_with_options` (src/solver.rs) validates a
   `ParallelismConfig`, places ranks, then calls
   `Solver::estimate_compute_latency_s`.
2. Per `InferencePhase`:
   - Effective FLOP/s per phase: `Solver::peak_flops` x
     `compute_efficiency_at(prefill_tokens_per_pass)` (`batch x prompt`) for
     prefill and x `compute_efficiency_at(decode_tokens_per_pass)` (`batch`)
     for decode.
   - Prefill: `Solver::prefill_baseline_s` = max((dense FLOPs / shard +
     `prefill_attention_flops` / attention shard) / effective FLOP/s x
     `prefill_compute_scale`, weight bytes / shard / bandwidth x
     `decode_compute_scale`).
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
   score; a pure-prefill step equals the prefill score at any prompt length
   (both take the weight-read floor). The engine evaluates the efficiency
   curve at each step's total token count (see
   docs/features/serving_iteration_engine.md).
6. The phase-pipeline serving scheduler derives per-token decode cost from a
   `decode_tokens = 1` score and a whole-decode score (`decode_tail_scale`,
   src/serving/scheduling.rs), so the context-dependent KV-read term makes
   later decode iterations costlier than the first.

## Related files

- `src/calibration/efficiency_curve.rs` — `ComputeEfficiencyCurve` (`new`,
  `efficiency_at`, `points`), `EfficiencyPoint`, `EfficiencyCurveError`,
  `MAX_EFFICIENCY_CURVE_POINTS`; fixed-capacity so `SimulationCalibration`
  stays `Copy`.
- `src/calibration.rs` — `SimulationCalibration::compute_efficiency_at`
  (curve or scalar), `compute_efficiency_curve` field.
- `src/config/calibration_config.rs` — `parse_compute_efficiency_curve`,
  `calibration_with_defaults` (validates the curve and frontend keys).
- `src/workload.rs` — `ParameterCountSource`, `ParameterCountMismatch`,
  `ModelSpec::parameter_count_mismatch`, `PARAMETER_COUNT_MISMATCH_TOLERANCE`.
- `src/solver/placement.rs` — `parameter_count_mismatch_approximation`,
  `MODEL_PARAMETER_COUNT_MISMATCH`.
- `src/solver/efficiency_curve_tests.rs`, `src/config/calibration_curve_tests.rs`
  — curve, floor, and mismatch tests.
- `src/solver.rs` — `estimate_compute_latency_s`, `prefill_baseline_s`,
  `peak_flops`, `prefill_tokens_per_pass`, `decode_tokens_per_pass`,
  `decode_compute_latency_s`, `flop_latency_s`, `latency_shard_factor`,
  `attention_shard_factor`, `kv_cache_bytes`,
  `prefill_attention_flops`, `decode_attention_flops`,
  `decode_kv_read_bytes`, `average_decode_context_tokens`,
  `effective_hbm_bandwidth`.
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

- Without `compute_efficiency_curve`, every pass uses the scalar
  `compute_efficiency` (existing configs are unchanged). With a curve, the
  scalar is ignored for latency but still reported.
- A curve value always has 2-32 points with strictly increasing positive
  tokens and efficiencies in (0, 1]; invalid curves are config errors, never
  sanitized.
- Prefill is never cheaper than one weight read (`decode_compute_scale`
  applies to it, as to every memory term).
- `model_parameter_count_mismatch` fires only for
  `ParameterCountSource::ShapeWithDefaultFfnWidth` with more than 10%
  disagreement; an explicit `ffn_hidden_size` or `parameter_count_billion`
  silences it.

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
