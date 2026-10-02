# Serving Iteration Engine

An iteration-level, discrete-event serving loop modeled on vLLM V1's
scheduler. It simulates colocated continuous-batching serving one engine step
at a time and prices each step from what that step actually carries.

## Scope

- Colocated serving candidates with continuous prefill and continuous decode
  batching, where prefill and decode share one parallelism config and one
  placement (no data-parallel replicas). Every request's prefill and decode
  routes must name the same GPUs. `select_scheduler_model` checks this per
  candidate.
- One engine worker per routed GPU set. Several workers (e.g. one replica per
  node of a multi-node colocated pool) run in one global time order and share
  global and traffic-class capacity limits.
- vLLM V1 step semantics:
  - each running decoding request gets one token per lockstep sequence;
  - the rest of the token budget goes to prefill chunks, first for running
    requests in admission order, then for waiting requests in queue order
    (priority, then arrival);
  - a waiting request is admitted only when its KV footprint fits every
    limit. Otherwise it and everything behind it waits (head-of-line, FCFS);
  - the step that computes a prompt's last token samples the first output
    token.
- Per-step latency from the solver's roofline for the step's composition
  (`IterationCostModel`, src/solver/step_cost.rs): one forward pass,
  `max(compute, memory)`, plus per-step tensor/expert/pipeline communication,
  plus `scheduler_overhead_us`.
- Queueing instead of rejection. A request is rejected only when its own
  footprint exceeds a limit (it can never fit), when it waits longer than
  `max_queue_delay`, or when capacity held by other workers can never be
  freed for it. Cancellations apply while waiting or running.

## Non-scope

- Disaggregated and partially disaggregated pools run on the same loop with
  handoffs and KV pulls; see `docs/features/disaggregated_serving_engine.md`.
  Independent batching, data-parallel replicas, colocated candidates with
  split prefill/decode parallelism, and worker layouts the engine cannot
  express stay on the phase-pipeline scheduler
  (`src/serving/scheduling/pipeline.rs`) and carry a
  `phase_pipeline_scheduler` approximation naming the reason.
- Preemption. KV is reserved for a request's whole `max_sequence_tokens` at
  admission and never preempted (`iteration_engine_kv_reserved_at_admission`).
  vLLM allocates blocks incrementally and recomputes preempted requests.
- API-server latency (tokenization, detokenization, HTTP streaming)
  (`iteration_engine_no_frontend_overhead`).
- Pipeline micro-batching. A step runs its stages back to back
  (`iteration_engine_pipeline_stages_serialized`), like the static solver.
- Calibration-profile phase fits (fitted prefill/decode latency models) do
  not price individual steps; the profile's roofline scalars do. Serving
  metric fits still apply to the reported metrics afterwards.
- Phase-pipeline-only knobs: worker slots (`max_*_worker_slots_per_gpu`),
  `max_kv_queue_delay`, `max_decode_queue_delay`,
  `max_decode_iteration_queue_delay`, and the `decode_capacity_policy`
  distinction. The engine has no separate KV, decode, or iteration queues,
  and both policies queue.

## Data and control flow

1. `ServingSolver::score_pair` (src/serving.rs) calls
   `schedule_serving_simulation` (src/serving/scheduling.rs). That function
   generates arrivals and request shapes, routes every request, and builds
   one `DecodeRequestState` per request, the same way for both schedulers.
2. `select_scheduler_model` (src/serving/engine.rs) returns
   `SchedulerModel::IterationEngine`, `SchedulerModel::DisaggregatedEngine`
   (docs/features/disaggregated_serving_engine.md), or
   `SchedulerModel::PhasePipeline(reason)`.
3. `run_iteration_engine`:
   - `IterationCostModel::new(cluster, model, decode_score, calibration)`
     precomputes the roofline constants for the placed config: effective
     peak FLOPs and HBM bandwidth, dense FLOPs per token, attention FLOPs per
     (query, key) pair, KV bytes per context token, sharded weight bytes, and
     the per-step collectives (an embedding all-reduce per first-stage tensor
     group and an LM-head logits all-gather per last-stage tensor group sized
     by the step's sampling sequences, which is every decode plus each prefill
     chunk that completes its prompt; two all-reduces per layer per tensor group,
     one all-to-all per layer per expert group, one send/recv per stage
     edge);
   - `engine_workers` groups states by routed GPU set; `engine_limits` maps
     `[serving.traffic]` onto `EngineLimits` (table below);
     `engine_requests` builds one `EngineRequest` per state (lockstep
     sequences = `batch_size`, cached prompt tokens, prompt tokens to
     compute (at least 1), output tokens, KV footprint, class, cancellation,
     queue limit).
4. `run_engine` (src/serving/engine/core.rs; a wrapper over
   `run_engine_jobs` with only colocated jobs) loops until every worker is
   idle. Each iteration it:
   - picks the worker with the earliest clock;
   - applies capacity releases up to that time (`CapacityLedger`);
   - enqueues arrivals, rejecting never-fitting requests;
   - expires cancelled and queue-timed-out requests;
   - plans a step (`plan_step`: decodes, running chunks, admissions under
     `StepBudget`);
   - prices it with `StepCost::step_latency(&StepWork)`;
   - advances the worker clock to the step's end, emitting tokens and
     scheduling releases at that instant.
   A worker with nothing runnable sleeps until its next arrival, the next
   release, a waiting deadline, or another worker's next real event (a step,
   arrival, pull completion, or waiting deadline; another blocked worker's
   clock does not count, so mutual blocking ends in `Starved`).
5. `record_engine_outcome` (src/serving/engine/record.rs) writes each
   `RequestTimeline` back into its `DecodeRequestState`:
   - prefill start and finish, chunk count, and `prefill_token_spans`;
   - KV handoff at the first-token instant (colocated, zero bytes);
   - per-token start and finish (token 0 is an instant at the end of the
     prompt-finishing step);
   - worker assignments and dependencies;
   - terminal fates through the shared `cancel_request`, `reject_admission`,
     and `reject_prefill_capacity_admission` helpers.
   It also emits one `ScheduledOperation` per step (`engine step N ...` on
   the worker's `gpu compute` / `gpu HBM` resources) and one
   `ServingDecodeIterationObservation` per token-emitting step.
6. `apply_terminal_statuses`, then `summarize_serving_simulation`
   (src/serving/scheduling/summary.rs, shared with the phase pipeline),
   produce metrics (TTFT, TPOT, ITL, E2EL, throughput, percentiles, SLOs),
   capacity profiles, worker and service observations, utilization, and
   measurement windows. JSON and CSV outputs are unchanged.
7. `serving_approximations` adds `iteration_engine_approximations` or
   `phase_pipeline_approximation` according to `ServingSimulation::scheduler_model`.

### Knob mapping

| `[serving.traffic]` knob | engine meaning |
|---|---|
| `max_prefill_batch_tokens` | per-step token budget (vLLM `max_num_batched_tokens`), decode tokens included |
| `max_prefill_chunk_tokens` | largest prefill chunk one request takes per step |
| `max_decode_batch_tokens`, `max_decode_sequences_per_node`, `max_decode_sequences_per_gpu` | running sequences per worker (vLLM `max_num_seqs`) |
| `max_decode_sequences`, `max_resident_tokens`, `max_kv_blocks` | admission limits summed over all workers |
| `max_resident_tokens_per_node`/`_per_gpu`, `max_kv_blocks_per_node`/`_per_gpu` | per-worker KV limits, times the worker's node/GPU count |
| `max_prefill_tokens`, `max_prefill_tokens_per_node`/`_per_gpu` | prefill tokens per worker step |
| traffic class `max_decode_sequences`, `max_resident_tokens`, `max_kv_blocks`, `max_prefill_tokens` | the same limits per class |
| `max_queue_delay` (traffic, class, trace) | waiting longer rejects the request (`prefill_queue_delay_exceeded`) |
| `request_timeout`, deadlines | applied to completed requests by `apply_terminal_statuses` |
| `kv_block_tokens` | block size for each request's `kv_cache_blocks` footprint |

No new knobs were added.

### Step cost

For a step with prefill chunks (c tokens on top of p cached tokens, for s
lockstep sequences) and decoding sequences at context n:

- compute = `[2 P (sum s c) + 2 L H (sum s ((p + c)^2 - p^2)) / tp] / F *
  prefill_compute_scale + [2 P (sum s) + 4 L H (sum s n) / tp] / F *
  decode_compute_scale`
- memory = `[W / (tp ep) + kv_bytes_per_token (sum s n + sum s p) / tp] / B *
  decode_compute_scale`
- total = `max(compute, memory) + communication + scheduler_overhead_us`
  (`max(forward, communication)` with `allow_compute_comm_overlap`)

Here P is the parameter count / (tp ep), L layers, H hidden size,
F = peak FLOPs x `compute_efficiency`, B = HBM bandwidth x
`decode_memory_bandwidth_scale`, and W is weight bytes. Communication prices
each per-step collective with a message of `step tokens x hidden x dtype`
through `Solver::estimate_collective_with_calibration`, cached per token
count.

## Files

| file | role | key exports |
|---|---|---|
| `src/solver/step_cost.rs` | per-step roofline | `IterationCostModel` (`new`, `step_latency`), `StepWork` (`add_prefill_chunk`, `add_decode`), `StepLatency`, `StepCostError` |
| `src/solver/step_cost/tests.rs` | static/step agreement, flat decode, TP/PP | - |
| `src/serving/engine.rs` | module root, scheduler selection, entry point, approximation records | `SchedulerModel`, `PhasePipelineReason`, `select_scheduler_model`, `run_iteration_engine`, `EngineTimeline`, `IterationEngineError`, `iteration_engine_approximations`, `phase_pipeline_approximation` |
| `src/serving/engine/types.rs` | engine value types | `EngineRequest`, `KvFootprint`, `CapacityLimits`, `StepLimits`, `WorkerLimits`, `ClassLimits`, `EngineLimits`, `CapacityExcess`, `RequestFate`, `ChunkRecord`, `TokenRecord`, `RequestTimeline`, `EngineStep`, `EngineOutcome`, `StepCost` |
| `src/serving/engine/core.rs` | the discrete-event loop (colocated and disaggregated jobs) | `run_engine`, `run_engine_jobs`, `EngineError` |
| `src/serving/engine/capacity.rs` | shared capacity ledger with timestamped releases | `CapacityLedger` |
| `src/serving/engine/limits.rs` | workers, limits, and requests from states and traffic | `EngineWorker`, `engine_workers`, `engine_limits`, `engine_requests` |
| `src/serving/engine/record.rs` | engine outcome -> request states, operations, decode iterations | `record_engine_outcome`, `record_engine_jobs` |
| `src/serving/engine/tests.rs`, `tests/serving.rs` | loop semantics with a closed-form cost; end-to-end 3090 checks and the runtime bound | - |
| `src/serving/scheduling.rs` | request-state construction and scheduler dispatch | `schedule_serving_simulation`, `ServingSimulation` (`scheduler_model`) |
| `src/serving/scheduling/pipeline.rs` | phase-pipeline scheduler (unchanged behavior) | `schedule_phase_pipeline`, `ScheduledTimeline` |
| `src/serving/scheduling/summary.rs` | shared metrics/observation summary | `summarize_serving_simulation`, `SummaryContext` |
| `src/serving/approximations.rs` | attaches the scheduler's approximation records | `serving_approximations` |

## Invariants and constraints

- Deterministic: worker choice breaks ties by worker index, queue order by
  (priority desc, arrival, request index), and the loop has no randomness.
- A step never carries more than its token budget, except for the first
  item of an otherwise empty step, which always makes progress (at least one
  token). This prevents livelock when a single request exceeds the budget.
- Running sequences never exceed any sequence limit, and admitted KV never
  exceeds any token or block limit, at any instant. Post-hoc capacity peaks
  therefore stay within limits, so candidate-level capacity rejections do not
  fire for engine candidates.
- A release scheduled at time t is invisible to a worker planning at a time
  before t (`CapacityLedger::apply_releases_until`).
- Exactly `decode_tokens` tokens per completed request: one from the
  prompt-finishing step, then one per decode step. Token 0 has
  start == finish == prefill finish, so `TTFT = prefill_finish - arrival` and
  `TPOT = (last - first) / (n - 1)` match `vllm bench serve`.
- Every request has at least one prompt token to compute, so a full
  prefix-cache hit still runs one step to sample its first token.
- Pure-decode steps equal the static solver's one-token decode latency at
  context `prompt + 1`. Pure-prefill steps equal the static prefill latency
  whenever compute exceeds the weight read. Small prefill-only steps pay at
  least one weight read, which the static prefill does not.
- The cost model and the engine never panic. A config without a placement is
  `StepCostError::EmptyPlacement` (the scheduler then falls back to the
  phase pipeline); a request routed to an unknown worker is
  `EngineError::UnknownWorker`.
- The workload's `[request].batch_size` (the reference batch) does not affect
  engine results. It still sizes the static scores used for feasibility and
  memory headroom.

## Validation

`lab-runs/2026-09-30-serving-baseline/report-iteration-engine.md` and
`docs/validation_ledger.md` entry 6, from the static-batch-fitted 3090
profile alone. TPOT p50 and ITL p50 are within 15% at 1-4 req/s; TPOT rising
while ITL stays flat is reproduced; saturation throughput is +4.8%; TTFT
grows into seconds past saturation. Low-load TTFT is under-predicted by
33-39%, from frontend latency the engine does not model and the static
calibration's 512-token prefill residual. A 200-request, 128-token
simulation takes ~0.1 s in a release build.
