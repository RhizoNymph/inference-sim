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
  - KV allocation (`KvAllocationMode`, chosen by
    `[serving.traffic] decode_capacity_policy`):
    `ReservedAtAdmission` (default; `candidate_reject` / `request_reject`)
    reserves each request's whole footprint at admission and never preempts;
    `PagedRecompute` (`decode_capacity_policy = "preempt"`) is vLLM V1's
    paged allocation with recompute preemption (see "Preemption" below);
  - the step that computes a prompt's last token samples the first output
    token.
- Per-step latency from the solver's roofline for the step's composition
  (`IterationCostModel`, src/solver/step_cost.rs): one forward pass,
  `max(compute, memory)`, plus per-step tensor/expert/pipeline communication,
  plus `scheduler_overhead_us`. Compute efficiency is evaluated at the
  step's total token count when the calibration has a
  `compute_efficiency_curve`.
- Calibrated per-request frontend (API-server) latency:
  `calibration.frontend_latency_us + frontend_latency_per_prompt_token_us x
  prompt_tokens` delays each request's arrival at the engine
  (`apply_frontend_latency`), so client-observed TTFT and E2EL include it
  while TPOT and ITL do not.
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
- KV swapping (vLLM's swap preemption mode), and prefix-cache hits on a
  preempted request's own freed blocks: a preempted request keeps its
  `cached_prompt_tokens` hit and recomputes everything else
  (`iteration_engine_recompute_preemption`). Under the default policies KV
  is reserved at admission and never preempted
  (`iteration_engine_kv_reserved_at_admission`).
- Load-dependent API-server latency. The frontend latency is a constant
  per-request delay; it never contends with engine steps or grows with load
  (`iteration_engine_constant_frontend_latency`). With both frontend keys
  unset (the default) no frontend latency is modeled
  (`iteration_engine_no_frontend_overhead`). The phase-pipeline scheduler
  ignores the frontend keys.
- vLLM's priority scheduling policy beyond the victim rule below; preempted
  requests are prepended to the waiting queue (FCFS behavior).
- Exact post-hoc capacity profiles under preemption: the summary's peaks and
  utilizations still count each request's full footprint over its residency
  window, so they can exceed limits the engine enforced; capacity
  rejections are skipped for preempting engine candidates.
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
   Under `PagedRecompute`, planning grows each scheduled item's KV holding
   first and preempts when it cannot (next section).
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

### Preemption (`decode_capacity_policy = "preempt"`)

`engine_kv_allocation` (src/serving/engine/limits.rs) maps the policy to
`KvAllocationMode::PagedRecompute { block_tokens }` (`kv_block_tokens`,
default 16). Then (src/serving/engine/core.rs, core/paged.rs):

1. A request's holding is `sequences x ceil(kv_tokens / block_tokens)`
   blocks (tokens counted block-rounded). Admission needs blocks only for
   the cached prefix plus its first chunk; a disaggregated decode admission
   needs blocks for the prompt it is about to pull.
2. While planning a step, each running request (in admission order) computes
   its item (decode, prefill chunk, or the decode-instance last-token
   recompute) and `grant_kv` grows its holding to cover the KV after the
   item (`Item::kv_tokens_after`). If the ledger refuses, the victim is the
   running request not yet scheduled in this step with the lowest priority,
   then latest in the running list (vLLM FCFS `running.pop()`); requests
   still pulling KV are never victims. The victim frees all its blocks
   (`CapacityLedger::release_now_all`), leaves the running set, is prepended
   to the waiting queue as `Phase::Preempted`, and gets a
   `PreemptionRecord` (worker, step, time, emitted tokens, discarded KV
   tokens). If the victim is the requester itself the step schedules no
   further running requests.
3. A step that preempted admits nothing (vLLM); a step left empty by its
   preemptions admits immediately.
4. Readmission (head of line, at the front of the queue) restarts a
   `PrefillPass`: `Prompt` if the request was preempted before its first
   token, else `Resume { emitted }`, which recomputes `prefill_tokens +
   emitted` tokens through the normal chunked token budget
   (`ChunkPass::Resume` chunks); the pass's last chunk samples token
   `emitted + 1` (an instant first token when `emitted == 0`).
5. Disaggregated: a request preempted on its decode worker recomputes
   prompt plus emitted tokens locally on that worker (vLLM's NixlConnector
   consumes the remote-prefill parameters on the first load, so a
   rescheduled request is recomputed by the decode instance); there is no
   second pull. Prefill workers preempt prompts the same way; KV awaiting a
   pull is not running and never preempted.
6. Never-fits uses the block-rounded peak: the prompt on a prefill worker,
   prompt + output - 1 tokens where the request decodes.

Outputs: `EngineStep::kv_blocks` and `preempted_requests`,
`RequestTimeline::preemptions`, `EngineOutcome::preemption_stats` ->
`EngineTimeline::preemption` -> `ScheduledTimeline::preemption` ->
`ServingMetrics::{preemptions, preempted_requests,
preemption_recompute_tokens}` (JSON `metrics`). Record keeps prefill fields
(start, finish, chunks, spans) to the prompt pass.

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
| `kv_block_tokens` | block size for each request's `kv_cache_blocks` footprint and for paged allocation |
| `decode_capacity_policy` | `preempt`: paged allocation with recompute preemption; otherwise whole footprints reserved at admission |

Calibration knobs the engine reads: `compute_efficiency` /
`compute_efficiency_curve`, `decode_memory_bandwidth_scale`,
`prefill_compute_scale`, `decode_compute_scale`, `scheduler_overhead_us`,
`allow_compute_comm_overlap`, and `frontend_latency_us` /
`frontend_latency_per_prompt_token_us`.

### Frontend latency: why calibration, how it was measured

The frontend latency is a property of the serving stack and host (vLLM's
API server, tokenizer, HTTP streaming on that CPU), not of the workload, so
it lives in `[calibration]` next to `scheduler_overhead_us` and travels with
a calibration profile. Lab value (3090 node0, vLLM 0.29.0):
`frontend_latency_us = 5081`, `frontend_latency_per_prompt_token_us =
14.44`, from `lab-runs/2026-09-30-frontend/` (isolated one-output-token
requests at 0.5 req/s: median TTFT 27.05 ms at 16 prompt tokens, 126.88 ms at
512) minus the engine-only batch-1 static prefill of the same prompts
(`lab-runs/2026-10-01-qwen7b-prefill-token-sweep/`: 21.73 ms, 114.37 ms),
i.e. 5.3 ms and 12.5 ms of overhead, fitted as a line in prompt length by
`lab.py fit-curve --frontend-dir`. The engine applies it as an ingress delay:
shifting every arrival by a constant leaves inter-arrival gaps and queueing
unchanged, so every request's TTFT and E2EL grow by exactly the delay
(tested in `src/serving/engine/tests/frontend.rs`).

### Step cost

For a step with prefill chunks (c tokens on top of p cached tokens, for s
lockstep sequences) and decoding sequences at context n:

- compute = `[2 P (sum s c) + 2 L H (sum s ((p + c)^2 - p^2)) / tp] / F(T) *
  prefill_compute_scale + [2 P (sum s) + 4 L H (sum s n) / tp] / F *
  decode_compute_scale`, with `F(T)` the effective FLOP/s at the step's
  total tokens `T = sum s c + sum s`
- memory = `[W / (tp ep) + kv_bytes_per_token (sum s n + sum s p) / tp] / B *
  decode_compute_scale`
- total = `max(compute, memory) + communication + scheduler_overhead_us`
  (`max(forward, communication)` with `allow_compute_comm_overlap`)

Here P is the parameter count / (tp ep), L layers, H hidden size,
F(T) = peak FLOPs x `compute_efficiency_at(T)` (the curve, or the scalar
`compute_efficiency`), B = HBM bandwidth x
`decode_memory_bandwidth_scale`, and W is weight bytes. Communication prices
each per-step collective with a message of `step tokens x hidden x dtype`
through `Solver::estimate_collective_with_calibration`, cached per token
count.

## Files

| file | role | key exports |
|---|---|---|
| `src/solver/step_cost.rs` | per-step roofline | `IterationCostModel` (`new`, `step_latency`), `StepWork` (`add_prefill_chunk`, `add_decode`), `StepLatency`, `StepCostError` |
| `src/solver/step_cost/tests.rs` | static/step agreement, flat decode, TP/PP | - |
| `src/serving/engine.rs` | module root, scheduler selection, entry point, frontend-latency hook, approximation records | `SchedulerModel`, `PhasePipelineReason`, `select_scheduler_model`, `run_iteration_engine`, `apply_frontend_latency`, `EngineTimeline`, `IterationEngineError`, `iteration_engine_approximations` (takes the calibration), `phase_pipeline_approximation` |
| `src/serving/engine/tests/frontend.rs` | frontend latency shifts TTFT/E2EL exactly, leaves TPOT/ITL; approximation code switch | - |
| `src/serving/engine/types.rs` | engine value types | `KvAllocationMode`, `ChunkPass`, `PreemptionRecord`, `PreemptionStats`, `EngineRequest`, `KvFootprint`, `CapacityLimits`, `StepLimits`, `WorkerLimits`, `ClassLimits`, `EngineLimits`, `CapacityExcess`, `RequestFate`, `ChunkRecord`, `TokenRecord`, `RequestTimeline`, `EngineStep`, `EngineOutcome`, `StepCost` |
| `src/serving/engine/core.rs` | the discrete-event loop (colocated and disaggregated jobs) | `run_engine_jobs_with` (production), `run_engine` / `run_engine_jobs` (tests, reserved), `EngineError` |
| `src/serving/engine/core/paged.rs` | KV holdings per allocation mode, victim choice, preemption | `KvGrant`, `Engine::grant_kv`, `never_fits_paged` |
| `src/serving/engine/core/budget.rs` | per-step token budget and chunk work | `StepBudget`, `add_chunk` |
| `src/serving/engine/capacity.rs` | shared capacity ledger with timestamped releases, holding growth, immediate release | `CapacityLedger` (`try_grow`, `held`, `release_now_all`, `worker_blocks`) |
| `src/serving/engine/tests/preemption.rs`, `tests/preemption_serving.rs` | preemption semantics with the closed-form cost; end-to-end decode-sweep cliff | - |
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
- Running sequences never exceed any sequence limit, and allocated KV never
  exceeds any token or block limit, at any instant (also under preemption:
  growth happens only through `try_grow`). With reserved allocation post-hoc
  capacity peaks therefore stay within limits, so candidate-level capacity
  rejections do not fire for engine candidates; with preemption they are
  skipped.
- Reserved allocation never preempts; paged allocation without KV pressure
  produces exactly the reserved run (tested).
- A preempted request still emits exactly `output_tokens` tokens; its resume
  pass recomputes `prefill_tokens + emitted` tokens and samples the next one.
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
  at every prompt length (both pay at least one weight read), with or
  without an efficiency curve.
- A request's engine arrival is its client arrival plus the frontend
  latency; request states keep the client arrival.
- The cost model and the engine never panic. A config without a placement is
  `StepCostError::EmptyPlacement` (the scheduler then falls back to the
  phase pipeline); a request routed to an unknown worker is
  `EngineError::UnknownWorker`.
- The workload's `[request].batch_size` (the reference batch) does not affect
  engine results. It still sizes the static scores used for feasibility and
  memory headroom.

## Validation

Preemption (validation ledger entry 19,
`lab-runs/2026-10-01-preemption-sim/`): on the decode batch sweep with
vLLM's logged 24,896-token cache, decode ms/step at batch 40-64 is -3.8% /
-7.5% / -7.7% / -6.2% (mean 6.3%; reserved allocation 27.2%, no KV limit
50.9%); batches 1-32 are unchanged.

Current (validation ledger entry 15): with the token-dependent efficiency
curve and the measured frontend latency, low-load TTFT p50 is -15% / -17% /
-19% at 1 / 2 / 4 req/s (was -33% / -35% / -39%); mean |error| over 7 rates
TTFT p50 17.7%, TPOT p50 6.5%, ITL p50 2.9%, E2EL p50 7.1%, throughput 8.4%.

History:

`lab-runs/2026-09-30-serving-baseline/report-iteration-engine.md` and
`docs/validation_ledger.md` entry 6, from the static-batch-fitted 3090
profile alone. TPOT p50 and ITL p50 are within 15% at 1-4 req/s; TPOT rising
while ITL stays flat is reproduced; saturation throughput is +4.8%; TTFT
grows into seconds past saturation. Low-load TTFT is under-predicted by
33-39%, from frontend latency the engine does not model and the static
calibration's 512-token prefill residual. A 200-request, 128-token
simulation takes ~0.1 s in a release build.
