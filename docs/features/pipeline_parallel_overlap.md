# Pipeline-Parallel Overlap Inside One Static Batch

## Scope

How the static solver prices a multi-request batch on two or more pipeline
stages when the serving backend keeps several engine steps in flight, as
vLLM V1 does with pipeline parallelism:

- `calibration.pipeline_microbatching`: `"whole_batch"` (default; the batch
  is one forward pass that walks the stages in order) or
  `"vllm_batch_queue"` (vLLM 0.29's batch queue; see below). Accepted in a
  workload `[calibration]`, a calibration profile `[calibration]` and a run
  scenario's `[scenarios.calibration]`; reported in the JSON `calibration`
  block.
- Under `"vllm_batch_queue"`, a static `prefill` or `end_to_end` request with
  `batch_size >= 2` on `pipeline_ranks >= 2` runs as two microbatches, the
  first request alone and then the other `batch_size - 1`, whose prompt
  passes overlap across stages. `end_to_end` adds one whole-batch decode pass
  after both. Every such score carries the approximation
  `pipeline_microbatch_overlap`.

## Non-scope

- Decode-only requests and the decode part of `end_to_end` are not split
  (two alternating microbatches each read the stage's weights per token, so
  a decode cycle costs about the serial round trip; a per-token prototype
  moved decode steps by under 1%).
- The arrival-timed size of the second step: vLLM's real split depends on
  how many offline requests reach the engine core before it schedules the
  second step (measured residual suggests 1-3 requests besides the first for
  B = 8). `1 + (B - 1)` is the parameter-free lower bound on overlap.
- More than two microbatches, steps beyond the first two, and PP=1 batches
  (at PP=1 vLLM's first step also carries one request, which explains
  per-sequence prefill times on node1, but the simulator keeps PP=1
  unchanged).
- The serving iteration engine: its steps still run their stages back to
  back (`iteration_engine_pipeline_stages_serialized`). Serving with PP is
  unmeasured; modeling the batch queue there means overlapping consecutive
  engine steps, a change to the engine core.
- Stages on GPUs of the same node: the trace's compute resources are per
  node (`gpu compute node N`), so such stages cannot overlap.
- Phases priced by a calibration fit: a fit is a measured whole-batch
  latency and already contains whatever overlap the backend achieved, so
  fitted phases are never split.

## What vLLM 0.29 does

Read from the v0.29.0 tag (no local install):

1. `vllm/config/vllm.py` `VllmConfig.max_concurrent_batches` returns
   `pipeline_parallel_size` when async scheduling is off; async scheduling is
   off for executors without `supports_async_scheduling()` (the abstract
   default is false and `RayDistributedExecutor` does not override it).
2. `vllm/v1/engine/core.py`: with `batch_queue_size > 1` the busy loop runs
   `step_with_batch_queue`, which schedules and submits a new step without
   blocking while the queue has room and only then blocks on the oldest one.
3. `vllm/v1/core/sched/scheduler.py` `schedule()`: a request whose prompt is
   fully scheduled but in flight gets `num_new_tokens == 0` and is skipped,
   so concurrent steps hold disjoint requests.
4. `vllm/entrypoints/offline_utils.py` adds offline requests one at a time;
   the multiprocess engine core (`SyncMPClient`) drains only what has arrived
   and steps at once, so the first step usually carries the first request
   and the next step the requests that arrived meanwhile.
5. `vllm/v1/worker/gpu_worker.py` `Worker.execute_model` sends activations
   with a non-blocking `isend_tensor_dict` and waits for the previous step's
   sends before the next forward on that stage.

## Data/control flow

1. `Solver::score_config_with_options` (src/solver.rs) prices the whole
   request with `estimate_compute_latency_s` as before.
2. `Solver::microbatch_compute` (src/solver/microbatch.rs) returns `None`
   unless no calibration fit applied, the phase is not `Decode`, and
   `MicrobatchPlan::for_static_batch(policy, batch, pipeline_ranks)` is
   `FirstThenRest { first: 1, rest: B - 1 }`. Otherwise it prices each
   microbatch's prompt pass (the request with `batch_size = k`, phase
   `Prefill`) and, for `EndToEnd`, the whole batch's `Decode` pass, each
   through `estimate_compute_latency_s`.
3. `Solver::build_microbatched_trace` uses `operations::TraceBuilder`: one
   `push_pass` per microbatch (operation names prefixed `microbatch N `),
   where microbatch 1's first layer on stage s depends on microbatch 0's
   `pipeline sendrecv edge s` (`PassLinks::stage_entry`, from
   `PassTrace::stage_sends`), then a `decode ` pass whose first layer
   depends on both prompt passes' tails (`PassLinks::entry`).
   Without a split, `build_operation_trace` pushes the single pass with no
   links and no prefix, producing exactly the previous trace.
4. `schedule_operations` (greedy list scheduler) serializes each node's
   compute resource, so microbatch 1 runs on stage 0 while microbatch 0 is
   on stage 1; the makespan is the latency.
5. `microbatch_overlap_approximation` adds `pipeline_microbatch_overlap`
   naming the split.

## Related files

- `src/calibration.rs` — `PipelineMicrobatching` (`WholeBatch`,
  `VllmBatchQueue`, `as_str`), `SimulationCalibration::pipeline_microbatching`.
- `src/solver/microbatch.rs` — `MicrobatchPlan` (`WholeBatch`,
  `FirstThenRest`), `for_static_batch`, `MicrobatchCompute`,
  `Solver::microbatch_compute`, `Solver::build_microbatched_trace`,
  `microbatch_overlap_approximation`, `PIPELINE_MICROBATCH_OVERLAP`.
- `src/solver/operations.rs` — `TraceBuilder` (`new`, `push_pass`,
  `finish`), `PassLinks`, `PassTrace`; `build_operation_trace` is one
  unlinked pass.
- `src/config/sections.rs`, `src/config/calibration_config.rs`,
  `src/config/run_config.rs`, `src/config.rs` — parsing into
  `SimulationCalibration` and `RunScenarioCalibrationConfig`.
- `src/cli/scenario.rs` (scenario override), `src/cli/calibration.rs`
  (JSON `pipeline_microbatching`).
- `src/solver/microbatch_tests.rs`, `src/config/pipeline_microbatching_tests.rs`
  — tests.
- `lab-runs/2026-10-01-pipeline-overlap-sim/` — validation (`revalidate.py`,
  `summary.md`, `revalidation.json`, `profile-curve-net-overlap.toml`);
  docs/validation_ledger.md entry 23.

## Invariants and constraints

- The default policy is `whole_batch`: every existing configuration prices
  exactly as before.
- A split always covers the batch: `first + rest = batch_size`, both
  nonzero (`NonZeroU32`), so an empty microbatch is unrepresentable.
- Single requests, PP=1, decode-only phases and fitted phases never split,
  and their traces are identical to the whole-batch trace.
- Microbatch 1 never starts a stage's forward before microbatch 0's send
  out of that stage has finished, and one node never runs two microbatches
  at once.
- `end_to_end` latency equals the split prefill makespan plus the
  whole-batch decode latency.

## Validation

Ledger entry 23, no constant refitted: 7B PP=2 prefill 10.4% -> 8.3%
(8x512 +23.9% -> +18.9%, 8x2048 +16.1% -> +9.7%), 14B PP=2 prefill 8.7% ->
6.5%; end-to-end 1.3% -> 1.7% (7B, prefill no longer cancels batch-8 decode
under-prediction) and 5.1% -> 4.8% (14B); PP=1 unchanged.
