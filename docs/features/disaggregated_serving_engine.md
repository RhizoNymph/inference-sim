# Disaggregated Serving Engine

Disaggregated and partially disaggregated prefill/decode serving on the
iteration engine's primitives (`docs/features/serving_iteration_engine.md`),
modeled on vLLM's disaggregated prefill with the NixlConnector: a prefill
instance, a decode instance, a KV connector that pulls each request's KV
cache from the prefill GPU(s) to the decode GPU(s), and a proxy that sends
each request to prefill and then to decode.

## Scope

- Candidates with continuous prefill and decode batching whose requests'
  prefill and decode routes name different GPU sets: fully disaggregated
  pools, partially disaggregated pools (some requests colocated on a shared
  GPU set, others moved), several prefill and decode workers, and split
  prefill/decode parallelism configs (prefill TP/PP different from decode).
- Prefill workers run vLLM-style chunked-prefill engine steps under their own
  token budget, sequence, and KV limits.
- KV transfer per request: prompt KV bytes sharded by the prefill and decode
  TP/PP layouts, priced per directed GPU pair from a measured `send_recv`
  curve or the routed alpha-beta path, contending on the directed route
  resources either as FIFO queues or by max-min fair sharing
  (`calibration.kv_link_sharing`, see "Link sharing").
- An optional calibrated proxy hop (`calibration.
  disaggregation_proxy_latency_us`) between the prefill instance finishing a
  prompt and the request reaching the decode instance.
- Decode workers admit a request (reserving its KV and a sequence slot)
  in queue order, then pull its KV, then decode it in their steps alongside
  other requests.
- Two first-token conventions (`serving.traffic.disaggregated_first_token`):
  `decode_instance` (default, vLLM's proxy) and `prefill_instance`.

## Non-scope

- Independent prefill or decode batching and data-parallel replicas stay on
  the phase pipeline (`phase_pipeline_scheduler`, reasons
  `independent_*_batching`, `data_parallel_replicas`).
- Worker layouts the engine cannot express fall back to the phase pipeline:
  two routed GPU sets that share some but not all GPUs
  (`partially_overlapping_workers`), one GPU set serving both phases with
  different prefill and decode configs (`split_prefill_decode_parallelism`),
  and transfers with no topology route (`kv_transfer_unplannable`).
- Push-style connectors (P2pNcclConnector, LMCache) and layer-wise overlap of
  transfer with prefill: not modeled (`kv_transfer_decode_initiated_pull`).
- Under fair sharing, per-stream throughput beyond a single stream's measured
  price (many parallel streams beating one) and per-stream overheads: each
  resource's capacity is one standalone flow. vLLM-side limits on how many
  pulls are in flight are not modeled beyond decode admission.
- A load- or prompt-dependent proxy hop: the hop is one constant per moved
  request (`disaggregated_constant_proxy_latency`; zero and
  `disaggregated_proxy_hop_not_modeled` by default). The client-facing
  frontend latency (`frontend_latency_us`, serving_iteration_engine) is
  applied once to every request's arrival, colocated or moved, and is not
  part of the hop.
- Cancelling a request mid-pull does not abort its transfer: the flow keeps
  its share until it finishes.
- Egress sharing at one NIC between transfers to different destinations is
  captured only where the topology routes share a resource.
- Routing still uses the existing policies' phase-pipeline estimates
  (`routing.rs`); the engine does not feed load back into routing.

## The vLLM flow being modeled

vLLM 0.29 with NixlConnector and the NIXL toy proxy:

1. The proxy sends a non-streaming request with `max_tokens = 1` and
   `kv_transfer_params = {"do_remote_decode": true}` to the prefill instance.
2. The prefill instance schedules it like any request (chunked prefill
   under `max_num_batched_tokens`); the step that computes the last prompt
   token samples one token, and the request finishes. Its KV blocks stay
   allocated ("delay free") until the decode side has read them.
3. The proxy receives the response (carrying the remote block ids) and sends
   a streaming request with those `kv_transfer_params` to the decode
   instance, streaming its output to the client.
4. The decode instance's scheduler allocates KV blocks for the prompt when it
   reaches the request in its waiting queue, marks it
   `WAITING_FOR_REMOTE_KVS`, and NIXL reads the blocks from the prefill GPU.
5. When the read completes, the request is scheduled with
   `num_computed_tokens = prompt - 1`: the decode instance recomputes the
   last prompt token and samples the first token the client sees, then
   decodes the rest (`max_tokens` tokens in total from decode).

So client-measured TTFT (`vllm bench serve` against the proxy) is: prefill
queue + prefill compute (including the prefill instance's own token) + proxy
hop + decode-side queueing + KV pull + the decode step that recomputes the
last prompt token. The simulator adds the client-facing frontend latency to
the arrival and the calibrated proxy hop between prefill finish and decode
queue entry. The prefill instance's token is never seen by the client.

## TTFT convention

`disaggregated_first_token = "decode_instance"` (default) models exactly
that: every client token comes from the decode worker; token 0 is an instant
at the end of the decode step that recomputes the last prompt token (a
one-token prefill chunk at context `prompt - 1`, priced in that step with
the other decodes); TTFT = that instant - arrival. All `decode_tokens`
client tokens come from decode. The prefill worker's sampled token is
internal: it appears in no decode iteration observation and the prefill step
has `first_token_sequences = 0`.

`"prefill_instance"` models a proxy that forwards the prefill token
immediately: token 0 is an instant at the end of the prefill worker's
prompt-finishing step (TTFT = prefill finish - arrival), and after the pull
the decode worker decodes token 2 directly at context `prompt + 1`, with no
recompute. A request with one output token then never reaches decode.

## Data and control flow

1. `schedule_serving_simulation` (src/serving/scheduling.rs) builds and
   routes the request states as before, then `select_scheduler_model`
   (src/serving/engine.rs):
   - independent batching, unplaced configs, data parallelism -> phase
     pipeline (unchanged);
   - every request colocated -> `IterationEngine` (unchanged), or the phase
     pipeline if prefill and decode configs/placements differ;
   - otherwise `disaggregated_workers` groups routes into workers;
     `Ok` -> `DisaggregatedEngine`, a layout error -> phase pipeline with its
     reason.
2. `run_disaggregated_engine(states, DisaggregatedRun)`
   (src/serving/engine/disaggregated.rs):
   - workers: one per distinct routed GPU set (prefill and decode routes
     both), `prefill_only` marks workers that never decode;
   - limits: `engine_limits` as for colocated serving, applied to every
     worker (each prefill and decode instance has its own
     `max_num_batched_tokens`, `max_num_seqs`, and KV);
   - jobs: `engine_request` per state. Colocated states keep their footprint.
     A moved state gets the prefill footprint (`batch x (prompt + 1)`
     tokens and blocks, one sequence per lockstep sequence) on the prefill
     worker and a `Handoff` carrying the decode worker, the decode footprint
     (`max_sequence_tokens`, as for colocated), the transfer plan, and the
     first-token convention;
   - transfer plans: `PlacedKvLayout::from_score` for the prefill and decode
     scores (placement moved to the routed node for single-node placements;
     the worker's GPU set is authoritative if routing chose other GPUs), then
     `KvTransferPlanner::plan` (cached per layout pair and byte count);
   - costs: `RoleCosts` prices prefill-only workers with an
     `IterationCostModel` of the prefill score and every other worker with
     the decode score's.
3. `run_engine_jobs` (src/serving/engine/core.rs) runs the shared
   discrete-event loop. Disaggregated additions:
   - phases `AwaitingDecode`, `Pulling { ready_s }`, `Recomputing`;
   - the prefill step that completes a moved prompt calls `hand_off`: the
     request leaves the prefill worker's running set, its prefill sequence is
     released at that instant (`release_sequences_at`), and it is inserted
     into the decode worker's pending arrivals at the step's end plus the
     handoff delay (`HandoffModel::delay`, from
     `disaggregation_proxy_latency_us`; `decode_queued_s` records it);
   - at the decode worker's next step boundary it joins the waiting queue
     (priority, then original arrival order);
   - `admit_for_pull`: when it is at the head of the queue and its decode
     footprint fits, the decode KV is allocated (`HoldingKey::Decode`), the
     transfer starts on the `KvNetwork` now, and the request runs
     `Pulling { arrived_s: None }` (it takes no token budget, does not block
     later admissions);
   - before any worker acts at time `t`, `complete_transfers(t)` advances the
     network to `t`; every transfer that finished by then records its
     `TransferWindow`, schedules the prefill KV release at its exact finish
     time, and marks the request `Pulling { arrived_s: Some(finish) }`;
   - `promote_pulled` at each step start turns arrived pulls into
     `Recomputing` (decode instance) or `Decoding { emitted: 1 }` (prefill
     instance);
   - a worker whose only running requests are pulling keeps its clock and
     wakes at its next pending arrival (a handoff landing mid-pull is
     admitted, and its pull started, at once) or its earliest pull
     completion, read from the network's projection each time; a blocked
     worker also wakes at other workers' real events (steps, arrivals, pull
     completions, waiting deadlines) and at the next transfer finish (which
     frees prefill KV), so cross-worker capacity deadlocks end in `Starved`
     instead of looping;
   - after the loop, transfers of requests cancelled mid-pull are drained so
     their windows are recorded.
4. `KvNetwork` (src/serving/engine/transfer/network.rs) runs one of:
   - `Fifo`: `LinkQueues::reserve` reserves each flow at `max(ready, free_at
     of every resource on its route)` for its service time; the window is
     known at start. Starts happen in non-decreasing time order (the loop
     always advances the earliest worker), so queues are FIFO by readiness.
   - `FairShare` (`FairShareLinks`, transfer/fair_share.rs): see "Link
     sharing".
5. `record_engine_jobs` (src/serving/engine/record.rs) writes each state:
   - prefill spans, chunks, and `prefill` worker assignments on the prefill
     GPUs;
   - `kv_start_s`/`kv_finish_s` = transfer window; `kv_worker_queue_s` =
     decode admission - decode queue entry (waiting for decode capacity);
     `kv_resource_queue_s` = transfer start - admission (FIFO link
     queueing) + `TransferWindow::contention_s` (fair share: sojourn minus
     the uncontended duration), so admission -> finish = `kv_resource_queue_s`
     + `kv_transfer_s` under both;
     `kv_transfer_s` = uncontended plan duration; `kv_transfer_bytes`,
     `kv_transfer_paths` (one per GPU-pair flow), `kv_transfer_resources`
     (directed route resource ids), `kv_transfer_bottlenecks`,
     `kv_transfer_resource_dependencies` (FIFO: earlier transfer operations
     it queued behind or followed on its resources; fair share: transfers
     sharing one of its resources when it started), `kv_transfer_fit`; `kv_transfer` worker assignments on
     the prefill and decode GPUs;
   - client tokens and `decode` worker assignments on the decode GPUs;
     `decode_resource_queue_s` = first decode step start - KV arrival;
   - one `ScheduledOperation` per pull, named
     `request N kv-transfer P->D` like the phase pipeline's, on the directed
     route resources, after the engine step operations;
   - cancellation reasons distinguish waiting for decode admission, during
     the KV transfer, and before the first decode step.
   The lifecycle events (`KvTransferStarted`/`Finished`,
   `DecodeIteration*`), route evidence, KV route resource summaries, worker
   observations, metrics, JSON, and CSV all derive from these fields, as for
   the phase pipeline.
6. `serving_approximations` adds `iteration_engine_approximations` plus
   `disaggregated_engine_approximations(traffic, prefill_score,
   calibration)`: `kv_transfer_fifo_link_queues` or
   `kv_transfer_fair_share_links`, `kv_transfer_decode_initiated_pull`,
   `disaggregated_proxy_hop_not_modeled` or
   `disaggregated_constant_proxy_latency`, and
   `iteration_engine_pipeline_stages_serialized` when prefill uses PP.

### Transfer model

For each pair of a source rank and a destination rank whose shards overlap
(`kv_shard_flows`, src/serving/engine/transfer/shards.rs):

- pipeline stage `p` of `pp` holds layers `[p/pp, (p+1)/pp)`; tensor rank
  `t` of `tp` holds KV heads `[t/tp, (t+1)/tp)`, or, with fewer KV heads than
  tensor ranks, the replicated head `floor(t x kv_heads / tp)`; expert ranks
  hold replicas;
- every destination rank receives its whole shard (replicated destination
  heads are each transferred); a piece held by several source replicas is
  read from replica `destination_rank % replicas`;
- fraction = layer overlap x head overlap; flows on the same GPU pair merge;
  flows between the same GPU are dropped.

Bytes per flow = fraction x `batch x prompt_tokens x layers x kv_heads x
head_dim x 2 x kv dtype bytes` (the same total as
`kv_transfer_bytes_for_routes`; prefix-cache hits still move, because the
KV exists on the prefill GPU).

Per flow (`KvTransferPlanner`, src/serving/engine/transfer/plan.rs):

- route: `kv_transfer_paths` for the single GPU pair (directed graph route,
  intra-node fabric for same-node pairs);
- service time: the cluster's `send_recv` curve for `[src_node, dst_node]`
  (`node_pair` is directed; `intra_node`/`inter_node` scopes also match) at
  the flow's bytes, with no calibration scalars; otherwise
  `latency x collective_latency_scale + bytes / bottleneck bandwidth x
  kv_transfer_scale / collective_bandwidth_scale`, with the route's
  per-direction bandwidth;
- resources: `kv_route_directed_resource_id` of every route resource
  (`kv_route:<kind>|<label>|rail=<r>|<from>-><to>`), interned as `LinkId`s.
  The two directions of a full-duplex link are separate queues.
- a calibration-profile `kv_transfer` fit rescales every flow so the plan's
  uncontended duration equals the fitted seconds.

### Link sharing

`calibration.kv_link_sharing` (default `fifo`, the historical behavior; the
lab measurement fair-shares, ledger 21):

- `fifo`: a flow holds every resource on its route for its uncontended
  service time; concurrent transfers serialize.
- `fair_share`: processor sharing with max-min fairness. Every directed
  resource has a capacity of one standalone flow; a flow alone progresses at
  rate 1 (one second of its uncontended price per second). Active flows get
  the progressive-filling allocation (`max_min_rates`): all rates rise
  together, a flow freezes when a resource on its route saturates (rates on
  it sum to 1) or it reaches rate 1. Rates are recomputed only when a flow
  starts or finishes; remaining work is kept at a common epoch and rebased
  only then, so a flow that is alone from start to finish ends at
  `start + price` bit for bit, and a run with no overlapping pulls is
  identical to FIFO apart from `predecessors` (fair share lists only
  transfers that were concurrent). A transfer finishes with its last flow.
  Projections (`finish_s`, `next_finish_after`) run the same completion code
  on a clone, so they equal the recorded finish unless another transfer
  starts first; the engine re-reads them at every loop iteration.
  Normalizing capacity by each flow's own price, not a nominal link
  bandwidth, is what the measurement supports: NIXL's aggregate throughput
  under burst stays at its standalone rate while each transfer slows down.
  For flows of one transfer that share resources only pairwise in a chain
  (no single common resource), max-min alone can finish earlier than the
  FIFO-in-plan-order price; contention is clamped at zero there. Every
  layout the planner emits on two-node topologies shares one common link.

Example: Qwen2.5-7B (28 layers, 4 KV heads, head_dim 128, bf16), 512-token
prompt: 57,344 bytes/token, 29.36 MB per request. Over the lab's measured
node0 -> node1 curve (~0.36 GB/s) one pull takes ~81 ms; node1 -> node0
~25 ms.

## Files

| file | role | key exports |
|---|---|---|
| `src/serving/engine.rs` | scheduler selection, entry points, approximation records | `SchedulerModel::DisaggregatedEngine`, `PhasePipelineReason::{PartiallyOverlappingWorkers, KvTransferUnplannable}`, `IterationEngineError::{KvPlan, WorkerLayout, fallback_reason}`, `disaggregated_engine_approximations` |
| `src/serving/engine/disaggregated.rs` | workers, footprints, plans, role costs, recording | `disaggregated_workers`, `DisaggregatedWorkers`, `WorkerLayoutError`, `DisaggregatedRun`, `run_disaggregated_engine` |
| `src/serving/engine/core.rs` | the loop with handoffs and pulls | `run_engine` (colocated), `run_engine_jobs` (default handoff model), `run_engine_jobs_with`, `EngineError::HandoffToSameWorker` |
| `src/serving/engine/types.rs` | job and handoff types | `EngineJob`, `JobRoute`, `Handoff`, `FirstTokenSource`, `HandoffModel`, `HandoffDelay`, `InvalidHandoffDelay`, `HandoffRecord`, `RequestTimeline::handoff`, `WorkerStepCost`, `UniformCost` |
| `src/serving/engine/capacity.rs` | two holdings per request, partial release | `HoldingKey`, `CapacityLedger::{fits_on, never_fits_on, allocate_on, release_sequences_at}` |
| `src/serving/engine/limits.rs` | shared worker/request builders | `EngineWorker::on_gpus`, `engine_request` |
| `src/serving/engine/record.rs` | outcome -> states, step and transfer operations | `record_engine_jobs`, `RecordedEngine` |
| `src/serving/engine/transfer.rs` | flows, plans, FIFO link queues | `KvFlow`, `KvTransferPlan`, `LinkQueues`, `TransferWindow` (incl. `contention_s`), `TransferError` |
| `src/serving/engine/transfer/network.rs` | event-driven transfer network under either discipline | `KvNetwork::{new, start, advance_to, take_completed, drain, finish_s, next_finish_after}`, `FifoLinks` |
| `src/serving/engine/transfer/fair_share.rs` | max-min fair progress, projections | `FairShareLinks`, `max_min_rates` |
| `src/calibration.rs` | sharing and proxy knobs | `KvLinkSharing`, `SimulationCalibration::{kv_link_sharing, disaggregation_proxy_latency_us, disaggregation_proxy_latency_s}` |
| `src/config/calibration_config.rs`, `src/config/sections.rs`, `src/config/run_config.rs`, `src/cli/scenario.rs`, `src/cli/calibration.rs` | TOML parsing, scenario overlays, JSON echo | `parse_kv_link_sharing` |
| `src/serving/engine/transfer/shards.rs` | rank-to-rank KV fractions | `KvShardLayout`, `ShardFlow`, `kv_shard_flows` |
| `src/serving/engine/transfer/plan.rs` | cluster-priced plans | `KvTransferPlanner`, `KvPlanContext`, `PlacedKvLayout`, `PlannedTransfer`, `FlowPricing`, `KvPlanError` |
| `src/serving/topology.rs` | directed queue keys | `kv_route_directed_resource_id` |
| `src/solver/network_cost.rs` | fit hook | `Solver::fitted_kv_transfer_seconds` |
| `src/serving/model/traffic.rs`, `src/config/serving_config/traffic.rs` | first-token knob | `ServingDisaggregatedFirstToken`, `ServingTraffic::disaggregated_first_token`, `parse_disaggregated_first_token` |
| `src/serving/engine/transfer/tests.rs` | shards, link queues, curve vs alpha-beta, direction | - |
| `src/serving/engine/transfer/fair_share_tests.rs` | max-min rates, exact lone price, projections, FIFO facade parity | - |
| `src/serving/engine/tests/fair_share.rs` | loop semantics of fair-share pulls and the handoff delay | - |
| `src/serving/engine/tests/fair_share_serving.rs` | end-to-end fair share and proxy on the lab cluster | - |
| `src/config/calibration_handoff_tests.rs` | TOML keys | - |
| `lab-runs/2026-10-01-fair-share-sim/predict.py` | ledger 21 re-prediction and proxy estimate | - |
| `src/serving/engine/tests/disaggregated.rs` | loop semantics with a closed-form cost | - |
| `src/serving/engine/tests/disaggregated_serving.rs` | end-to-end on the lab cluster, runtime bound | - |
| `examples/rtx3090_lab_cluster_measured_curves.toml`, `examples/rtx3090_qwen7b_disaggregated_workload.toml` | lab reference case | - |

## Configuration

```toml
[serving]
mode = "fully_disaggregated"   # or partially_disaggregated
prefill_nodes = [0]
decode_nodes = [1]

[serving.traffic]
prefill_batching = "continuous"          # both continuous -> engine
decode_batching = "continuous"
disaggregated_first_token = "decode_instance"   # or "prefill_instance"
max_prefill_batch_tokens = 2048          # per worker (prefill and decode)
max_decode_batch_tokens = 64             # max_num_seqs per worker
max_decode_sequences_per_gpu = 64
max_resident_tokens_per_gpu = 82864      # per-GPU limits apply per worker
max_kv_blocks_per_gpu = 5179
```

```toml
[calibration]
kv_link_sharing = "fair_share"            # or "fifo" (default)
disaggregation_proxy_latency_us = 44000   # lab NIXL toy proxy, ledger 21
```

Global limits (`max_decode_sequences`, `max_resident_tokens`,
`max_kv_blocks`) sum over every worker and every holding, including the
prefill-side KV of requests waiting to be pulled; for per-instance vLLM
limits use the `_per_gpu`/`_per_node` and `max_decode_batch_tokens` knobs.

## Invariants and constraints

- Deterministic: same tie-breaks as the colocated engine; handoffs enter the
  decode worker's pending queue by (time, request index).
- A moved request holds its prefill KV from prefill admission until its pull
  completes, and its decode KV from decode admission until completion or
  cancellation. Its prefill sequence slot frees when its prompt completes.
- A pull starts no earlier than decode admission, which is no earlier than
  the prefill finish; the request computes on decode no earlier than the
  pull's end. Hence `kv_start >= prefill_finish` and
  `first_decode_start >= kv_finish`.
- FIFO: at any instant every directed route resource carries at most one
  flow. Fair share: at any instant the rates of the flows crossing a
  resource sum to at most 1 and every rate is in (0, 1]; a lone flow takes
  exactly its uncontended price; work is conserved (a saturated resource
  ends its busy period when FIFO would).
- The proxy hop applies to moved requests only and is the same under both
  first-token conventions; colocated requests in a partially disaggregated
  pool are untouched.
- A request's own footprints are checked on arrival: if either can never
  fit, it is rejected (`NeverFits`) before prefilling.
- Exactly `decode_tokens` client tokens per completed request under both
  conventions.
- Colocated candidates produce the same results as before this feature
  (the colocated path is `run_engine_jobs` with only colocated jobs).
- The engine never panics; invalid handoffs are typed `EngineError`s, an
  unroutable or inconsistent plan is a `KvPlanError` that sends the candidate
  to the phase pipeline.

## Validation

Measured in ledger 18 (blind, FIFO) and re-predicted in ledger 21 (fair share
+ 44 ms proxy constant fitted on rates 1-4): TTFT p50 -6..-9% at rates 1-4,
+9% / +23% at rates 6 / 8, p90 within 2-14% at rates 1-8, TPOT within 0-8%;
burst still -46% TTFT p50 / +43% throughput (vLLM keeps far more pulls in
flight than decode admission allows here, and NIXL's aggregate degrades with
them). The original blind predictions follow.

Originally not yet measured. `examples/rtx3090_qwen7b_disaggregated_workload.toml`
is the reference case for the planned lab run (node0 prefill, node1 decode,
NixlConnector over UCX TCP on the 10GbE bond). Predictions from the
recalibrated 3090 constants and the measured network, 200 requests:

| rate (req/s) | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | ITL p50 ms | ITL p99 ms | out tok/s | KV pull p50 ms |
|---|---|---|---|---|---|---|---|
| 1 | 226 | 398 | 19.6 | 19.6 | 19.7 | 136 | 81 |
| 2 | 227 | 448 | 19.7 | 19.7 | 19.9 | 268 | 81 |
| 3 | 231 | 587 | 19.8 | 19.8 | 20.1 | 396 | 81 |
| 4 | 234 | 609 | 19.9 | 19.9 | 20.2 | 521 | 81 |
| 5 | 255 | 763 | 20.0 | 20.0 | 20.3 | 641 | 81 |
| 6 | 311 | 827 | 20.1 | 20.1 | 20.4 | 756 | 81 |
| 7 | 425 | 961 | 20.2 | 20.2 | 20.5 | 867 | 81 |
| 8 | 589 | 1144 | 20.3 | 20.4 | 20.6 | 971 | 81 |
| burst | 10256 | 19788 | 20.5 | 20.5 | 20.6 | 1142 | 81 |

The ~81 ms pull is priced from the node0 -> node1 curve's derived region
(the legacy NCCL sweep measured that direction only at 64-256 MiB). A
200-request run takes well under a second in a release build.
