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
  curve or the routed alpha-beta path, contending on FIFO queues per directed
  route resource.
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
- Fair-share bandwidth between concurrent transfers: links are FIFO
  (`kv_transfer_fifo_link_queues`).
- Proxy and HTTP latency (the prefill response's trip back to the proxy and
  the decode request's dispatch): zero (`disaggregated_proxy_hop_not_modeled`).
  Frontend latency is not modeled for colocated serving either.
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
last prompt token. The prefill instance's token is never seen by the client.

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
     into the decode worker's pending arrivals at the step's end;
   - at the decode worker's next step boundary it joins the waiting queue
     (priority, then original arrival order);
   - `admit_for_pull`: when it is at the head of the queue and its decode
     footprint fits, the decode KV is allocated (`HoldingKey::Decode`), the
     transfer is reserved on the link queues starting now, the prefill KV
     release is scheduled at the transfer's end, and the request runs
     `Pulling` (it takes no token budget, does not block later admissions);
   - `promote_pulled` at each step start turns completed pulls into
     `Recomputing` (decode instance) or `Decoding { emitted: 1 }` (prefill
     instance);
   - a worker whose only running requests are pulling sleeps until the
     earliest pull completes; a blocked worker also wakes at other workers'
     real events (steps, arrivals, pull completions, waiting deadlines), so
     cross-worker capacity deadlocks end in `Starved` instead of looping.
4. `LinkQueues::reserve` (src/serving/engine/transfer.rs) reserves each flow
   at `max(ready, free_at of every resource on its route)` for its service
   time. Reservations happen in non-decreasing time order (the loop always
   advances the earliest worker), so queues are FIFO by readiness.
5. `record_engine_jobs` (src/serving/engine/record.rs) writes each state:
   - prefill spans, chunks, and `prefill` worker assignments on the prefill
     GPUs;
   - `kv_start_s`/`kv_finish_s` = transfer window; `kv_worker_queue_s` =
     decode admission - decode queue entry (waiting for decode capacity);
     `kv_resource_queue_s` = transfer start - admission (link queueing);
     `kv_transfer_s` = uncontended plan duration; `kv_transfer_bytes`,
     `kv_transfer_paths` (one per GPU-pair flow), `kv_transfer_resources`
     (directed route resource ids), `kv_transfer_bottlenecks`,
     `kv_transfer_resource_dependencies` (earlier transfer operations it
     queued behind), `kv_transfer_fit`; `kv_transfer` worker assignments on
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
   `disaggregated_engine_approximations`
   (`kv_transfer_fifo_link_queues`, `kv_transfer_decode_initiated_pull`,
   `disaggregated_proxy_hop_not_modeled`, and
   `iteration_engine_pipeline_stages_serialized` when prefill uses PP).

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

Example: Qwen2.5-7B (28 layers, 4 KV heads, head_dim 128, bf16), 512-token
prompt: 57,344 bytes/token, 29.36 MB per request. Over the lab's measured
node0 -> node1 curve (~0.36 GB/s) one pull takes ~81 ms; node1 -> node0
~25 ms.

## Files

| file | role | key exports |
|---|---|---|
| `src/serving/engine.rs` | scheduler selection, entry points, approximation records | `SchedulerModel::DisaggregatedEngine`, `PhasePipelineReason::{PartiallyOverlappingWorkers, KvTransferUnplannable}`, `IterationEngineError::{KvPlan, WorkerLayout, fallback_reason}`, `disaggregated_engine_approximations` |
| `src/serving/engine/disaggregated.rs` | workers, footprints, plans, role costs, recording | `disaggregated_workers`, `DisaggregatedWorkers`, `WorkerLayoutError`, `DisaggregatedRun`, `run_disaggregated_engine` |
| `src/serving/engine/core.rs` | the loop with handoffs and pulls | `run_engine` (colocated), `run_engine_jobs`, `EngineError::HandoffToSameWorker` |
| `src/serving/engine/types.rs` | job and handoff types | `EngineJob`, `JobRoute`, `Handoff`, `FirstTokenSource`, `HandoffRecord`, `RequestTimeline::handoff`, `WorkerStepCost`, `UniformCost` |
| `src/serving/engine/capacity.rs` | two holdings per request, partial release | `HoldingKey`, `CapacityLedger::{fits_on, never_fits_on, allocate_on, release_sequences_at}` |
| `src/serving/engine/limits.rs` | shared worker/request builders | `EngineWorker::on_gpus`, `engine_request` |
| `src/serving/engine/record.rs` | outcome -> states, step and transfer operations | `record_engine_jobs`, `RecordedEngine` |
| `src/serving/engine/transfer.rs` | flows, plans, FIFO link queues | `KvFlow`, `KvTransferPlan`, `LinkQueues`, `TransferWindow`, `TransferError` |
| `src/serving/engine/transfer/shards.rs` | rank-to-rank KV fractions | `KvShardLayout`, `ShardFlow`, `kv_shard_flows` |
| `src/serving/engine/transfer/plan.rs` | cluster-priced plans | `KvTransferPlanner`, `KvPlanContext`, `PlacedKvLayout`, `PlannedTransfer`, `FlowPricing`, `KvPlanError` |
| `src/serving/topology.rs` | directed queue keys | `kv_route_directed_resource_id` |
| `src/solver/network_cost.rs` | fit hook | `Solver::fitted_kv_transfer_seconds` |
| `src/serving/model/traffic.rs`, `src/config/serving_config/traffic.rs` | first-token knob | `ServingDisaggregatedFirstToken`, `ServingTraffic::disaggregated_first_token`, `parse_disaggregated_first_token` |
| `src/serving/engine/transfer/tests.rs` | shards, link queues, curve vs alpha-beta, direction | - |
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
- At any instant every directed route resource carries at most one flow.
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

Not yet measured. `examples/rtx3090_qwen7b_disaggregated_workload.toml`
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
