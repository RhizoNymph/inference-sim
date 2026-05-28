# Inference Simulator Goals

This document is the roadmap for turning the current MVP into a realistic,
calibratable GPU-cluster inference simulator.

The target is not cycle-accurate simulation. The target is an explainable
planning tool that can compare parallelism, placement, topology,
prefill/decode split, routing, batching, and calibration assumptions well
enough to guide deployment decisions.

## V1 Scope Boundary

V1 should stay deliberately small. The simulator should keep the current
scheduled-timeline approximation, simple TOML configs, coarse topology costs,
bounded solver search, and structured approximation evidence. The detailed
realism checklist below is a roadmap for future versions, not a requirement for
the first usable release.

For V1, avoid adding a full online event loop, exact PCIe/NUMA/NVSwitch
resource contention, full physical fabric graphs, worker-local cache ownership,
autoscaling, failures, or fully realistic prefill/decode service orchestration.
Add those only after the MVP can reliably load configs, run solver sweeps, and
explain why candidates pass, fail, or rely on approximations.

The practical V1 cutoff is a calibratable placement/planning simulator, not a
production-accurate cluster emulator. V1 should include heterogeneous logical
cluster descriptions, prefill/decode pool search, coarse topology-domain
constraints, route/placement rejection evidence, and TTFT/TPOT/throughput/E2EL
reporting. NIC rails, PCIe/NUMA/NVSwitch resource contention, exact
worker-local queue state, and exact KV allocator behavior should remain explicit
future work unless a concrete user scenario proves they are required.

## V1 MVP Acceptance Checklist

This is the practical stop line for the first usable release. Once these gates
pass, new physical-runtime mechanics should stay in the post-v1 backlog unless
they are required to explain a concrete user scenario.

- [x] Load cluster, workload, run, and calibration-profile TOML files through
  the CLI, with cross-file validation before solving.
- [x] Search bounded parallelism and serving candidates, including colocated,
  partially disaggregated, and fully disaggregated prefill/decode pools.
- [x] Support heterogeneous logical clusters with mixed GPU inventories,
  topology domains, node/GPU labels, disabled resources, NIC rails, GPU-to-NIC
  locality, GPU/NIC NUMA-domain maps, custom links, and KV-route constraints.
- [x] Support per-GPU and per-NIC operational state for healthy, disabled,
  maintenance, draining, and reserved resources, with placement/routing avoiding
  non-healthy resources and inventory output preserving the reason.
- [x] Support base-cluster per-GPU HBM, HBM-bandwidth, and TFLOP profile
  overrides so static heterogeneous GPU differences feed placement, memory,
  throughput, and inventory evidence.
- [x] Support base-cluster per-NIC bandwidth and latency overrides so static
  heterogeneous NIC differences feed routing costs and inventory evidence, not
  only run-scenario degradation overlays.
- [x] Support base-cluster GPU/NIC NUMA-domain maps with configurable
  cross-domain GPU-to-NIC bandwidth/latency penalties so default route selection
  can account for same-NUMA versus cross-socket locality without enumerating
  every path override.
- [x] Include route coverage, placement domain spread, GPU/NIC locality,
  rail-dependency, and modeled route-resource contention bottlenecks in the
  topology-risk penalty score used for serving candidate ranking.
- [x] Report TTFT, TPOT, ITL, E2EL, throughput, request outcomes, SLO/deadline
  denominators, and measurement-window provenance for serving candidates.
- [x] Emit calibration, approximation, rejection, bottleneck, placement,
  route-resource, service, memory/utilization, rank-sensitivity, and
  scenario-sensitivity evidence in JSON and CSV artifacts.
- [x] Keep the remaining realism gaps visible as explicit approximations or
  unchecked post-v1 goals, rather than hiding them behind solver scores.
- [x] Pass the release validation commands: `cargo fmt -- --check`,
  `cargo test`, and `cargo clippy --all-targets --all-features -- -D warnings`.

## Current Review Answers

These are the direct answers to the current design questions. The detailed
backlog below expands each item into concrete implementation targets.

- **What is still missing for realistic simulation:** a single authoritative
  online event loop, worker-local queues, worker-local KV/cache ownership,
  exact memory timelines, physical PCIe/NUMA/NVSwitch/NIC/rail/switch paths,
  shared-resource contention, topology-aware placement and online routing,
  layer/backend-aware compute, production traces, measured calibration,
  uncertainty accounting, and bottleneck attribution tied to physical
  resources and policy decisions.
- **Whether more heterogeneous clusters can be expressed today:** yes, at the
  MVP level. Configs can already use node groups, explicit node IDs, mixed-GPU
  node inventories, disabled GPUs/NICs, NIC counts, rail counts, NIC-to-rail
  maps, GPU-to-NIC locality maps, per-GPU/per-NIC path overrides, and custom
  inter-node links scoped by node pair, node group, rail, or local GPU
  endpoint subset. What is still missing is a first-class physical topology
  model with arbitrary endpoint selectors, PCIe/NUMA/NVSwitch/rack/fabric
  resources, asymmetric or missing links, reserved capacity, and contention on
  the resources that those links share.
- **Whether prefill/decode disaggregation is handled today:** yes
  approximately. The solver can search separate prefill and decode pools,
  colocated/partial/full disaggregation modes, routing, KV handoff, and
  TTFT/TPOT/ITL/throughput/E2EL on one scheduled request timeline. Realistic
  disaggregation still needs independent online prefill and decode services,
  service-specific health and scaling, worker-local queues, exact KV
  ownership/allocation, decode-to-prefill backpressure, KV-transfer
  backpressure, and service-specific failure behavior.
- **How to keep the MVP future-proof:** keep stable node/GPU/NIC/rail/link/
  route/request IDs, keep prefill, KV transfer, and decode as separate phases,
  emit structured approximation and rejection evidence, and keep simple TOML
  configs valid while adding optional physical topology sections later.

## Design Review Missing Pieces To Track

This is the explicit checklist from the realism review. Each item should either
be modeled, rejected with structured evidence, or called out as an approximation
in user-facing output.

### Simulation Source Of Truth

- [ ] Replace the current scheduled-timeline approximation with one
  authoritative online event loop for arrivals, admission, queueing, batching,
  prefill, KV handoff, decode iterations, cancellation, timeout, preemption,
  backpressure, and completion.
- [ ] Make request lifecycle events the source of metrics and scheduler state,
  not just derived JSON artifacts.
- [ ] Add worker-local queues and active sets for prefill workers, decode
  workers, KV-transfer workers, cache-owner workers, scheduler shards, and
  admission queues.
- [ ] Track queue delay, resource delay, service time, admission decision,
  timeout decision, cancellation decision, and backpressure cause as separate
  state transitions.
- [x] Add configurable measurement windows, warmup/cooldown trims, and optional
  steady-state detection so trace and synthetic runs can report serving metrics
  separately from startup and drain behavior.
- [x] Emit approximation evidence when serving metrics use an auto-selected
  steady-state window, or when steady-state detection was requested but not
  applied.
- [x] Expose the selected serving measurement window as structured result
  evidence, including configured bounds, steady-state selection status, and
  measured request count.
- [x] Improve steady-state detection with convergence diagnostics and
  uncertainty evidence for selected windows, including sample count, matching
  window count, selected request count, E2EL mean/stddev/CV, and standard
  error.
- [x] Extend steady-state diagnostics with multi-metric candidate evidence
  across TTFT, TPOT, E2EL, queueing phases, request-throughput, aggregate
  output-token throughput, worst-metric CV, and output-token counts.
- [x] Add utilization convergence diagnostics for worker slots, route
  resources, compute, HBM, and KV residency inside steady-state windows.

### Prefill/Decode Disaggregation

- [ ] Model prefill and decode as independent services with separate health,
  scaling, routing, admission, queueing, priority, and worker-capacity state.
- [x] Add v1 service-level health and worker-scale configuration for prefill,
  decode, and KV-transfer services, with solver rejections for unavailable or
  draining required services and JSON evidence for effective worker slots.
- [x] Expose v1 service-level admission/backpressure evidence from the current
  queue-cap paths, including per-service admitted/rejected/timed-out counts,
  queue-cap hit counts, configured queue caps, max/p95 queue pressure, and
  backpressure state in JSON and CSV artifacts.
- [x] Add an optional v1 service-backpressure penalty weight so the solver can
  down-rank candidate configs that hit service queue-cap/backpressure policies
  without making overload rejection the default objective behavior.
- [ ] Keep colocated, partially disaggregated, and fully disaggregated modes on
  the same service/worker/cache primitives instead of one special-case path per
  mode.
- [ ] Add decode-to-prefill backpressure and KV-transfer-to-prefill
  backpressure.
- [x] Emit v1 approximation evidence when decode or KV-transfer queue pressure
  is handled downstream instead of being fed back into prefill admission or
  scheduling.
- [x] Emit v1 generated pool-search deployment-mode mix evidence, so flexible
  searches show whether colocated, partially disaggregated, and fully
  disaggregated pool classes were actually produced.
- [ ] Track exact prefill source worker/GPU, decode owner worker/GPU,
  cache-owner worker/GPU, and KV-transfer worker/path per request.
- [x] Expose per-request prefill, KV-transfer, and decode worker-slot
  assignment evidence from the current scheduler so disaggregated routing and
  worker placement are inspectable.
- [x] Attach current KV block ownership records to decode worker slots and
  decode operation IDs so cache-owner evidence can be traced back to the
  worker execution timeline.
- [x] Add compact per-request worker summaries for prefill source,
  KV-transfer, decode owner, and KV cache-owner roles so downstream consumers
  can inspect worker/cache placement without reconstructing it from raw
  assignment arrays.
- [x] Add v1 logical KV block allocator evidence: each admitted request now
  partitions KV blocks across cache-owner GPUs with stable block ranges,
  block-table overhead, and per-node/per-GPU capacity peaks derived from the
  same ownership records to avoid double-counting.
- [x] Include v1 KV block-table/page-table overhead in serving memory
  components, capacity metrics, worker/node/GPU/class observations, lifecycle
  messages, and CLI JSON output so allocator evidence affects reported
  headroom instead of only appearing as request metadata.
- [x] Thread configured `serving.traffic.kv_block_tokens` into serving
  memory/headroom estimates so block-table overhead follows workload TOML
  instead of always using the default block size.
- [x] Add v1 per decode-worker-slot KV allocator summaries under worker
  observations so GPU-local KV ownership is not reported only as aggregate
  per-GPU capacity evidence.
- [x] Attach exact per-request decode-worker-slot KV block ranges to each
  logical cache-owner allocation so request JSON, worker-slot summaries, and
  capacity peaks are traceable to the same allocator partition.
- [ ] Convert aggregate/per-node/per-GPU decode residency into exact
  per-worker KV block ownership and allocator state.
- [ ] Model KV allocation, block tables, page-table overhead, fragmentation,
  eviction, migration, spill, reuse, prefix-cache residency, and cache misses.
- [ ] Model KV-transfer contention with collectives, activation sends,
  PCIe/NVLink copies, RDMA, storage/control traffic, and other point-to-point
  transfers.
- [ ] Derive TTFT, TPOT, ITL, E2EL, throughput, SLO misses, and deadline misses
  from the same event history.
- [x] Add v1 per-request lifecycle-event metric sourcing for TTFT, TPOT, ITL,
  E2EL, SLO misses, deadline misses, and phase durations, with explicit fallback
  labeling for requests whose lifecycle is rejected or truncated before decode.
- [x] Anchor v1 aggregate serving TTFT, TPOT, ITL, E2EL, throughput, metric
  breakdowns, and measurement-window accounting to the lifecycle-derived
  request observations, so reported solver metrics use the same per-request
  metric source that is emitted for calibration artifacts.
- [x] Expose v1 measurement-window metric-source counts in solver output, so
  aggregate metrics can be audited as lifecycle-derived versus fallback-derived
  without reconstructing every request observation.
- [x] Expose v1 measurement-window request outcome denominators in JSON and
  the serving metrics CSV, including in-window request, completed, failed,
  rejected, timed-out, cancelled, deadline-constrained, and deadline-missed
  counts, so latency/throughput samples can be distinguished from SLO/deadline
  denominators.
- [x] Add v1 request-observation metric-derivation JSON, including the metric
  source, lifecycle endpoint events, measurement-window inclusion, output-token
  count, TPOT sample count, and per-request throughput sample, so JSON-only
  calibration consumers can audit TTFT/TPOT/ITL/E2EL and throughput back to the
  request lifecycle.
- [x] Make v1 request-observation metric-derivation JSON terminal-aware for
  rejected, timed-out, cancelled, or pending requests, with nullable
  decode-derived endpoint events, terminal-event timestamps, and explicit
  unavailable-metric reasons.
- [x] Carry v1 terminal-aware request metric provenance into the request
  metrics CSV, including event-sourced status, terminal event/time,
  unavailable-metric reason, decode endpoint event labels, decode-finish event
  count, and TPOT sample count, so CSV calibration consumers do not need JSON
  to distinguish failed/truncated requests from completed latency samples.
- [x] Carry richer v1 candidate calibration summary fields into the serving
  metrics CSV, including active/calibrated/uncalibrated phase counts,
  extrapolated/unbounded/uncertainty-bearing fit counts, min confidence,
  maximum extrapolation ratio, and soft/hard calibration gate violation counts,
  so CSV ranking artifacts expose the same calibration risk signals as JSON.
- [x] Carry v1 selected hardware-footprint summaries into the serving metrics
  CSV, including aggregate/prefill/decode GPU type counts, GPU label counts,
  HBM capacity, HBM bandwidth, effective TFLOPs, and throughput normalized by
  GPU/HBM/TFLOP, so heterogeneous cluster rankings can be audited without
  joining against JSON hardware-footprint output.

### Heterogeneous Cluster Shape

- [ ] Add first-class endpoint selectors for node groups, explicit nodes, GPUs,
  GPU labels, NICs, rails, racks, islands, failure domains, fabrics, service
  pools, tenants, and models.
- [x] Add GPU-type endpoint selectors for custom inter-node links so mixed-GPU
  clusters can express fabric differences without enumerating local GPU IDs.
- [x] Add v1 GPU tag/label metadata and custom inter-node link endpoint
  selectors so configs can express fabric differences across topology or
  locality subsets such as sockets, NVLink islands, rail affinity, or service
  pools without hard-coding every local GPU ID.
- [x] Add v1 node topology metadata for node labels, racks, islands, and
  failure domains, plus custom inter-node link endpoint selectors over those
  domains so heterogeneous configs can express rack/island/failure-domain
  fabric differences without duplicating node groups.
- [x] Emit v1 topology bottleneck evidence and optional topology-risk penalty
  for selected serving placements concentrated in a single rack, island, or
  failure domain, so heterogeneous domain metadata affects solver explanations
  and ranking when `topology_risk_penalty_weight` is enabled.
- [x] Add v1 prefill/decode pool-search topology-domain spread constraints for
  minimum rack, island, and failure-domain counts, so configs can avoid
  obviously over-concentrated generated pools without modeling physical
  rail/PCIe/NUMA contention.
- [x] Add v1 prefill/decode pool GPU tag filters so service-specific rank
  placement and generated pool search candidates can be constrained to
  topology/locality subsets while preserving node-level pool search.
- [x] Add v1 prefill/decode pool-search node topology include/exclude filters
  for node tags, racks, islands, and failure domains, so heterogeneous service
  pools can be generated from topology subsets without hand-enumerating every
  node or editing base cluster inventory to avoid maintenance domains.
- [x] Add v1 node topology include/exclude filters to explicit named
  prefill/decode pool candidates, so stable service pools can target or avoid
  node tags, racks, islands, and failure domains without duplicating node lists.
- [x] Add v1 topology-domain spread constraints to explicit named
  prefill/decode pool candidates, matching generated pool search for minimum
  rack, island, and failure-domain counts.
- [x] Emit v1 pool-search summary evidence for generated serving pools,
  including topology-filter survivors, GPU-filter survivors, candidate counts,
  truncation, and overlap/mode/domain-spread rejection counts.
- [x] Emit v1 selected serving-pool topology summaries for prefill/decode node
  counts, shared nodes, racks, islands, failure domains, and node labels, so
  heterogeneous and disaggregated pool choices are auditable in solver output.
- [x] Include dedicated prefill/decode node counts in selected serving-pool
  topology summaries, so partially disaggregated pools show shared versus
  role-dedicated capacity directly.
- [x] Emit selected serving hardware GPU-type summaries for aggregate,
  prefill, and decode placements, so heterogeneous solver results show which
  GPU classes each service role actually uses.
- [x] Emit selected serving hardware GPU-label summaries for aggregate,
  prefill, and decode placements, so topology/locality tags used by configs are
  auditable after rank placement.
- [x] Add v1 cluster TOML GPU profile overrides for HBM capacity, HBM
  bandwidth, f16/bf16 tensor throughput, and optional FP8 throughput on both
  homogeneous node entries and explicit per-GPU inventories, so calibrated or
  custom SKUs can be modeled without changing hardcoded profiles.
- [ ] Add optional physical topology sections for CPU sockets, NUMA domains,
  PCIe switches, NVSwitch domains, GPUs, NICs, rails, rack switches, fabric
  switches, copy engines, storage/control fabrics, and failure domains.
- [ ] Let interconnect resources connect arbitrary endpoint selectors, not only
  the current node-pair, node-group, rail, and local-GPU endpoint shortcut
  links.
- [ ] Support asymmetric links, missing links, degraded links, explicit
  disconnected islands, reserved capacity, mixed interconnect generations, and
  separate inference/storage/control fabrics.
- [ ] Extend per-GPU inventory with labels, health state, maintenance state,
  HBM size, HBM bandwidth, FLOPs, supported dtypes, NVLink generation, vendor,
  runtime compatibility, PCIe/NUMA/NVSwitch locality, and placement tags.
- [ ] Extend per-NIC inventory with health, rail membership, bandwidth,
  latency, GPUDirect support, route policy, queue count, degraded state, and
  failure-domain membership.
- [ ] Add topology templates/generators for common layouts: single-node
  NVSwitch, rail-aligned InfiniBand, fat tree, dragonfly, rack-local islands,
  oversubscribed racks, disconnected clusters, and mixed-generation clusters.

### Physical Contention And Routing

- [ ] Model paths such as `GPU -> NVSwitch/PCIe -> NIC -> rail -> fabric ->
  NIC -> PCIe/NVLink -> GPU` as resource sequences with stable IDs.
- [ ] Track sharing and occupancy for GPU compute, HBM bandwidth, copy engines,
  PCIe, NVLink/NVSwitch, NICs, rails, rack/fabric switches, collectives,
  activation sends, KV transfers, storage traffic, control-plane traffic, and
  CPU overhead.
- [ ] Distinguish direct, same-PCIe-switch, same-NUMA, cross-socket,
  host-staged, GPUDirect, degraded, and unavailable GPU-to-NIC paths.
- [ ] Model duplex mode, startup overhead, protocol overhead, route hashing,
  ECMP/adaptive routing, rail pinning, rail striping, oversubscribed uplinks,
  incast/outcast, PFC/head-of-line blocking, and route sharing.
- [ ] Add online routing that considers worker load, queue delay, cache
  affinity, KV ownership, topology path cost, route contention, tenant/model
  constraints, and partial outages.
- [x] Expose structured v1 route-coverage evidence for disaggregated serving
  pools, including routable and unroutable prefill/decode candidate counts in
  solver results, JSON output, and KV-transfer route rejections.
- [x] Add v1 KV-route topology summaries that report route resource counts,
  rail IDs, rail counts, and single-rail dependency for feasible
  prefill/decode disaggregated candidates.
- [x] Emit structured v1 topology bottleneck observations for partial
  prefill/decode route coverage, single-rail KV-route dependency, and missing
  rail metadata on inter-node route resources.
- [x] Report v1 route-locality bottlenecks for cross-socket GPU/NIC paths,
  host-staged or non-GPUDirect KV paths, and slow GPU-to-NIC KV path segments
  using structured topology bottleneck observations.
- [ ] Report dynamic physical contention bottlenecks such as overloaded rails,
  oversubscribed uplinks, route-sharing congestion, switch contention, copy
  engine contention, disconnected pools, and other time-varying topology risk.
- [x] Report v1 dynamic KV route-resource contention bottlenecks when modeled
  KV handoffs queue on shared route resources or when a KV route resource is hot
  in the scheduled timeline.

### Placement Search

- [ ] Search rank, tensor, expert, sequence, context, pipeline, prefill-worker,
  decode-worker, KV-transfer-worker, cache-owner, tenant, and model placements
  over GPUs, nodes, racks, islands, rails, NUMA domains, PCIe domains,
  NVSwitch domains, service pools, and failure domains.
- [ ] Add user-supplied placement constraints for ranks, pools, workers,
  tenants, models, topology domains, failure domains, and service pools.
- [ ] Score placements by route cost, HBM headroom, GPU capability, queueing
  pressure, cache affinity, failure-domain spread, topology risk, cost, power,
  and calibration uncertainty.
- [ ] Add pruning and runtime budgets so topology-aware placement remains
  tractable on large heterogeneous clusters.
- [ ] Explain selected and rejected placements with structured evidence tied to
  physical topology resources and policy constraints.

### Compute, Memory, And Serving Stack

- [ ] Replace coarse FLOP estimates with layer-aware prefill and decode models
  for attention, MLP, MoE routing, logits, sampling, normalization, KV kernels,
  fused kernels, launch overhead, CUDA graphs, and backend effects.
- [ ] Model tensor-core utilization by dtype, batch shape, sequence length,
  hidden size, backend, serving stack, and GPU generation.
- [ ] Model GQA/MQA, KV dtype, quantized KV, quantized weights, mixed
  precision, sliding-window attention, paged attention, LoRA, speculative
  decoding, prefix-cache behavior, and cache misses.
- [x] Add v1 `model.kv_dtype` / `model.kv_cache_dtype` workload config so KV
  cache memory, serving memory pressure, KV transfer bytes, and calibration
  features can use a different precision from weight/compute dtype without
  changing the rest of the model spec.
- [x] Add v1 explicit `model.parameter_count_billion` workload config so
  compute FLOP estimates and calibration features can be decoupled from
  `parameters_gb` weight-memory footprint for quantized or compressed-weight
  models.
- [ ] Replace coarse component memory estimates with time-aware per-worker and
  per-GPU timelines for weights, KV cache, activations, temporary buffers,
  communication buffers, page tables, runtime reserve, graph-capture reserve,
  allocator reserve, and fragmentation.
- [ ] Represent serving stack assumptions for vLLM, TensorRT-LLM, SGLang,
  Dynamo, Ray Serve, Triton, NCCL/RCCL/UCX, CUDA graphs, paged attention,
  prefix cache, and KV-transfer implementation as explicit calibration and
  approximation inputs.
- [x] Emit result-level approximation evidence when serving candidates run
  without a calibration profile or without `profile.serving_stack`, so
  runtime/backend assumptions are visible and can be gated by approximation
  policy.
- [x] Emit result-level calibration-profile dtype evidence for serving
  candidates when profile dtype is missing or mismatched with the workload
  model dtype, so dtype-sensitive latency and memory assumptions are visible.
- [x] Emit result-level calibration-profile hardware and inter-node fabric
  evidence for serving candidates when profile metadata is missing or
  mismatched with the candidate cluster, so hardware/fabric-specific fits are
  visible before using them for comparisons.
- [x] Add optional workload model identity metadata and emit result-level
  calibration-profile model evidence when profile model metadata is missing,
  cannot be checked, or mismatches the workload model family.
- [x] Emit result-level calibration evidence for serving traces whose
  per-request `model_id`s are mixed or only partially covered by the loaded
  calibration profile.
- [x] Carry applied-fit sample count, validation sample count, and source
  metadata into JSON and emit serving calibration evidence when applied fits
  lack sample, holdout, source, or uncertainty metadata.
- [x] Add calibration policy gates for minimum applied-fit sample count and
  validation/holdout sample count so weak fitted candidates can be warned or
  rejected.
- [x] Add calibration policy gates for missing applied-fit source and
  uncertainty metadata so unauditable fitted candidates can be warned or
  rejected.
- [x] Add calibration policy gates for missing calibration-profile source and
  date metadata so unauditable or stale-profile candidates can be warned or
  rejected.
- [x] Emit result-level phase calibration evidence when a loaded calibration
  profile does not apply fits to active prefill, decode, or KV-transfer serving
  phases.
- [x] Emit structured per-phase serving calibration status in JSON, including
  active/calibrated flags, applied fit targets, fit counts, estimated latency,
  and status for prefill, decode, and KV transfer.
- [x] Emit structured calibration coverage for active queue components, including
  prefill queue, decode queue, KV-transfer worker queue, and KV route-resource
  queue delay, so queueing can be gated as simulator-only evidence instead of
  being hidden inside TTFT/E2EL.
- [x] Add result-level calibration policy gates for active serving phases and
  queue components that have no applied fit when a calibration profile is loaded,
  so configured coverage policy can warn or reject candidates instead of only
  displaying uncalibrated phase evidence.
- [x] Apply aggregate serving calibration fits for TTFT, TPOT, output-token
  throughput, and E2EL summary metrics, so fitted serving-stack measurements can
  influence the solver objective while preserving the simulated request timeline
  as separate evidence.
- [x] Make serving memory overhead assumptions calibratable in TOML/profile
  inputs, including temporary-buffer, activation-communication,
  weight-communication, runtime-reserve, and fragmentation fractions, and emit
  those coefficients in calibration JSON.

### Workload, Policy, And Admission

- [ ] Add production-like trace fixtures, replay transforms, warmup trimming,
  downsampling, repeat windows, and lifecycle metadata.
- [x] Model bursty, diurnal, self-similar, and trace-derived arrivals.
- [x] Add a deterministic bursty synthetic arrival mode with configurable burst
  size, burst interval, and intra-burst request gap.
- [x] Add a seeded diurnal synthetic arrival mode with configurable minimum
  rate, maximum rate, period, phase, and seed.
- [x] Add a seeded self-similar synthetic arrival approximation with
  configurable mean rate, Pareto shape, max-gap cap, and seed.
- [x] Add trace-derived arrival mode so production trace timestamps can drive
  synthetic request-shape sweeps without letting trace rows override shape and
  metadata.
- [ ] Add correlated prompt/decode sizes, multi-model mixes, tenant/model
  routing constraints, cache keys, retries, streaming disconnects,
  cancellations, deadlines, timeout policies, and per-request class metadata.
- [x] Add weighted synthetic shape profiles so batch size, prompt tokens, decode
  tokens, and max sequence length can be sampled as correlated request tuples.
- [x] Allow synthetic shape profiles to carry tenant, model, cache-key,
  priority, and prefix-cache metadata so multi-tenant/model mixes can exercise
  routing, traffic-class, prefix-cache, and calibration evidence without a
  full trace file.
- [x] Add stable v1 shape-profile names for synthetic workload mixes and carry
  the selected profile into request observations, request-scoped CSV artifacts,
  and metric breakdowns so tenant/model shape mixes can be audited per request.
- [x] Allow synthetic shape profiles to carry relative deadline and cancellation
  offsets so deadline-miss and streaming-disconnect behavior can be exercised
  without a full trace file.
- [x] Allow synthetic shape profiles to carry per-request TTFT/TPOT/ITL/E2EL
  SLOs and request timeouts so mixed synthetic workloads can exercise
  request-specific latency policy without a full trace file.
- [ ] Extend traffic classes beyond current SLO defaults, queue caps, timeouts,
  hard miss limits, soft penalties, and admission priority with class-scoped
  capacity policies, burst limits, reserved capacity, cancellation behavior,
  tenant isolation, class interaction rules, and multi-objective weights.
- [ ] Extend admission control beyond queue delay and aggregate capacity caps to
  max prefill tokens, max decode batch tokens, SLO pressure, worker-local
  service state, tenant/model constraints, reserved capacity, and topology
  domains.
- [ ] Add priority, fairness, preemption, aging, starvation prevention, and
  retry policies.

### Metrics, Calibration, And Validation

- [ ] Add benchmark-backed fits for compute, collectives, RDMA, PCIe/NVLink
  copies, KV transfer, queueing, and end-to-end serving.
- [ ] Fit profiles by hardware, fabric, model family, dtype, backend, driver,
  CUDA/ROCm version, NCCL/RCCL/UCX version, kernel settings, and serving stack.
- [ ] Add provenance, coverage scoring, confidence intervals, holdout
  validation, interpolation/extrapolation policy, and rank sensitivity.
- [x] Add v1 holdout-validation error metadata for calibration fits
  (`validation_rmse`, `validation_rmse_pct`, validation mean/max absolute
  percent error), expose it in fit evidence, and prefer it over training-fit
  error when deriving fit confidence and uncertainty.
- [x] Add v1 confidence-interval metadata for calibration fits
  (`confidence_interval`, `confidence_interval_pct`, `confidence_level`),
  expose it in profile and applied-fit JSON, and prefer interval bounds when
  deriving fit uncertainty for uncertainty-adjusted rankings and fit gates.
- [x] Add a v1 calibration-policy threshold for minimum fit confidence level,
  so configs can warn or reject fits whose reported uncertainty interval is
  below the required coverage level.
- [x] Add v1 calibration-policy gates for partially bounded and unbounded
  calibration fits, so configs can separately warn or reject fitted models that
  lack complete feature-range evidence even when they are not numerically
  extrapolated.
- [x] Count non-latency calibration uncertainty in candidate summaries and
  rank-sensitivity evidence, so throughput/value fitted metrics are not
  underreported as lacking uncertainty just because their uncertainty is not
  measured in seconds.
- [x] Add v1 calibration-profile runtime provenance metadata for backend,
  driver, CUDA/ROCm, NCCL/RCCL, UCX, kernel settings, and environment hash, and
  emit it in JSON so fitted metrics can be audited against the measured stack.
- [x] Add a v1 calibration-policy gate for incomplete profile runtime
  provenance, so configs can warn or reject profiles missing backend, driver,
  accelerator runtime, communication library, kernel setting, or environment
  hash evidence.
- [x] Emit per-benchmark calibration residuals in JSON, including measured-vs-
  predicted latency error in milliseconds and percent plus a compact residual
  status, so calibration profiles expose their error surface without external
  post-processing.
- [x] Add a v1 calibration residuals CSV artifact, available from run TOML or
  `--calibration-residuals-csv`, so measured-vs-predicted benchmark errors can
  be exported for calibration dashboards and regression checks.
- [x] Propagate uncertainty into ranking and report when results depend on
  coarse topology, ignored contention, aggregate memory, uncalibrated fits, or
  unsupported runtime effects.
- [ ] Add bottleneck attribution by pool, worker, node, GPU, NIC, rail, link,
  switch, copy engine, tenant, model, phase, request, objective term, and
  physical route.
- [x] Add v1 objective-term bottleneck attribution for selected serving
  candidates, including base objective metric, SLO penalty, service
  backpressure penalty, and topology-risk penalty rows in JSON and the serving
  bottlenecks CSV.
- [x] Add v1 request-scoped bottleneck rows to the serving bottlenecks CSV,
  including request ID, tenant, model, traffic class, request outcome,
  deadline misses, and TTFT/TPOT/ITL/E2EL SLO misses so problematic requests
  can be attributed without parsing nested JSON.
- [x] Add stable CSV/JSON artifacts for metrics, utilization, timelines,
  occupancy, memory pressure, request lifecycle, calibration, route paths,
  placement evidence, rejection evidence, and scenario sensitivity.
- [x] Add a v1 calibration-ready serving request metrics CSV artifact, available
  from run TOML or `--request-metrics-csv`, with scenario/candidate context,
  request lifecycle metric source, TTFT, TPOT, ITL, E2EL, throughput sample,
  queue/phase durations, SLO miss flags, and deadline miss flags.
- [x] Add a v1 serving request lifecycle-events CSV artifact, available from
  run TOML or `--request-lifecycle-events-csv`, with scenario/candidate/request
  context, metric source, ordered lifecycle events, phases, timestamps, decode
  iteration indexes, and event messages so TTFT/TPOT/E2EL derivation can be
  audited outside nested JSON.
- [x] Add a v1 serving candidate metrics CSV artifact, available from run TOML
  or `--serving-metrics-csv`, with scenario/candidate context, deployment mode,
  prefill/decode pools, TTFT, TPOT, ITL, E2EL, throughput, request outcomes,
  calibration uncertainty bounds for TTFT, TPOT, ITL, E2EL, and throughput,
  SLO constrained/missed denominators for TTFT, TPOT, ITL, E2EL, and deadline,
  capacity peaks, calibration status, approximation status, approximation
  category/top-code summaries, bottleneck counts, top bottleneck attribution,
  rejection counts, and top rejection attribution.
- [x] Add a v1 serving metric-breakdowns CSV artifact, available from run TOML
  or `--serving-metric-breakdowns-csv`, with scenario/candidate context,
  deployment mode, pool, breakdown group/key, request outcomes, output tokens,
  scoped throughput, metric-source counts, TTFT, TPOT, ITL, E2EL, tail
  latencies, deadline misses, SLO denominators, and SLO miss rates for
  tenant/model/traffic-class/priority/topology groups.
- [x] Add a v1 serving services CSV artifact, available from run TOML or
  `--serving-services-csv`, with scenario/candidate context, deployment mode,
  pool, prefill/decode/KV-transfer service health, worker scale, worker-slot
  capacity, admission outcomes, queue caps, backpressure, queue timing,
  service timing, and worker-slot utilization.
- [x] Add a v1 KV-route resource CSV artifact, available from run TOML or
  `--kv-route-resources-csv`, with scenario/candidate context, route resource
  IDs, resource kind, labels, rail IDs, endpoints, transfer bytes, path
  observations, and estimated route-resource service time.
- [x] Add a v1 serving bottleneck CSV artifact, available from run TOML or
  `--serving-bottlenecks-csv`, with scenario/candidate context, bottleneck
  source, phase, category, resource, code, severity, observed/limit values,
  units, messages, and remediation hints.
- [x] Add a v1 serving phase-calibration CSV artifact, available from run TOML
  or `--serving-phase-calibration-csv`, with scenario/candidate context,
  deployment mode, pool, active/calibrated phase flags, applied targets,
  estimated phase time, phase uncertainty, fit confidence/extrapolation status,
  and candidate-level calibration summary fields.
- [x] Add a v1 serving approximations CSV artifact, available from run TOML or
  `--serving-approximations-csv`, with scenario/candidate context,
  deployment mode, pool, approximation records, approximation-policy
  violations, phases, categories, scopes, codes, messages, actions, and
  remediation hints.
- [x] Add a v1 serving utilization CSV artifact, available from run TOML or
  `--serving-utilization-csv`, with scenario/candidate context, deployment
  mode, pool, phase resource utilization, scheduled resource utilization,
  busy time, utilization fraction, operation counts, and observation windows.
- [x] Add a v1 serving memory-pressure CSV artifact, available from run TOML
  or `--serving-memory-pressure-csv`, with scenario/candidate context,
  deployment mode, pool, phase windows, active requests/tokens, KV blocks,
  per-GPU HBM pressure, headroom, limiting GPU, dominant component, and memory
  component breakdowns.
- [x] Add a v1 serving timeline CSV artifact, available from run TOML or
  `--serving-timeline-csv`, with scenario/candidate context, deployment mode,
  pool, scheduled operation IDs, inferred phase, resources, start/finish times,
  durations, and dependency IDs for calibration and critical-path analysis.
- [x] Add a v1 serving occupancy CSV artifact, available from run TOML or
  `--serving-occupancy-csv`, with scenario/candidate context, deployment mode,
  pool, resource IDs, bucket windows, busy time, utilization fraction, and
  operation counts using the same bucket controls as JSON occupancy output.
- [x] Add a v1 serving placement-evidence CSV artifact, available from run TOML
  or `--serving-placement-evidence-csv`, with scenario/candidate context,
  deployment mode, pool, prefill/decode phase, placement decision, scope,
  resource, code, observed/limit values, unit, message, and remediation.
- [x] Add a v1 serving worker/KV evidence CSV artifact, available from run TOML
  or `--serving-worker-evidence-csv`, with scenario/candidate/request context,
  worker summaries, worker-slot assignments, KV-cache owner records, block
  ranges, decode sequence counts, resident tokens, allocated tokens, and block
  table bytes for auditing disaggregated prefill/decode ownership outside
  nested JSON.
- [x] Add a v1 serving rejections CSV artifact, available from run TOML or
  `--serving-rejections-csv`, with scenario/candidate context, deployment
  mode, pool, phase, category, resource, code, observed/limit values, unit,
  message, and remediation.
- [x] Add a v1 serving route-paths CSV artifact, available from run TOML or
  `--serving-route-paths-csv`, with scenario/candidate context, request
  identity, prefill/decode transfer path source/destination GPUs, path transfer
  bytes, route resources, resource IDs, rail IDs, endpoint details, latency,
  bandwidth, and estimated route service time.
- [x] Add v1 JSON scenario-sensitivity summaries for serving sweeps, comparing
  each scenario's top candidate TTFT, TPOT, throughput, E2EL, calibration
  summary, heterogeneous hardware-footprint summary, approximation summary
  with category/top-code evidence, top bottleneck evidence, and rejection
  evidence against the first available serving baseline scenario.
- [x] Add a v1 scenario-sensitivity CSV artifact, available from run TOML or
  `--scenario-sensitivity-csv`, with scenario context, baseline identity,
  top-candidate identity/status, deployment mode, TTFT, TPOT, throughput,
  E2EL, absolute/percentage deltas against the first available baseline,
  selected hardware footprint, calibration status/uncertainty, approximation
  counts/flags/top codes, and top bottleneck/rejection attribution.
- [x] Add a v1 rank-sensitivity CSV artifact, available from run TOML or
  `--rank-sensitivity-csv`, with scenario/candidate context, solver mode,
  feasibility, nominal rank, uncertainty-adjusted rank, rank delta, nominal
  and uncertainty-adjusted scores, calibration uncertainty, fit uncertainty
  coverage count, minimum confidence, maximum extrapolation, applicability
  status, serving hardware-footprint summaries, rejection and top bottleneck
  context, and key parallelism or serving metrics.
- [x] Add v1 per-candidate objective breakdown output that ties the selected
  solver objective to its base metric, score convention, SLO/backpressure/
  topology penalties, uncertainty-adjusted score, and largest nominal objective
  term so ranking decisions are inspectable.

### Operations, Reliability, And Testing

- [ ] Model disabled, degraded, reserved, draining, maintenance, and failed
  states for nodes, GPUs, NICs, links, rails, switches, racks, fabrics, service
  pools, and failure domains.
- [ ] Add rolling upgrades, failover capacity, anti-affinity, replica spread,
  rack/power/thermal constraints, and operational-risk scoring.
- [x] Add v1 run-scenario rail degradation overlays that can scale bandwidth
  and latency for matching NIC rails and rail-scoped custom inter-node links
  without duplicating the base cluster file.
- [x] Add v1 run-scenario custom-link degradation selectors for node tags,
  racks, islands, and failure domains, so heterogeneous scenario sweeps can
  degrade topology-scoped links without enumerating every node pair.
- [x] Add v1 run-scenario node-scoped topology selectors for node-state,
  disabled GPU/NIC, degraded GPU/NIC, and degraded rail overlays, so
  heterogeneous scenario sweeps can target node tags, racks, islands, and
  failure domains without copying node IDs.
- [x] Preserve v1 scenario node-state labels (`disabled`, `maintenance`,
  `draining`, `reserved`) in cluster inventory evidence while conservatively
  removing those nodes from available GPU/NIC capacity.
- [ ] Expand run-scenario overlays beyond rail to cover switch, rack, fabric,
  partial-reserved-capacity, draining policy, and rolling-maintenance changes
  without duplicating the base cluster file.
- [ ] Add cross-file validation for duplicate IDs, disconnected pools,
  impossible ranks, invalid rails, invalid affinity, unsupported dtypes,
  incompatible hardware, invalid GPU-to-NIC paths, missing links, invalid
  disaggregation routes, and missing calibration coverage.
- [x] Add v1 CLI invalid-config smoke coverage for duplicate node IDs, invalid
  rail counts, invalid NIC affinity, invalid GPU-to-NIC paths, impossible rank
  placement, unsupported dtype/hardware combinations, disconnected
  prefill/decode pools, missing KV links, and invalid fully-disaggregated
  serving routes, so bad TOML pairs fail before solver output is written.
- [x] Add v1 serving trace identity validation so duplicate inline or imported
  `request_id` values fail fast after trace window/replay processing, keeping
  request metrics, lifecycle events, bottlenecks, and worker/KV evidence
  unambiguous.
- [x] Add v1 traffic-class selector validation so two classes cannot target the
  same tenant/model/priority selector with different names, avoiding hidden
  first-match policy precedence in SLO, capacity, and attribution reports.
- [x] Add opt-in cross-file validation for routable prefill/decode serving
  pools via `serving.require_routable_pools`, so production-style configs can
  fail fast on disconnected KV handoff routes while failure scenarios can still
  leave the gate disabled and observe solver rejections.
- [x] Add v1 cross-file validation for configured KV-route constraints, so
  `serving.min_kv_route_rail_count`, `serving.require_kv_route_rail_metadata`,
  and `serving.require_gpudirect_kv_paths` fail fast when no configured static
  or generated prefill/decode pool can satisfy the requested rail/GPUDirect
  shape, using explicit prefill/decode placements as the route endpoints when
  they are configured.
- [x] Add cross-file validation that explicit `serving.prefill_placement` and
  `serving.decode_placement` ranks fit at least one configured serving pool,
  including static pools, named pool candidates, pool-search candidates, and
  GPU-label filters, so disaggregated configs cannot silently place prefill
  work on decode-only GPUs or vice versa.
- [x] Add v1 cross-file validation and rank-placement filtering for
  `model.dtype = "fp8"` so base search, explicit placements, serving prefill
  and decode placements, and serving pool capacity checks only use available
  GPUs with FP8 throughput metadata instead of silently falling back to f16
  throughput on A100-class pools.
- [x] Add v1 workload shape validation for model dimensions and request
  sequence lengths, including positive model/request fields, hidden-size to
  attention-head divisibility, valid GQA/MQA head ratios, and
  `max_sequence_tokens >= prompt_tokens + decode_tokens`, so invalid shapes
  fail before FLOP, KV-memory, or TTFT/TPOT estimates silently truncate.
- [x] Add v1 serving traffic shape validation for inline requests, CSV traces,
  and weighted shape profiles, so per-request `max_sequence_tokens` cannot be
  configured below `prompt_tokens + decode_tokens` before KV residency,
  transfer, and decode metrics are derived.
- [x] Add checked example fixtures and CLI smoke tests for homogeneous,
  heterogeneous, islanded, rail-aware, oversubscribed, colocated, partially
  disaggregated, and fully disaggregated setups.
- [x] Add a compact homogeneous colocated serving example workload plus CLI
  smoke test against the H100 cluster fixture, so the simplest serving path has
  checked TTFT, TPOT, throughput, E2EL, service evidence, and local KV-transfer
  output.
- [x] Add a compact heterogeneous fully-disaggregated example workload and CLI
  smoke test that exercises TOML pool topology filters, route evidence,
  selected pool topology output, and TTFT/TPOT/throughput/E2EL reporting.
- [x] Add compact heterogeneous colocated and partially-disaggregated example
  workloads plus CLI smoke tests, so all explicit serving deployment modes have
  checked examples with TTFT, TPOT, throughput, E2EL, and pool-topology output.
- [x] Add a compact rail-aware, islanded heterogeneous example cluster and
  fully-disaggregated workload with CLI smoke coverage for different
  interconnects between node sets, rack/island/failure-domain metadata,
  route-resource evidence, topology bottlenecks, and TTFT/TPOT/throughput/E2EL
  reporting.
- [x] Add a compact oversubscribed heterogeneous example cluster and
  fully-disaggregated workload with CLI smoke coverage for oversubscribed link
  bandwidth, single-rail dependency, modeled KV route-resource queueing, route
  resource summaries, and TTFT/TPOT/throughput/E2EL reporting.
- [x] Add a v1 calibrated trace-run example smoke test that runs
  `examples/trace_run.toml` and checks calibration profile provenance,
  applied-fit uncertainty, phase calibration, and TTFT/TPOT/throughput/E2EL
  output.
- [x] Add a v1 deterministic JSON contract test for the checked serving
  examples, normalizing only runtime duration, so homogeneous,
  heterogeneous, islanded, rail-aware, oversubscribed, colocated, partially
  disaggregated, and fully disaggregated examples have stable machine-readable
  output for MVP regression checks.
- [x] Add a v1 deterministic CLI JSON edge-outcome regression for TOML-loaded
  request admission rejection, timeout, cancellation, SLO/deadline miss,
  lifecycle events, rejection evidence, and per-request metric derivation.
- [x] Parse `serving.traffic.arrival_gap_ms` alongside `arrival_gap_s`, so
  millisecond fixed-gap arrival settings used by example TOML files are not
  silently ignored.
- [x] Add a v1 deterministic CLI JSON control-plane regression for TOML-loaded
  fixed arrivals, seeded Poisson arrivals, trace replay, continuous
  prefill/decode batching, topology-aware routing, queueing, route contention,
  and topology diagnostics.
- [x] Add a README quickstart that documents the v1 MVP contract, key example
  commands, calibrated trace-run path, supported evidence artifacts, and the
  realism boundary that remains post-v1.

## Missing-Piece Goal Index

This is the high-level inventory of what remains to make the simulator
realistic enough for stronger capacity and deployment recommendations. The
sections below expand each item into concrete backlog tasks, MVP guardrails,
and suggested implementation phases.

- **Authoritative online serving simulation:** one event loop for arrivals,
  admission, worker queues, batching, active decode sets, preemption,
  cancellation, timeout, backpressure, and completion.
- **Production prefill/decode disaggregation:** independent prefill and decode
  services with separate health, queues, worker capacity, routing, admission,
  scaling, backpressure, and KV ownership.
- **Exact KV cache state:** per-worker/GPU KV allocation, block tables,
  residency, ownership, fragmentation, eviction, migration, spill, reuse, and
  prefix-cache behavior.
- **Physical heterogeneous topology:** first-class CPU sockets, NUMA domains,
  PCIe switches, NVSwitch domains, GPUs, NICs, rails, rack switches, fabric
  switches, copy engines, storage/control fabrics, and failure domains.
- **Arbitrary heterogeneous configs:** endpoint selectors for nodes, GPUs,
  GPU labels, NICs, rails, racks, islands, fabrics, service pools, and failure
  domains, while keeping simple node-group TOML concise.
- **Shared-resource contention:** occupancy and contention policy for GPU
  compute, HBM, copy engines, PCIe, NVLink/NVSwitch, NICs, rails, switches,
  collectives, KV transfers, activation sends, storage, control-plane traffic,
  and CPU overhead.
- **Topology-aware placement and routing:** rank, pipeline, expert, prefill,
  decode, cache-owner, tenant, and model placement/routing over physical
  topology, load, cache affinity, outages, constraints, and uncertainty.
- **Communication realism:** calibrated collective algorithms, point-to-point
  copies, RDMA/GPUDirect behavior, protocol thresholds, chunking, startup
  costs, route sharing, and congestion assumptions.
- **Layer/backend-aware compute:** attention, MLP, MoE, logits, sampling,
  KV kernels, fused kernels, launch overhead, CUDA graphs, paged attention,
  speculative decoding, quantization, dtype, GPU generation, and serving stack
  effects.
- **Time-aware memory accounting:** per-worker/per-GPU timelines for weights,
  activations, KV cache, temporary buffers, communication buffers, page tables,
  runtime reserve, graph-capture reserve, allocator reserve, and
  fragmentation.
- **Production workload realism:** richer trace replay, bursty/diurnal arrivals,
  correlated prompt/decode lengths, multi-model mixes, tenant constraints,
  cache keys, retries, streaming disconnects, cancellations, deadlines, and
  timeout policy.
- **Calibration and uncertainty:** benchmark-backed fits for compute,
  collectives, PCIe/NVLink copies, RDMA, KV transfer, queueing, and
  end-to-end serving, with provenance, coverage, confidence intervals,
  holdouts, and interpolation/extrapolation policy.
- **Bottleneck attribution and reporting:** resource- and policy-level
  explanations by pool, worker, node, GPU, NIC, rail, link, switch, tenant,
  model, phase, request, objective term, and physical route.
- **Operational scenario modeling:** disabled/degraded/reserved/maintenance
  state for nodes, GPUs, NICs, links, rails, switches, racks, fabrics, plus
  rolling upgrades, draining, failover capacity, power, thermal, and
  failure-domain risk.
- **Validation, fixtures, and solver controls:** cross-file validation, golden
  configs, invalid-config tests, CLI smoke tests, deterministic JSON,
  runtime/pruning budgets, Pareto frontiers, sensitivity analysis, and grouped
  rejection root causes.

## Current Trust Boundary

The simulator is already useful for relative exploration at the node, pool,
parallelism, coarse topology, and approximate serving-policy level. It should
not yet be treated as an exact production latency model or as proof of
fine-grained locality behavior.

Current strengths:

- TOML cluster, workload, run, and calibration-profile inputs.
- Homogeneous clusters and custom heterogeneous clusters.
- Node groups, node IDs, per-node GPU type/count, NIC count, NIC bandwidth,
  NIC affinity, explicit per-NIC rail membership, explicit per-GPU NIC locality
  maps, optional per-GPU/per-NIC path bandwidth/latency/GPUDirect overrides,
  rail count, intra-node fabric, disabled GPU/NIC state, and pairwise or
  group-expanded inter-node links.
- Disabled GPU and NIC state in configs and run-scenario overlays for
  first-pass degraded or maintenance scenarios.
- Explicit per-node GPU inventories for mixed-GPU nodes, so one node can
  contain different local GPU types while preserving stable local GPU IDs.
- Cluster TOML can override per-GPU HBM capacity, HBM bandwidth, f16/bf16
  throughput, and FP8 throughput for calibrated or custom hardware profiles,
  including mixed explicit GPU inventories.
- Workload TOML can set KV cache dtype separately from model compute/weight
  dtype, and v1 uses it for KV memory, KV-transfer byte sizing, and calibration
  fit features.
- Workload TOML can set parameter count separately from model weight-memory
  footprint, so compute estimates can represent quantized or compressed-weight
  deployments without inflating HBM requirements.
- Workload parsing validates model/request dimensions that feed FLOP, KV-cache,
  and transfer sizing, so impossible or silently truncating shapes are rejected
  early.
- Serving traffic parsing validates inline request, CSV trace, and shape-profile
  max-sequence consistency before deriving per-request KV residency, transfer,
  and decode metrics.
- Explicit per-GPU NIC locality maps for heterogeneous nodes where the simple
  dedicated/shared/uniform affinity formulas are too coarse.
- Disabled GPU and NIC state in cluster TOML and run-scenario topology
  overlays, with placement and topology routing avoiding unavailable resources
  and inventory output reporting physical versus available capacity.
- Custom inter-node links can be scoped to one rail, a list of rails, or
  explicit local-GPU endpoint subsets, allowing different fabrics between the
  same node groups on different NIC rails or between selected GPU pairs.
- Heterogeneity-aware rank placement that considers HBM capacity, dtype-relevant
  FLOPs, HBM bandwidth, HBM size, and stable IDs.
- FP8 workloads fail fast when the configured cluster, explicit placements, or
  serving pools target GPUs without FP8 throughput metadata, and rank placement
  filters those GPUs out before scoring.
- Explicit rank placements from workload TOML for the base solver and for
  serving prefill/decode pools, with cluster/search validation and structured
  selected/rejected placement evidence.
- Candidate-level placement evidence for selected and rejected rank placements,
  including decision, scope, resource, code, observed value, limit, unit,
  remediation, and message. Serving JSON emits separate prefill and decode
  placement evidence so disaggregated pool placement is auditable.
- Bounded search over tensor, pipeline, expert, data, prefill, decode, and
  serving-pair candidates.
- Basic scenario sweeps from run TOML files, including traffic/request-shape
  scales, per-scenario calibration profile selection, and per-scenario scalar
  calibration overrides, plus coarse per-scenario topology degradation for
  interconnect bandwidth, interconnect latency, node NIC bandwidth, node-state
  overlays for disabled, maintenance, draining, and reserved nodes, disabled
  GPU/NIC overlays, per-GPU and per-NIC bandwidth/latency overlays, and
  rail-level bandwidth/latency overlays. Node-scoped scenario overlays can
  target node IDs, node groups, node tags, racks, islands, and failure domains,
  and custom-link degradation overlays can target node-pair, node-tag, rack,
  island, failure-domain, rail, and local-GPU endpoint-scoped links. Scenario
  node-state labels are preserved in cluster inventory evidence even though v1
  treats all non-healthy node states as unavailable capacity.
- Serving scenario-sweep JSON includes a compact `scenario_sensitivity`
  artifact for top-candidate TTFT, TPOT, throughput, E2EL, calibration,
  approximation, top-bottleneck, and rejection deltas/evidence against the
  first available serving scenario.
- Serving scenario sweeps can optionally write a stable scenario-sensitivity
  CSV artifact for the same top-candidate TTFT, TPOT, throughput, and E2EL
  baseline deltas plus calibration, approximation, bottleneck, and rejection
  evidence, so failure and calibration sweeps can be compared without parsing
  nested JSON.
- Parallelism and serving solves can optionally write a stable rank-sensitivity
  CSV artifact, so nominal rank, uncertainty-adjusted rank, rank deltas,
  calibration uncertainty, fit uncertainty coverage count, minimum confidence,
  maximum extrapolation, applicability status, and the objective scores behind
  candidate ordering can be compared without parsing nested JSON.
- Workload validation can optionally require routable serving pools with
  `serving.require_routable_pools = true`, catching disconnected prefill/decode
  KV handoff configs before solving while preserving the default failure-
  simulation behavior.
- Approximate prefill/decode disaggregation with separate pools, separate search
  spaces, pool candidates, pool search, request routing, KV handoff cost, and
  shared TTFT/TPOT/ITL/throughput/E2EL reporting. Serving configs can now
  explicitly request flexible, colocated, partially disaggregated, or fully
  disaggregated pool modes, and output reports each scored candidate's
  effective deployment mode.
- Request-level serving observations now label their metric source and derive
  TTFT, TPOT, ITL, E2EL, SLO misses, deadline misses, and phase durations from
  lifecycle events when a request reaches decode, preserving explicit fallback
  evidence for rejected or truncated requests.
- Aggregate serving metrics and metric-breakdown throughput now use the
  lifecycle-derived request observations selected by the measurement window,
  so TTFT, TPOT, ITL, E2EL, output-token throughput, SLO misses, and deadline
  misses are auditable against the same per-request evidence used by CSV
  calibration artifacts.
- Measurement-window output reports metric-source counts for measured requests,
  including lifecycle-event and fallback request counts, making aggregate metric
  provenance visible in top-level JSON.
- Measurement-window output reports in-window request outcome counts separately
  from completed measured requests, so calibration consumers can distinguish
  latency/throughput sample counts from SLO/deadline failure denominators.
- Request-observation JSON marks whether each completed request is included in
  the aggregate measurement window and reports actual emitted output tokens, so
  JSON-only calibration consumers can trace aggregate metrics back to rows.
- Request-observation JSON now also includes a compact `metric_derivation`
  block with lifecycle endpoint event names, timestamp endpoints, TPOT sample
  counts, and request output-token throughput, so JSON-only calibration
  consumers can audit each request metric without joining the lifecycle CSV.
- Metric-derivation JSON now marks failed or truncated requests with terminal
  event labels/timestamps, nullable decode-derived endpoints, and explicit
  unavailable-metric reasons, avoiding misleading decode-finish provenance for
  requests that never reached decode.
- Serving solves can optionally write a stable request-metrics CSV artifact for
  calibration workflows, so TTFT/TPOT/ITL/E2EL, actual emitted output-token
  counts, per-request throughput samples, metric source, and measurement-window
  inclusion can be consumed without scraping nested JSON.
- Serving solves can optionally write a stable request lifecycle-events CSV
  artifact, so arrival, prefill, KV-transfer, KV-cache, decode-iteration, and
  terminal events that source request metrics can be audited and calibrated
  without parsing nested JSON.
- Serving solves can optionally write a stable candidate-metrics CSV artifact,
  so TTFT/TPOT/ITL/E2EL, throughput, deployment mode, pool placement,
  measurement-window source, lifecycle/fallback metric-source counts,
  metric-level calibration uncertainty bounds, calibration status,
  approximation status, and rejection counts can be compared across top
  candidates and scenario sweeps without parsing nested JSON.
- Serving solves can optionally write a stable metric-breakdowns CSV artifact,
  so tenant, model, traffic-class, shape-profile, priority, node, and route
  scoped TTFT, TPOT, ITL, E2EL, throughput, request outcomes, metric-source
  counts, and SLO denominators can be compared across candidates and scenario
  sweeps without parsing nested JSON.
- Serving solves can optionally write a stable services CSV artifact, so
  prefill, decode, and KV-transfer service health, worker capacity, queue caps,
  backpressure, queue timing, service timing, and worker-slot utilization can
  be compared across disaggregated candidates and scenario sweeps without
  parsing nested JSON.
- Serving solves can optionally write a stable KV-route resource CSV artifact,
  so disaggregated route resources, rails, endpoints, transfer bytes, and
  route-service estimates can be compared across heterogeneous candidates and
  scenario sweeps without parsing nested JSON.
- Serving solves can optionally write a stable bottleneck CSV artifact, so
  memory, topology, queueing, calibration, utilization, and rejection evidence
  can be compared across heterogeneous candidates and scenario sweeps without
  parsing nested JSON.
- Serving solves can optionally write a stable phase-calibration CSV artifact,
  so active prefill/decode/KV/queue calibration coverage, applied targets,
  uncertainty, fit confidence, and extrapolation status can be compared across
  candidates and scenario sweeps without parsing nested JSON.
- Serving solves can optionally write a stable approximation-evidence CSV
  artifact, so v1 simulator assumptions and approximation-policy violations
  can be audited beside candidate metrics and calibration evidence without
  parsing nested JSON.
- Serving solves can optionally write a stable utilization CSV artifact, so
  phase-level and scheduled-resource utilization can be compared across
  heterogeneous candidates and scenario sweeps without parsing nested JSON.
- Serving solves can optionally write a stable memory-pressure CSV artifact,
  so HBM/KV pressure, limiting GPUs, and component-level memory estimates can
  be compared across heterogeneous candidates and scenario sweeps without
  parsing nested JSON.
- Serving solves can optionally write a stable timeline CSV artifact, so
  prefill, KV-transfer, decode, and other scheduled operations can be compared
  across heterogeneous candidates and scenario sweeps without parsing nested
  JSON.
- Serving solves can optionally write a stable occupancy CSV artifact, so
  bucketed resource busy time and utilization can be compared across
  heterogeneous candidates and scenario sweeps without parsing nested JSON.
- Serving solves can optionally write a stable placement-evidence CSV artifact,
  so selected and rejected prefill/decode placement decisions can be compared
  across heterogeneous candidates and scenario sweeps without parsing nested
  JSON.
- Serving solves can optionally write a stable worker/KV evidence CSV artifact,
  so request-level prefill/decode worker slots, KV-cache owners, and KV block
  ranges can be compared across heterogeneous candidates and scenario sweeps
  without parsing nested JSON.
- Serving solves can optionally write a stable rejections CSV artifact, so
  infeasible candidate causes can be compared across heterogeneous candidates
  and scenario sweeps without parsing nested JSON.
- Serving solves can optionally write a stable route-paths CSV artifact, so
  request-level KV handoff paths, source/destination GPUs, route resources,
  rail IDs, endpoints, bandwidth, latency, and estimated service time can be
  inspected across heterogeneous candidates and scenario sweeps without parsing
  nested JSON.
- Serving bottleneck CSV output now includes request-scoped rows for admission
  failures, timeout/cancellation outcomes, deadline misses, and
  TTFT/TPOT/ITL/E2EL SLO misses with request ID, tenant, model, and traffic
  class attribution.
- Calibration-profile JSON now includes per-benchmark latency residuals and a
  compact residual status, making measured-vs-predicted fit error auditable
  alongside the existing aggregate benchmark summary.
- Run output can optionally write a calibration residuals CSV artifact for the
  active profile in each scenario, preserving profile, benchmark shape, residual
  error, and provenance fields outside nested JSON.
- Generated serving pool search can filter prefill and decode candidate nodes
  by node tags, racks, islands, failure domains, and GPU tags, then require
  minimum rack/island/failure-domain spread for generated pools.
- Explicit named serving pool candidates can also filter their prefill and
  decode node groups by node tags, racks, islands, and failure domains before
  GPU-tag placement constraints are applied.
- Explicit named serving pool candidates can enforce minimum prefill/decode
  rack, island, and failure-domain spread after those filters resolve.
- Serving JSON and text notes now include compact pool-search summary evidence
  for generated pools, so heterogeneous pool filters and spread constraints are
  auditable during solver sweeps.
- Pool-search summary evidence also includes generated candidate counts by
  effective deployment mode, making it visible when flexible searches generated
  colocated, partially disaggregated, or fully disaggregated pool classes.
- A compact checked example,
  `examples/heterogeneous_disaggregated_workload.toml`, runs against
  `examples/heterogeneous_cluster.toml` and has CLI smoke coverage for
  heterogeneous fully-disaggregated TOML inputs, pool topology evidence, route
  evidence, and TTFT/TPOT/throughput/E2EL output.
- Compact checked examples,
  `examples/heterogeneous_colocated_workload.toml` and
  `examples/heterogeneous_partially_disaggregated_workload.toml`, run against
  `examples/heterogeneous_cluster.toml` and have CLI smoke coverage for
  colocated and partially-disaggregated deployment mode output plus
  TTFT/TPOT/throughput/E2EL metrics.
- A compact rail/island checked example,
  `examples/heterogeneous_rail_island_cluster.toml` and
  `examples/heterogeneous_rail_island_workload.toml`, covers a prefill island
  connected to two decode islands through different fabrics and rail layouts,
  with CLI smoke coverage for route-resource evidence, topology bottlenecks,
  and serving metrics.
- A compact oversubscribed checked example,
  `examples/heterogeneous_oversubscribed_cluster.toml` and
  `examples/heterogeneous_oversubscribed_workload.toml`, covers a prefill rack
  connected to a decode rack through an oversubscribed single-rail uplink, with
  CLI smoke coverage for route-resource queueing, topology bottlenecks, and
  serving metrics.
- A compact homogeneous checked example,
  `examples/h100_cluster.toml` and
  `examples/homogeneous_serving_workload.toml`, covers the colocated serving
  path on H100 nodes with local KV transfer, service evidence, and
  TTFT/TPOT/throughput/E2EL metrics.
- Selected pool topology evidence now distinguishes shared nodes from
  role-dedicated prefill and decode nodes, improving partial-disaggregation
  auditability without requiring a full physical placement model.
- Serving hardware footprint output now includes aggregate, prefill, and decode
  GPU-type counts, making role-specific heterogeneous placement choices
  inspectable in result JSON.
- Serving hardware footprint output also includes aggregate, prefill, and decode
  GPU-label counts, so service-role selections over locality tags such as
  socket or fast-NIC labels are visible after solving.
- Fixed-gap, seeded Poisson, deterministic bursty, seeded diurnal, seeded
  self-similar, and trace-derived traffic,
  independent request-shape distributions, weighted correlated synthetic shape
  profiles with tenant/model/cache/deadline/cancellation/SLO/timeout metadata,
  inline trace requests, CSV/JSONL trace imports, replay windows, and replay
  repeats.
- Configurable measurement start/end windows and warmup/cooldown trims for
  serving metrics, plus optional steady-state detection from stable completed
  request latency windows, so TTFT/TPOT/ITL/throughput/E2EL can exclude startup
  and drain regions without editing the trace.
- Selected steady-state windows now include multi-metric diagnostics for TTFT,
  TPOT, E2EL, queue phases, per-request throughput, aggregate output-token
  throughput, output-token counts, and worst-metric CV, so calibration reports
  can show when a latency-stable window is still unstable on another metric.
- Selected steady-state windows also include bucketed utilization diagnostics
  for scheduled compute/HBM/route resources, serving worker slots, and
  configured KV-residency capacity pressure, including mean/max utilization and
  utilization CV.
- Approximate independent and continuous prefill/decode batching, including
  chunked prefill for continuous prefill schedules.
- Routed prefill/decode worker readiness is now tracked during scheduling, with
  configurable per-GPU prefill and decode worker slots, so later requests are
  gated by selected worker/GPU availability in addition to trace dependencies
  and coarse resource reservations.
- Topology-aware request routing now estimates route-specific KV resource
  readiness from structured KV path resources, so a nominally faster
  prefill/decode route can be avoided when its modeled KV route segment is
  already queued.
- Prefill token pressure controls and reports at aggregate, per-node, and
  per-GPU levels, so coarse prefill pool capacity can reject candidates before
  production worker-local prefill admission is modeled.
- Configurable per-GPU prefill worker slot counts now gate worker-readiness
  queueing for independent and continuous prefill scheduling, preserving a
  simple MVP control for worker-local concurrency before a production prefill
  service loop exists.
- Configurable per-GPU decode worker slot counts now gate topology-aware
  routing estimates, request-level decode capacity admission, and independent
  or continuous decode worker-readiness queueing, preserving a simple MVP
  control for decode service concurrency before a production decode loop
  exists.
- Configurable per-GPU KV-transfer worker slot counts now optionally gate KV
  handoff scheduling, so copy-engine or handoff-worker pressure can be reported
  separately from modeled route/fabric contention.
- Configurable global and traffic-class KV-transfer queue-delay caps can reject
  requests before decode admission when KV handoff worker or route-resource
  backpressure exceeds policy.
- Configurable global and traffic-class decode queue-delay caps can reject
  requests before the first decode iteration reserves worker or compute
  resources when decode backpressure exceeds policy.
- Configurable global and traffic-class decode-iteration queue-delay caps can
  time out already-admitted requests when tail-token decode backpressure would
  violate TPOT/ITL-oriented policy.
- Reusable traffic classes for tenant/model/priority selectors with configurable
  admission priority, class-scoped active prefill token caps, class-scoped
  decode sequence/KV residency/KV block caps, SLO defaults, queue-delay caps,
  request timeouts, hard scoped miss-rate policies, and per-class capacity peak
  reporting.
- Aggregate and metric-specific soft SLO miss penalty weights for serving
  ranking, with JSON reporting of penalty weights, components, and final
  objective score.
- Traffic-class soft SLO miss penalty weights for tenant/model/priority
  classes, with JSON reporting of per-class penalty weights and component
  contributions to the serving objective.
- Configurable topology-risk penalty for serving ranking based on partial
  prefill/decode route coverage, so islanded or partially routable pools can be
  made less attractive without requiring exact physical contention yet.
- Request lifecycle outcomes for completed, admission-rejected, timed-out, and
  cancelled requests.
- Per-request structured rejection evidence for admission failures and
  timeouts, including phase, category, resource, code, observed value, limit,
  unit, remediation, and message for phase queue caps, backpressure timeouts,
  request timeouts, and request-level capacity admission paths.
- Per-request lifecycle event streams for arrival, queueing, prefill, KV
  transfer, decode iterations, and terminal outcomes, so the current scheduled
  timeline has a stable event-history shape for later online simulation.
- Decode capacity controls and reports at aggregate, per-node, and per-GPU
  levels, including active decode sequences, KV residency, KV block caps,
  block utilization, allocated KV tokens, and approximate fragmentation where
  currently modeled.
- Per-request approximate KV block ownership records for decode-owner GPUs,
  including allocation/release timestamps, resident tokens, allocated blocks,
  allocated tokens, and fragmentation. These records make decode-worker KV
  residency auditable before a production allocator is modeled.
- Metrics for TTFT, TPOT, ITL, E2EL, throughput, queue delay, KV-transfer time,
  request-aware SLO miss rates, deadline miss rates, scheduled makespan, decode
  iteration summaries, and prefix-cache effects.
- Optional hard serving metric ceilings for TTFT, TPOT, ITL, and E2EL, with
  TOML input, solver rejection evidence, and JSON output for the configured
  limits.
- Optional v1 KV-route topology constraints for minimum inter-node rail count,
  required rail metadata, and required GPUDirect-like GPU/NIC handoff paths,
  with TOML input, placement-aware cross-file validation, solver rejection
  evidence, and JSON output.
- Optional v1 serving cost and power estimates from configured GPU-hour,
  node-hour, GPU watt, node watt, and energy rates, with candidate-level JSON
  output and text summaries for capacity-planning comparisons.
- Per-request observations for routing, phase spans, phase contribution
  breakdowns, token timings, decode iterations, prefix-cache hits, and traffic
  classes. Phase contribution records split queue, service, and transfer time,
  including first-token decode versus decode tail, so TTFT and E2EL composition
  can be audited per request.
- Per-request KV handoff diagnostics and costs for prefill route GPUs,
  decode/owner GPUs, transfer bytes, bottleneck resources, and selected
  GPU-to-GPU topology paths, including same-node different-GPU handoffs.
  JSON keeps the legacy path labels and now also emits structured route
  segments with resource kind, endpoint IDs, rail IDs, bandwidth, and latency.
- Candidate-level KV route resource summaries aggregate those per-request
  segments by stable resource ID, kind, endpoints, rail, bytes, path
  observations, and estimated transfer time, giving the MVP a first
  resource-keyed route bottleneck artifact.
- Candidate-level topology bottleneck observations now report route coverage,
  single-rail dependency, missing rail metadata, host-staged/non-GPUDirect KV
  paths, cross-socket GPU/NIC paths, and slow GPU-to-NIC KV path segments.
- KV handoff scheduling now reserves those structured route resource IDs
  instead of only a coarse global fabric/NIC bottleneck string, so unrelated
  disaggregated handoffs can overlap while handoffs sharing a modeled route
  segment serialize in the approximate timeline.
- Per-request and aggregate output now split KV route-resource queue time from
  KV transfer service time and report the concrete KV route resources plus
  scheduler resource-dependency operation IDs that caused route contention.
- Candidate-level topology bottleneck observations now also promote dynamic KV
  route-resource queueing and hot route-resource utilization into structured
  `contention` observations, making route-sharing congestion visible without
  manually inspecting utilization tables.
- Per-request and aggregate output now also split KV-transfer worker queue time
  from KV route-resource queue time, preserving a clear distinction between
  copy-worker pressure and shared route/fabric pressure.
- KV handoff admission can now preview the scheduler timeline before reserving
  route resources, so queue-delay rejections do not consume decode capacity or
  create synthetic KV-transfer reservations.
- First-token decode admission can now preview the decode scheduler timeline
  before reserving decode resources, so decode queue-delay rejections do not
  create synthetic decode operations.
- Tail-token decode scheduling now checks projected iteration queue delay
  before reserving another decode operation, so partial streams can time out
  with auditable queue evidence instead of only reporting bad TPOT/ITL after
  completion.
- Per-request routing evidence now includes the router's estimated KV
  route-resource wait time, making it visible when route contention affected a
  topology-aware routing choice before the scheduler built the final timeline.
- Per-request JSON now includes routing-candidate observations with selected
  and routable flags, routed prefill/decode nodes and GPUs, estimated route
  waits, KV transfer bytes, KV bottlenecks, route resource IDs, and rejection
  reasons for unroutable candidates.
- Pool-level prefill/decode route-coverage reporting for serving candidates,
  including total route candidates, routable route candidates, and routable
  fraction. This makes partially connected heterogeneous or islanded pools
  visible even when topology-aware routing finds a working route.
- Per-GPU serving worker summaries for approximate prefill, KV-transfer, and
  decode workers, including request counts, worker/resource queue timing,
  service timing, token counts where applicable, prefill-token peaks, and decode
  sequence, residency, KV-block, allocated-token, and fragmentation peaks, plus
  configured worker slots, approximate peak active worker slots, and
  worker-slot utilization.
- JSON output for candidates, placements, scheduled operations, utilization,
  occupancy buckets, critical paths, request observations, route observations,
  decode iterations, metric breakdowns, memory estimates, cluster inventory,
  node groups, mixed GPU inventories, NICs, rail-scoped links, approximation
  records, and structured rejections.
- Text output includes a concise cluster inventory preamble with node groups,
  GPU-type counts, mixed-GPU node summaries, NIC/rail details, intra-node
  fabric, and custom inter-node links.
- Structured topology diagnostics in text and JSON for custom topology
  disconnected islands, node groups that span disconnected islands,
  rail-scoped custom links that reference rails unavailable on either endpoint,
  and configured GPU-to-NIC locality risks such as disabled endpoints,
  unavailable paths, host-staged/non-GPUDirect paths, and slow local NIC
  segments.
- Approximation policies that can warn or reject candidates by category/code.
- Calibration knobs, reusable calibration profiles, raw benchmark metadata,
  fitted linear latency models, benchmark error summaries, valid/invalid shape
  checks, benchmark coverage scoring, holdout-validation and confidence-
  interval fit error metadata, calibration gates for weak or unauditable
  profiles/fits, active queue-component calibration coverage, and
  aggregate TTFT/TPOT/throughput/E2EL serving-metric fits that can influence
  ranking, plus uncertainty-adjusted ranking.

Current limits:

- Heterogeneous clusters are supported at the node-group, node, per-node GPU
  inventory, explicit GPU-to-NIC map, NIC-count, rail-count, explicit
  NIC-to-rail map, GPU-to-NIC path override, node-tag/rack/island/
  failure-domain selector, and custom-link level, plus basic disabled GPU/NIC
  state, scenario-level whole-node disabled/maintenance/draining/reserved
  state, and scenario-level per-GPU, per-NIC, rail, and custom-link
  degradation. Fine-grained partial reservation, rolling maintenance behavior,
  PCIe, NUMA, NVSwitch, switch, and fabric-resource behavior are not yet
  modeled.
- Different interconnects between sets of nodes can be expressed with custom
  links today, including rail-scoped links for the same node pair or node
  groups and explicit local-GPU endpoint-scoped links. Realistic contention
  across shared rails, switches, oversubscribed uplinks, copy engines, and route
  hashing is still missing.
- Custom links are still logical shortcuts over node, node-group, rail, and
  local-GPU endpoint selectors. The schema cannot yet express arbitrary
  NIC-pair, PCIe-domain, NVSwitch-domain, rack-switch, or fabric-switch paths
  as first-class physical resources with their own sharing and contention.
- Route coverage and topology-risk penalties are candidate-level planning
  signals. They do not yet prove that every selected rank, worker, NIC, rail,
  PCIe path, or switch path has realistic capacity under concurrent traffic.
- Prefill/decode disaggregation is supported as an approximate solver dimension,
  including aggregate/per-node/per-GPU decode capacity limits and per-request
  decode-owner KV block ownership records, but prefill and decode are not yet
  independent online services with exact queue event loops, queue data
  structures, allocator behavior, backpressure, admission, and scaling.
- TTFT, TPOT, ITL, throughput, and E2EL come from one scheduled request
  timeline, but absolute values are only as credible as the calibration profile
  and the currently modeled topology/resource detail.
- Active queue components are now visible in calibration evidence and
  approximation policy, but queue-specific fitted correction models are not yet
  applied back into the scheduling timeline.
- Aggregate TTFT, TPOT, throughput, and E2EL fits adjust reported summary metrics
  and solver ranking, but request observations, traffic-class breakdowns, and
  scheduled operation timelines remain the uncorrected simulator timeline until
  request-level calibration/replay is implemented.

Review checkpoints:

- **What is missing for realistic simulation:** online serving state,
  worker-local queues, worker-local KV ownership, topology-aware placement,
  physical PCIe/NUMA/NVSwitch/NIC/rail/switch paths, shared-resource
  contention, layer-aware compute, exact memory timelines, production workload
  traces, benchmark-backed calibration, uncertainty accounting, and richer
  bottleneck attribution.
- **Serving-state gap:** request arrivals, admission, queueing, batching,
  prefill, KV handoff, decode iterations, cancellation, timeout, preemption,
  eviction, migration, backpressure, and completion should be driven by one
  authoritative event loop. The current lifecycle events are an output artifact,
  not yet the source of scheduling truth.
- **Capacity gap:** decode has aggregate/per-node/per-GPU sequence and
  KV-residency controls, and prefill has aggregate/per-node/per-GPU active-token
  caps and pressure reports plus approximate per-GPU prefill worker slot
  controls. Decode and KV handoff now also have approximate per-GPU worker slot
  controls, and KV handoff plus first-token decode have queue-delay admission
  caps. Tail-token decode has a queue-delay timeout cap. Realistic admission
  still needs exact worker-local queue state, KV-block allocation,
  request-level backpressure across all phases, class-aware admission, and
  policy evidence tied to the event loop.
- **Topology gap:** current custom links, GPU-to-NIC locality maps, NIC rails,
  and routed KV resources are useful MVP hooks. They are not yet a physical
  topology model with CPU sockets, NUMA domains, PCIe switches, NVSwitch
  domains, copy engines, NIC queues, rails, rack switches, fabric switches,
  oversubscribed uplinks, adaptive routing, or route hashing.
- **Contention gap:** current route-resource scheduling and optional
  KV-transfer worker slots are a first pass for KV handoff resources.
  Production realism still needs occupancy timelines and sharing policy for GPU
  compute, HBM bandwidth, exact copy engines, PCIe, NVLink/NVSwitch, NICs,
  rails, rack/fabric switches, collectives, activation sends, KV transfers,
  storage traffic, and control-plane traffic.
- **Placement/routing gap:** rank placement and pool routing are currently
  bounded, deterministic, and capability/topology aware only at a coarse level.
  Future versions need placement and online routing over physical topology
  domains, worker load, cache affinity, tenant/model constraints, failure
  domains, route contention, and calibration uncertainty.
- **Compute/memory gap:** prefill and decode are still coarse calibrated phase
  estimates with component memory checks. Realism needs layer/backend-aware
  attention, MLP, MoE, logits, sampling, KV-kernel, fused-kernel, launch,
  CUDA-graph, paged-attention, and runtime effects, plus time-aware per-worker
  and per-GPU accounting for weights, KV cache, activations, temporary buffers,
  communication buffers, allocator reserve, graph-capture reserve, and
  fragmentation.
- **Workload gap:** fixed-gap, Poisson, synthetic distributions, inline traces,
  and CSV/JSONL replay cover MVP exploration. Production realism still needs
  bursty/diurnal/self-similar arrivals, correlated prompt/decode sizes,
  multi-model mixes, cache keys, retries, streaming disconnects, cancellations,
  timeouts, deadline policies, and tenant/model routing constraints.
- **Calibration gap:** reusable calibration profiles exist, but realistic
  recommendations need measured fixtures for compute, collectives, RDMA,
  PCIe/NVLink copies, KV transfer, queueing, and end-to-end serving, with
  provenance, coverage scoring, confidence intervals, holdout validation, and
  explicit interpolation/extrapolation policy.
- **Whether more heterogeneous configs are possible:** yes for the MVP through
  node groups, node IDs, explicit local GPU inventories, explicit GPU-to-NIC
  locality maps, explicit NIC-to-rail maps, GPU-to-NIC path overrides,
  NIC/rail counts, and custom links between node pairs or node groups,
  including links scoped to specific rails or local GPU endpoint subsets, with
  disabled/degraded GPU, NIC, and custom-link overlays. Realism still needs
  endpoint selectors for arbitrary GPU/NIC/rack/fabric/failure-domain groups,
  PCIe, NUMA, NVSwitch, switch, rack, and fabric resources, asymmetric/missing
  links, reserved capacity, and contention on those resources.
- **Whether prefill/decode disaggregation is handled:** yes approximately
  through separate pools, search spaces, routing, KV handoff, and shared
  TTFT/TPOT/ITL/E2EL reporting. Realistic disaggregation still needs
  independent online prefill/decode services, worker queues, worker capacity,
  production cache ownership and allocation behavior, request-level admission,
  decode-to-prefill backpressure, KV-transfer backpressure, scaling, exact KV
  residency, and service-specific health/failure state.

## Canonical Missing Pieces

This is the concise answer to the current design review. These are the pieces
that are still missing or intentionally approximate, even when the MVP has a
schema hook or coarse metric for them.

- **Authoritative online simulation state:** request arrivals, admission,
  worker queues, batching windows, active decode sets, cancellation, timeout,
  preemption, completion, and backpressure need to be driven by one event loop.
- **Independent prefill/decode services:** prefill and decode should have
  separate service health, worker queues, admission policy, scaling, routing,
  worker capacity, and backpressure instead of sharing one approximate
  scheduled timeline.
- **Exact KV ownership and allocation:** KV source, destination, owner,
  block-table state, allocator reserve, fragmentation, eviction, migration,
  spill, reuse, and prefix-cache residency need worker/GPU-level state.
- **Prefill-side capacity and admission:** decode has aggregate/per-node/per-GPU
  sequence and KV-residency caps, while prefill now has aggregate/per-node/
  per-GPU token caps, observed peaks, and configurable approximate per-GPU
  worker slots. Prefill still needs exact worker-local admission, queue state,
  and backpressure.
- **Decode-side worker concurrency:** decode has aggregate/per-node/per-GPU
  sequence and KV-residency caps plus configurable approximate per-GPU worker
  slots plus an optional first-token queue-delay rejection policy. Decode still
  has an optional tail-token queue-timeout policy, but still needs exact service
  queues, per-worker active sequence ownership, preemption, and
  allocator-aware admission.
- **KV-transfer worker concurrency:** KV handoff has optional approximate
  per-GPU worker slots, separate worker-queue reporting, and an optional
  queue-delay rejection policy. It still lacks exact copy-engine inventories,
  PCIe/NVLink/RDMA copy paths, copy chunking, duplex behavior, and contention
  with other traffic classes.
- **Physical topology resources:** CPU sockets, NUMA domains, PCIe switches,
  NVSwitch domains, GPUs, NICs, rails, rack switches, fabric switches, copy
  engines, storage/control fabrics, and failure domains need stable resource
  IDs and path records.
- **Physical heterogeneous interconnects:** current custom links are logical
  shortcuts. Realistic heterogeneous clusters need endpoint selectors and
  physical links for arbitrary GPU groups, NIC subsets, racks, fabrics, rails,
  missing links, asymmetric links, degraded links, and reserved capacity.
- **Shared-resource contention:** HBM, compute, copy engines, PCIe,
  NVLink/NVSwitch, NICs, rails, fabric links, switches, collectives,
  KV transfers, activation sends, storage/control traffic, and CPU overheads
  need occupancy timelines and contention policy.
- **Topology-aware placement and routing:** rank placement, pipeline cuts,
  tensor/expert/sequence/context groups, prefill workers, decode workers,
  cache owners, tenants, and models need routing/placement search over physical
  topology domains, failure domains, worker load, cache affinity, and
  calibration uncertainty. The MVP has only first-pass load-aware routing for
  routed workers and modeled KV route resources.
- **Communication realism:** collective algorithms, GPUDirect/RDMA behavior,
  PCIe/NVLink copies, NCCL/RCCL/UCX assumptions, protocol thresholds, message
  chunking, startup overhead, and route sharing need calibrated models.
- **Layer/backend-aware compute:** prefill and decode costs should be split by
  attention, MLP, MoE, logits, sampling, KV kernels, fused kernels, launch
  overhead, CUDA graph behavior, runtime stack, dtype, and GPU generation.
- **Time-aware memory accounting:** weights, KV cache, activations, temporary
  buffers, communication buffers, page tables, runtime reserve, graph-capture
  reserve, allocator reserve, and fragmentation need per-worker/per-GPU
  timelines and OOM evidence.
- **Production workload behavior:** richer traces, bursty/diurnal/self-similar
  arrivals, correlated prompt/decode sizes, multi-model mixes, cache keys,
  tenant routing constraints, retries, streaming disconnects, cancellations,
  timeouts, and deadline policies are still incomplete.
- **Calibration and uncertainty:** compute, collectives, RDMA, PCIe/NVLink
  copies, KV transfer, queueing, and end-to-end serving need benchmark-backed
  fits with provenance, coverage, confidence intervals, holdout validation, and
  explicit interpolation/extrapolation policy.
- **Bottleneck attribution and reporting:** output needs drilldowns by pool,
  worker, node, GPU, NIC, rail, link, switch, copy engine, tenant, model,
  phase, request, objective term, and physical resource.
- **Operational scenarios:** node/GPU/NIC/link/rail/switch/rack/fabric
  failures, degradation, draining, maintenance, reserved capacity, rolling
  upgrades, rack/power/thermal domains, and failover capacity need richer
  scenario overlays.
- **Validation and fixtures:** cross-file validation, invalid-config tests,
  golden fixtures, CLI smoke tests, deterministic JSON outputs, and calibrated
  regression tests need to cover homogeneous, heterogeneous, islanded,
  rail-aware, oversubscribed, colocated, partially disaggregated, and fully
  disaggregated setups.
- **Solver controls and objectives:** runtime budgets, pruning, checkpointing,
  Pareto frontiers, sensitivity analysis, cost, power, topology risk,
  failure-domain risk, memory headroom, and grouped rejection root causes are
  still missing or partial.

## Concrete Goal Backlog

Unchecked items are intentionally missing or approximate today. They are kept
as explicit goals so MVP shortcuts remain visible and future topology,
disaggregation, and calibration work has stable targets.

### Simulation Core

- [ ] Make request lifecycle events the authoritative scheduler state, not only
  derived output.
- [ ] Build a true online event loop with arrivals, admission, worker queues,
  batching windows, active decode sets, timeout, cancellation, preemption,
  backpressure, and completion.
- [ ] Add worker-local state for prefill workers, decode workers, KV-transfer
  workers, cache owners, scheduler shards, and admission queues.
- [ ] Model worker readiness, queue delay, resource delay, service time, and
  backpressure as separate state transitions.
- [x] Attach request-level structured rejection evidence to the current
  prefill, KV-transfer, first-token decode, tail-token decode, request-timeout,
  and capacity-admission paths that can reject or time out requests.
- [ ] Add occupancy timelines for compute, HBM, memory pressure, copy engines,
  PCIe, NVLink/NVSwitch, NICs, rails, fabric links, switches, and CPU-side
  overheads.
- [x] Add configurable measurement start/end windows, warmup/cooldown trims, and
  optional steady-state auto-detection for trace and synthetic serving metrics.
- [x] Add richer steady-state convergence diagnostics and approximation
  evidence for selected TTFT/TPOT/E2EL/queue/throughput windows.
- [x] Add steady-state utilization convergence diagnostics for worker,
  route-resource, compute, HBM, and KV-residency pressure.

### Prefill/Decode Disaggregation

- [x] Express colocated, partially disaggregated, and fully disaggregated
  serving modes in config, pool validation, pool search, solver filtering, and
  JSON/text reporting.
- [ ] Back those modes with one production-like service/pool/worker model
  instead of the current approximate shared timeline.
- [ ] Model prefill and decode as independent online services with separate
  health, scaling, admission, routing, priority, and queue state.
- [ ] Add decode-to-prefill and KV-transfer-to-prefill backpressure.
- [ ] Track exact prefill source worker/GPU, decode owner worker/GPU,
  cache-owner worker/GPU, and KV-transfer path per request.
- [ ] Convert aggregate/per-node/per-GPU decode residency into exact
  per-worker KV block ownership.
- [ ] Model KV allocation, block tables, fragmentation, eviction, migration,
  spill, reuse, and prefix-cache residency.
- [ ] Model KV-transfer contention with collectives, activation sends,
  PCIe/NVLink copies, RDMA, storage/control traffic, and other point-to-point
  transfers.
- [x] Add an optional approximate per-GPU KV-transfer worker slot control and
  split KV worker queue time from KV route-resource queue time.
- [x] Add a KV-transfer queue-delay cap that rejects requests before decode
  admission when handoff backpressure exceeds policy.
- [x] Add a first-token decode queue-delay cap that rejects requests before
  decode resources are reserved when decode backpressure exceeds policy.
- [x] Add a tail-token decode iteration queue-delay cap that times out partial
  streams before reserving another decode operation when TPOT/ITL-oriented
  backpressure exceeds policy.
- [ ] Derive TTFT, TPOT, ITL, E2EL, throughput, SLO misses, and deadline misses
  from the same event history.

### Heterogeneous Cluster Configs

- [x] Express logical custom inter-node links between node pairs, node groups,
  rails, and explicit local GPU endpoint subsets.
- [ ] Add endpoint selectors for node groups, explicit nodes, GPUs, GPU labels,
  NICs, rails, racks, islands, failure domains, fabrics, and service pools.
- [ ] Let interconnect resources connect arbitrary endpoint selectors instead
  of only the currently supported node pairs, node groups, rails, and local GPU
  endpoint subsets.
- [ ] Turn logical GPU-scoped custom links into physical path resources that can
  model NIC subsets, PCIe domains, NVSwitch domains, racks, islands, rails, and
  fabrics, including asymmetric, missing, degraded, and reserved links.
- [ ] Add per-GPU labels, health state, maintenance state, dtype support, HBM
  size, HBM bandwidth, FLOPs, NVLink generation, and vendor/runtime
  compatibility.
- [x] Add optional per-NIC rail membership so configs are not forced into
  `rail = nic_id % rail_count`.
- [x] Add optional per-GPU/per-NIC path overrides for bandwidth, latency,
  GPUDirect support, and unavailable host paths.
- [x] Add per-NIC scenario latency degradation overlays so degraded NICs can
  affect routed GPU-to-NIC and inter-node fabric timing, not only bandwidth.
- [x] Add rail-level scenario bandwidth/latency degradation overlays for
  matching NIC rails and rail-scoped custom inter-node links.
- [x] Add node topology selectors to run-scenario whole-node, disabled GPU/NIC,
  per-GPU/per-NIC, and rail degradation overlays, so scenario sweeps can target
  node tags, racks, islands, and failure domains without enumerating node IDs.
- [x] Add per-GPU and per-NIC operational-state config for healthy, disabled,
  maintenance, draining, and reserved resources, treating non-healthy resources
  as unavailable while preserving state in cluster inventory evidence.
- [x] Add base-cluster per-NIC bandwidth and latency override config, wired into
  existing route selection/costing and cluster inventory output.
- [x] Add base-cluster per-GPU profile override config, wired into existing
  placement, memory, throughput, dtype-capability validation, and cluster
  inventory output.
- [x] Add base-cluster GPU/NIC NUMA-domain maps and cross-domain GPU-to-NIC
  bandwidth/latency scale config, wired into default GPU-to-NIC route
  selection/costing and cluster inventory output.
- [ ] Add per-NIC health and degradation state beyond current disabled and
  bandwidth/latency degradation controls.
- [ ] Extend scenario overlays beyond disabled GPU/NIC resources, whole-node
  disabled/maintenance/draining/reserved state, and per-GPU, per-NIC, rail,
  and custom-link degradation to switch, rack, fabric, partial reserved
  capacity, draining policy, and rolling maintenance behavior without
  duplicating the base cluster file.
- [ ] Add topology templates/generators for common layouts: single-node
  NVSwitch, rail-aligned InfiniBand, fat tree, dragonfly, rack islands,
  oversubscribed racks, and disconnected clusters.
- [ ] Keep simple TOML node-group configs working while advanced physical
  topology sections remain optional.

### Physical Topology

- [ ] Represent CPU sockets, NUMA domains, PCIe switches, NVSwitch domains,
  GPUs, NICs, rails, rack switches, fabric switches, copy engines, and failure
  domains as addressable resources.
- [ ] Model paths such as `GPU -> NVSwitch/PCIe -> NIC -> rail -> fabric ->
  NIC -> PCIe/NVLink -> GPU`.
- [ ] Distinguish direct, same-PCIe-switch, same-NUMA, cross-socket,
  host-staged, GPUDirect, degraded, and unavailable GPU-to-NIC paths.
- [ ] Model bandwidth, latency, duplex mode, startup overhead, protocol
  overhead, sharing scope, failure domain, and degradation state per resource.
- [ ] Model rail pinning, rail striping, adaptive routing, ECMP/hash behavior,
  shared uplinks, oversubscription, incast/outcast, and head-of-line blocking
  assumptions.
- [ ] Track route sharing across NICs, rails, PCIe links, NVLink/NVSwitch
  links, rack switches, fabric switches, and copy engines.
- [ ] Report topology bottlenecks such as cross-socket paths, unavailable
  GPUDirect, overloaded rails, slow host-staged copies, and oversubscribed
  uplinks.

### Placement And Routing

- [ ] Search placements across GPUs, nodes, racks, islands, rails, NUMA
  domains, PCIe domains, NVSwitch domains, and failure domains.
- [ ] Keep high-traffic tensor, expert, sequence, context, prefill, decode, and
  cache-owner groups inside fast topology domains when possible.
- [x] Support user-supplied explicit rank placements from config files for the
  base solver and serving prefill/decode pools.
- [ ] Add user-supplied placement constraints for pools, workers, tenants,
  models, service pools, failure domains, topology domains, and partial rank
  groups.
- [ ] Score placements by route cost, HBM headroom, GPU capability, queueing
  pressure, failure-domain spread, topology risk, cost, power, and calibration
  uncertainty.
- [x] Score serving candidates with topology-risk penalties for partial route
  coverage, single-domain placement, single-rail dependency, cross-socket and
  host-staged GPU/NIC paths, slow local NIC paths, and hot/queued KV route
  resources.
- [ ] Replace coarse pool routing with online routing based on worker load,
  queue delay, cache affinity, topology path cost, route contention,
  tenant/model constraints, and outages.
- [x] Account for routed prefill/decode worker readiness and modeled
  KV route-resource readiness in topology-aware request routing.
- [x] Explain selected and rejected route candidates with structured
  per-request evidence.
- [x] Explain selected and rejected rank placements with structured
  candidate-level evidence.
- [ ] Extend placement evidence from rank-placement summaries to physical
  topology-domain, rack, rail, NUMA, PCIe, NVSwitch, tenant, and service-pool
  constraints.

### Compute, Memory, And Communication

- [ ] Replace coarse FLOP estimates with layer-aware prefill and decode models.
- [ ] Split cost into attention, MLP, MoE routing, logits, sampling,
  normalization, KV kernels, fused kernels, launch overhead, and backend
  effects.
- [ ] Add model-family support for GQA/MQA, KV dtype, quantized KV, quantized
  weights, mixed precision, sliding-window attention, paged attention, MoE,
  LoRA, speculative decoding, and cache misses.
- [ ] Replace coarse component memory estimates with placement- and time-aware
  accounting for weights, KV cache, activations, temporary buffers,
  communication buffers, runtime reserve, graph capture reserve, allocator
  reserve, page tables, and fragmentation.
- [ ] Model collective algorithms explicitly: ring, tree, double binary tree,
  hierarchical, CollNet, SHARP, NVLS, and vendor-specific auto-selection.
- [ ] Add message-size thresholds, protocol choices, channels, chunking,
  startup costs, and per-algorithm traffic formulas.
- [ ] Distinguish tensor, expert, sequence, context, pipeline, KV-transfer,
  activation, serving-control, and storage traffic.

### Workloads And Policy

- [ ] Add larger production-like trace fixtures with replay transforms,
  warmup trimming, downsampling, and repeat windows.
- [ ] Model bursty, diurnal, self-similar, and trace-derived arrivals.
- [ ] Add correlated prompt/decode sizes, multi-model mixes, tenant/model
  routing constraints, cache keys, retries, streaming disconnects,
  cancellations, deadlines, and timeout policies.
- [ ] Extend traffic classes beyond current SLO defaults, queue caps, timeouts,
  hard miss limits, soft penalties, and admission priority with class-scoped
  capacity policies, burst limits, tenant isolation, reserved capacity,
  cancellation behavior, class interaction rules, and multi-objective weights.
- [ ] Add hard and soft constraints for SLOs, memory headroom, topology
  domains, failure domains, tenant isolation, model placement, cost, power, and
  operational risk.

### Calibration, Validation, And Reporting

- [ ] Import measured benchmark fixtures for compute, collectives, RDMA,
  PCIe/NVLink copies, KV transfer, queueing, and end-to-end serving.
- [ ] Fit profiles by hardware, fabric, model family, dtype, backend, and
  serving stack without simulator code changes.
- [ ] Add confidence intervals, holdout validation, interpolation/extrapolation
  policy, rank sensitivity, and uncertainty propagation.
- [ ] Validate solver rankings against known deployment choices and measured
  serving outcomes.
- [ ] Add bottleneck attribution by pool, worker, node, GPU, NIC, rail, link,
  switch, tenant, model, phase, request, and objective term.
- [x] Add stable CSV/JSON artifacts for metrics, utilization, timelines,
  occupancy, memory pressure, request lifecycle, calibration, route paths, and
  scenario sensitivity.
- [x] Make text and JSON output clearly label coarse topology, approximate
  queueing, aggregate memory, uncalibrated fits, extrapolation, ignored
  contention, and unsupported locality assumptions.

### Reliability And Testing

- [ ] Model failure domains across racks, rows, islands, availability zones,
  maintenance pools, power domains, and regions.
- [ ] Extend scenario inputs beyond disabled GPU/NIC overlays and per-GPU,
  per-NIC, and custom-link degradation to failures/degradation for nodes,
  rails, switches, racks, fabrics, reserved capacity, draining state, and
  maintenance state.
- [ ] Model reserved capacity, draining nodes, rolling upgrades, anti-affinity,
  failover capacity, replica spread, power, thermal, and rack-density
  constraints.
- [x] Add checked example fixtures and CLI smoke coverage for homogeneous,
  heterogeneous, islanded, rail-aware, oversubscribed, colocated, partially
  disaggregated, and fully disaggregated setups.
- [x] Add invalid-config tests for duplicate IDs, disconnected pools,
  impossible ranks, invalid rails, invalid affinity, unsupported dtypes,
  incompatible hardware, missing links, and invalid disaggregation routes.
- [x] Add invalid-config tests for duplicate serving trace request IDs across
  inline TOML requests and CSV trace imports.
- [x] Add invalid-config tests for duplicate serving traffic-class selectors.
- [ ] Add deterministic simulation tests for arrivals, trace replay, batching,
  routing, queueing, SLO misses, admission rejection, timeouts, cancellation,
  lifecycle events, and rejection evidence.

## Missing Pieces Matrix

This is the short list of gaps to keep visible while cutting MVP scope. The
MVP should either model each item coarsely with explicit approximation evidence
or leave stable IDs/schema hooks so a later version can replace the estimate
with a physical model.

| Area | Missing piece | MVP posture | Future-ready hook |
| --- | --- | --- | --- |
| Online serving | Independent event loop for request arrivals, worker queues, batching windows, active decode sets, timeout, cancellation, preemption, and backpressure. | Keep one request timeline with routed worker readiness and expose approximation records. | Stable request IDs, phase spans, worker IDs, and queue-state fields. |
| Prefill/decode disaggregation | Prefill and decode are not yet independent services with their own health, scaling, admission, and routing state. | Search separate pools, support explicit colocated/partial/full deployment modes, and report KV handoff costs. | Shared service/worker abstractions for colocated, partially disaggregated, and fully disaggregated modes. |
| KV cache | Exact KV block allocation, ownership, fragmentation, eviction, migration, spill, reuse, and page-table overhead. | Keep aggregate/per-node/per-GPU decode sequence, residency, KV-block, and approximate fragmentation caps/reports. | Per-worker/GPU cache owner IDs, block-size config, allocator state, and per-request cache lifecycle events. |
| Topology graph | Physical GPU-to-NIC paths through NUMA, PCIe, NVSwitch, copy engines, NICs, rails, rack switches, and fabric switches. | Keep node groups, mixed-GPU inventories, explicit GPU-to-NIC locality, explicit GPU-to-NIC path overrides, explicit NIC-to-rail maps, NIC/rail counts, custom links, and structured KV route segment records. | Addressable topology-domain/resource IDs and resource-keyed route path records. |
| Interconnect heterogeneity | Different fabrics between GPU groups, GPU pairs, racks, islands, and rails, including asymmetric, degraded, or missing links. | Use node-pair/node-group custom links, including rail-scoped and local-GPU endpoint-scoped links. | Link resources with endpoint selectors, health/degradation state, and sharing scope. |
| Config expressiveness | A schema for arbitrary topology-domain selectors, scenario overlays, templates, and generated large clusters. | Keep TOML simple: node groups, explicit node overrides, mixed GPU inventories, disabled GPUs/NICs, GPU-to-NIC maps, and custom links. | Endpoint selectors for nodes, GPUs, NICs, rails, racks, fabrics, labels, failure domains, and scenario mutations. |
| Shared contention | Contention across HBM, compute, copy engines, PCIe, NVLink/NVSwitch, NICs, rails, fabric links, collectives, KV transfers, and activation sends. | Use coarse bottleneck estimates and utilization summaries; KV transfers reserve structured route resource IDs where paths are modeled. | Resource-occupancy timelines keyed by physical resource IDs. |
| Placement | Topology-aware rank, pipeline, expert, prefill-worker, decode-worker, cache-owner, tenant, and model placement search. | Deterministic capability-aware placement with bounded candidate search plus explicit rank placements for base and serving prefill/decode configs. | User-supplied topology-domain/service constraints and structured selected/rejected placement evidence. |
| Routing | Online routing based on worker load, queue delay, cache affinity, topology path cost, route contention, tenant/model constraints, and outages. | Approximate round-robin or topology-aware pool routing, plus route coverage, optional topology-risk scoring, routed worker readiness, and modeled KV route-resource readiness. | Route candidates, route coverage, route rejection evidence, route-resource IDs, and per-request selected paths. |
| Compute | Layer/backend-aware prefill/decode costs for attention, MLP, MoE, logits, sampling, KV kernels, fused kernels, CUDA graphs, and launch overhead. | Calibrated coarse phase estimates. | Backend/model-family fit metadata and phase-level calibration targets. |
| Communication | Explicit collective algorithms, point-to-point copies, RDMA, GPUDirect, NCCL/RCCL/UCX behavior, protocol thresholds, and message chunking. | Coarse collective and KV-transfer estimates. | Operation type, route path, fabric/backend assumption, and calibration provenance fields. |
| Memory | Time-aware per-worker/per-GPU accounting for weights, activations, KV cache, temporary buffers, communication buffers, allocator reserve, and graph-capture reserve. | Component memory estimates and headroom checks. | Per-component memory timeline schema and OOM rejection evidence. |
| Workloads | Bursts, diurnal/self-similar arrivals, correlated prompt/decode sizes, tenant/model mixes, retries, streaming disconnects, cancellations, deadlines, and trace lifecycle metadata. | Fixed-gap, Poisson, synthetic distributions, inline traces, and CSV/JSONL imports. | Trace schema extensions for classes, cache keys, routing constraints, lifecycle, and timeout policy. |
| Calibration | Benchmark-backed fits for compute, collectives, RDMA, PCIe/NVLink copies, KV transfer, queueing, and end-to-end serving. | Reusable calibration profiles and gates. | Per-fit provenance, coverage, uncertainty, interpolation/extrapolation, and holdout validation. |
| Metrics | Bottleneck attribution and drilldowns by pool, worker, node, GPU, NIC, rail, link, switch, tenant, model, phase, and request. | Aggregate metrics plus request, route, worker, capacity observations, and KV route resource summaries. | Stable JSON artifacts and resource-keyed metric records. |
| Policy | Admission priority, tenant isolation, class interaction, reserved capacity, hard/soft constraints, objective weights, cost, power, and failure-domain policies. | Basic queue caps, timeouts, SLO defaults, and soft SLO penalties. | Policy IDs, class selectors, objective components, and constraint evidence. |
| Serving stack | Runtime-specific behavior for vLLM, TensorRT-LLM, SGLang, Dynamo, Ray Serve, Triton, NCCL/RCCL/UCX, CUDA graphs, paged attention, and prefix cache. | Keep stack assumptions explicit in approximation and calibration records. | Stack/backend IDs attached to compute, memory, communication, scheduler, and calibration formulas. |
| Operations | Failures, degraded resources, maintenance/draining state, failover capacity, rolling upgrades, rack/power/thermal domains, and operational risk reporting. | Disabled GPU/NIC state, whole-node disabled/maintenance/draining/reserved scenario overlays, per-GPU/per-NIC/rail/custom-link degradation overlays, and topology diagnostics. | Richer resource health state, failure domains, and scenario mutations. |
| Testing | Golden fixtures for heterogeneous, rail-aware, islanded, oversubscribed, colocated, and disaggregated setups plus invalid-config and calibration regression tests. | Focused unit and CLI smoke tests. | Stable fixture configs and deterministic JSON outputs. |

## Question-Driven Gap Backlog

These are the concrete missing pieces from the current design review. They are
written as implementation targets, not as claims about what the MVP can already
model exactly.

### Realistic Simulation

- **Event semantics.** Define a single authoritative event history for request
  arrival, admission, queueing, prefill, KV handoff, decode iterations,
  cancellation, timeout, migration, eviction, and completion. The MVP now emits
  a derived per-request lifecycle event stream from the scheduled timeline;
  remaining work is making that event stream the authoritative source for all
  scheduling and metrics.
- **Online serving simulation.** Replace the coarse scheduled timeline with an
  event loop that has request arrivals, worker-local queues, batching windows,
  active decode sets, admission, cancellation, timeout, preemption, and
  backpressure.
- **Worker and cache state.** Track prefill workers, decode workers, active
  sequences, KV-cache ownership, KV block allocation, fragmentation, eviction,
  migration, spill, and reuse.
- **Physical topology graph.** Represent GPUs, CPU sockets, NUMA domains, PCIe
  switches, NVSwitch domains, NICs, rails, rack switches, fabric switches,
  copy engines, and failure domains as addressable resources.
- **Contention on shared resources.** Account for multiple operations sharing
  HBM bandwidth, copy engines, PCIe paths, NVLink/NVSwitch paths, NICs, rails,
  rack/fabric switches, and oversubscribed uplinks.
- **Topology-aware placement and routing.** Search placements and request routes
  using physical path cost, cache affinity, worker load, tenant/model
  constraints, failure-domain spread, and calibration uncertainty.
- **Layer- and backend-aware compute.** Split prefill/decode cost into
  attention, MLP, MoE, logits, sampling, KV-cache kernels, fused kernels,
  launch overhead, CUDA graph behavior, and backend-specific effects.
- **Exact memory timelines.** Move from aggregate memory checks to time-aware
  per-worker/per-GPU accounting for weights, KV cache, activations,
  communication buffers, allocator reserve, graph capture reserve, and
  fragmentation.
- **Production workload behavior.** Add richer traces, bursts, correlated
  prompt/decode sizes, multi-model mixes, streaming disconnects, retries,
  cancellations, timeouts, and tenant-specific routing constraints.
- **Benchmark-backed calibration.** Validate compute, collectives, RDMA,
  PCIe/NVLink copies, KV transfer, queueing behavior, and end-to-end serving
  against measured fixtures.
- **Bottleneck attribution.** Report which physical resource or policy dominates
  TTFT, TPOT, ITL, E2EL, throughput, SLO misses, and objective ranking.
- **Uncertainty accounting.** Treat uncalibrated regions, extrapolated shapes,
  coarse topology, ignored contention, and unsupported runtime effects as
  explicit uncertainty terms instead of silent assumptions.

### Heterogeneous Configs

- **Current capability.** TOML can already describe heterogeneous node groups,
  node IDs, homogeneous GPU type/count nodes, explicit mixed-GPU local
  inventories, disabled GPUs, disabled NICs, NIC counts, rail counts, NIC
  bandwidth, NIC affinity, explicit NIC-to-rail maps, explicit GPU-to-NIC
  locality maps, per-GPU/per-NIC path bandwidth/latency/GPUDirect overrides,
  and custom pairwise or group-expanded inter-node links scoped by rail or
  local-GPU endpoint subsets. Run scenarios can also mark whole nodes disabled,
  maintenance, draining, or reserved; disable GPUs/NICs; and degrade selected
  GPU compute/HBM, NIC bandwidth, and custom inter-node link bandwidth/latency,
  including selected rail and local-GPU endpoint-scoped links.
- **Config target.** Extend the physical topology schema beyond the current
  per-node heterogeneous GPU inventories to cover per-GPU labels, per-GPU
  health, per-NIC health state, PCIe/NUMA/NVSwitch domains, switch topology,
  rack placement, failure domains, and separate inference/storage/control
  fabrics.
- **Interconnect target.** Support different interconnects between arbitrary
  GPU groups, GPU pairs, racks, and fabric domains, building on current
  node-pair, node-group, rail-scoped, and local-GPU endpoint-scoped custom
  links. Add asymmetric bandwidth/latency and missing/degraded links as
  first-class physical resources, not only logical shortcut links.
- **Topology generators.** Add reusable config generators for common layouts
  such as single-node NVSwitch, multi-node rail-aligned InfiniBand, fat tree,
  dragonfly, rack-local islands, oversubscribed racks, and disconnected
  clusters.
- **Constraint target.** Allow placement and serving configs to constrain
  ranks, pools, workers, tenants, and models to topology domains such as
  racks, islands, NUMA sockets, PCIe domains, rails, GPU labels, and failure
  domains.
- **Validation target.** Reject or warn on disconnected pools, invalid rails,
  invalid GPU-to-NIC paths, impossible GPUDirect routes, unsupported dtype/GPU
  combinations, asymmetric route assumptions, and degraded-resource usage.
  The MVP now reports a first pass of custom-topology disconnected-island and
  unusable-rail diagnostics; richer validation still needs endpoint/resource
  evidence and solver-level rejection policy.
- **Backward-compatible config shape.** Keep simple node-group TOML usable, but
  add optional lower-level sections for physical topology and scenario overlays
  so the common case stays terse while advanced configs can express arbitrary
  heterogeneity.

### Prefill/Decode Disaggregation

- **Current capability.** The solver can already search approximate separate
  prefill and decode pools, estimate KV handoff cost, route requests, and
  report TTFT, TPOT, ITL, throughput, and E2EL on one request timeline.
  Configs can explicitly request flexible, colocated, partially disaggregated,
  or fully disaggregated pool modes; validation and solver filtering reject
  incompatible pool candidates.
- **Service target.** Model prefill and decode as independent online services
  with worker-local queues, worker capacity, health, scaling policy, admission
  policy, priority policy, and backpressure.
- **Current service evidence.** Configs can now set prefill, decode, and
  KV-transfer service health plus worker-scale knobs. Required services that
  are unavailable or draining reject serving candidates, and JSON reports
  service health, acceptance state, configured/effective worker slots, request
  counts, admitted/rejected/timed-out counts, queue-cap policies, queue-cap
  hits, backpressure rejection counts, backpressure state, max/p95 queue
  pressure, service time, and worker-slot utilization. Configs can also set a
  v1 `service_backpressure_penalty_weight` under `[serving.traffic]` so
  candidates with queue-cap/backpressure hits are down-ranked by the solver.
  This is still evidence and scoring derived from the approximate scheduled
  timeline; true decode-to-prefill and KV-transfer-to-prefill throttling remains
  a target for the independent service event loop.
- **Worker-capacity target.** Extend the new per-GPU active decode sequence
  accounting into exact worker-slot ownership, queue-local admission,
  prefill-worker slots, and placement-aware prefill/decode worker pools.
- **Current worker evidence.** Scheduling now gates prefill and decode phases
  with routed worker/GPU readiness, KV transfers with optional transfer-worker
  readiness, and configurable per-GPU prefill/KV-transfer/decode worker slots.
  JSON includes approximate per-GPU prefill, KV-transfer, and decode worker
  summaries plus worker-vs-resource queue splits, configured worker slots,
  approximate peak active worker slots, and worker-slot utilization derived
  from the request timeline and capacity profile. Remaining work is true worker
  queue data structures, backpressure, preemption, and queue-state transitions.
- **Mode target.** The config/solver layer now distinguishes colocated,
  partially disaggregated, and fully disaggregated pool modes. Remaining work
  is making those modes share the same production-like service, worker, cache,
  and queue-state primitives instead of an approximate scheduled timeline.
- **KV target.** Track exact KV source, destination, owner, residency,
  transfer path, transfer contention, eviction, migration, and reuse at the
  worker/GPU level.
- **Current KV evidence.** Per-request output now records routed prefill GPUs,
  decode owner GPUs, KV block count, allocated KV tokens, block-fragmentation
  tokens, transfer bottlenecks, and selected GPU-to-GPU topology paths. KV
  handoff cost estimation now uses routed GPU sets where possible, including
  intra-node handoff cost when prefill/decode share a node but not the same GPU
  set. JSON route paths now include structured segment details for GPU/NIC,
  inter-node fabric, GPU-scoped inter-node fabric, and intra-node fabric
  resources, including endpoint IDs, rail IDs where known, bandwidth, and
  latency. Candidate output also aggregates those segments into
  `kv_route_resource_summary` records keyed by resource identity with request
  count, path observations, transfer bytes, and estimated transfer time. KV
  transfer operations now reserve those resource identities in the scheduler,
  enabling first-pass route-specific contention instead of one global KV fabric
  lock. Per-request observations expose `kv_resource_queue_ms`,
  `kv_transfer_resources`, and `kv_transfer_resource_dependencies`, while
  aggregate metrics expose `kv_resource_queue_ms` for TTFT/E2EL attribution.
  KV block ownership records now include decode worker slots and decode
  operation IDs for the owner GPU, tying approximate cache residency back to
  worker execution evidence. Per-request output also includes compact worker
  summaries for prefill source, KV-transfer, decode owner, and KV cache-owner
  roles. Capacity output exposes approximate KV-block peaks, allocated KV
  tokens, block utilization, and fragmentation. Remaining work is exact
  worker-local KV ownership, allocator state, contention, eviction, migration,
  spill, and reuse.
- **Routing target.** Replace coarse pool routing with an online router that
  considers worker load, queue delay, cache affinity, topology path cost,
  route contention, tenant/model constraints, and partial outages.
- **Current routing-risk evidence.** Route coverage and topology-risk penalties
  can flag or down-rank partially routable prefill/decode pool pairs.
  Topology-aware routing also considers selected worker readiness and modeled
  KV route-resource readiness when comparing per-request route candidates.
  JSON now emits each request's route candidates with selected/routable flags,
  estimates, KV bytes, route-resource IDs, bottlenecks, and unroutable
  rejection reasons. Remaining work is physical path capacity, full
  shared-resource occupancy, cache affinity, tenant/model constraints,
  outage-aware routing, placement rejection evidence, and a true online router.
- **Metric target.** Derive TTFT, TPOT, ITL, E2EL, throughput, SLO miss rates,
  deadline misses, and queueing from the same event history, with phase-level
  and worker-level breakdowns. The MVP now emits per-request phase contribution
  records for queue, service, transfer, first-token decode, and decode tail;
  remaining work is deriving all scheduling and metrics directly from the
  authoritative event stream.

## MVP Guardrails

These constraints keep the MVP simple while avoiding design choices that would
block later realism:

- Do not collapse the cluster into anonymous aggregate GPU/network capacity.
  Keep stable node, GPU, pool, route, topology-domain, and calibration IDs even
  when formulas are coarse.
- Keep prefill, KV transfer, and decode as separate phases in configs, solver
  candidates, observations, metrics, and calibration records.
- Derive TTFT, TPOT, ITL, throughput, and E2EL from one request timeline.
- Prefer explicit unsupported, rejected, or approximation records over silent
  fallback when a requested shape is outside the modeled envelope.
- Keep topology, workload shape, serving policy, placement, and calibration as
  separate concepts.
- Preserve a path from aggregate estimates to per-worker, per-GPU, per-NIC,
  per-rail, per-link, per-switch, and per-request accounting.
- Attach units and calibration provenance to values that can come from real
  measurements.
- Keep search APIs separable from file formats so future tools can generate
  configs programmatically.
- Avoid baking one serving stack into the core model. vLLM, TensorRT-LLM,
  SGLang, Dynamo, Ray Serve, Triton, NCCL, RCCL, UCX, SHARP, NVLS, paged
  attention, CUDA graphs, prefix caching, speculative decoding, and KV-transfer
  details should be explicit assumptions.

## MVP-Critical Missing Pieces

These are the highest-priority gaps because they can make solver output look
more trustworthy than it is.

- **Search budgets beyond candidate count.** Configs and CLI now support
  candidate caps for parallelism, prefill, decode, and serving-pair search,
  max-runtime budgets, plus explicit diagnostics for candidate-space size,
  searched/reported counts, candidate-budget and runtime truncation, and whether
  rejected candidates are retained in report artifacts. Remaining work is
  pruning strategy, dominance pruning, random seed, and resume/checkpoint
  support.
- **Richer scenario sweeps.** Sweep calibration profiles, topology degradation,
  disabled GPU/NIC overlays, per-GPU/per-NIC/custom-link degradation, broader
  failure/degraded resources, placement constraints, serving policy,
  approximation policy, and multi-parameter experiment matrices.
- **Richer soft SLO policy.** Configs now support aggregate, metric-specific,
  and traffic-class-specific soft SLO miss penalties for serving ranking.
  Remaining work includes richer tenant/model weighting, class interaction
  policies, and clearer units for non-latency objectives.
- **Richer traffic-class policy.** Extend traffic classes beyond current SLO
  defaults, queue caps, timeouts, hard miss limits, soft penalties, and
  admission priority with class-scoped capacity policies, burst limits, tenant
  isolation, reserved capacity, cancellation behavior, class interaction rules,
  and cost/weight in multi-objective ranking.
- **Approximation policy presets.** Configs now support reusable v1 presets for
  MVP exploration, topology-sensitive planning, memory-capacity planning,
  calibration-only validation, and production recommendation. Explicit policy
  fields can still replace preset defaults.
- **Metric-specific precision gates.** Configs now support
  `[[approximation_policy.metric_gates]]`, so policies can reject topology,
  calibration, runtime, queueing, memory, or specific approximation codes only
  when the selected serving objective/metric requires that precision.
- **Cross-file validation.** Validate cluster, workload, run, and calibration
  files together before solving, including disconnected pools, impossible ranks,
  unsupported dtypes, invalid locality constraints, missing calibration
  coverage, incompatible hardware, and invalid prefill/decode routes.
- **Structured unsupported-shape evidence.** Every invalid or unsupported
  candidate should report category, phase, resource, observed value, limit,
  unit, candidate ID, route or placement ID, and remediation hint.
  Prefill/decode parallelism rejections now preserve rejected rank-placement
  evidence when available, so serving-level rejection records retain resource,
  code, observed value, limit, unit, and remediation instead of falling back to
  a plain rejected-reason string.
- **Golden experiment fixtures.** Add checked examples and stable JSON smoke
  tests for homogeneous, heterogeneous, islanded, rail-aware, oversubscribed,
  colocated, and prefill/decode-disaggregated setups.
- **Clear trust-boundary output.** Text and JSON output should make it obvious
  when a result depends on coarse topology, aggregate memory, approximate
  queueing, uncalibrated fits, or unsupported locality assumptions.
- **Cluster inventory diagnostics.** Text/JSON topology diagnostics now cover
  disconnected custom-topology islands, island-spanning node groups, unusable
  rail-scoped custom links, and configured GPU-to-NIC locality risks such as
  disabled endpoints, unavailable paths, host-staged/non-GPUDirect paths, and
  slow local NIC segments. Remaining work is broader degraded-resource
  diagnostics, risky multi-hop routes, oversubscribed domains, and physical
  locality assumptions.

## Heterogeneous Cluster and Topology Gaps

The current custom node groups and custom links are enough for an MVP, including
different interconnects between different sets of nodes and selected local GPU
subsets. The missing work is to make that topology physically meaningful under
placement and contention.

- Promote node groups into first-class topology domains for placement,
  aggregation, constraints, and reporting.
- Support different GPU generations and vendors in one cluster, including
  per-GPU HBM size, HBM bandwidth, FLOPs, supported dtypes, NVLink generation,
  health state, and placement labels.
- Extend mixed-GPU nodes with per-GPU labels, health state, NUMA/PCIe/NVSwitch
  locality, per-GPU NIC path constraints, and maintenance/degraded state.
- Represent CPU sockets, NUMA domains, PCIe switches, NVSwitch domains, NICs,
  rail IDs, rack switches, fabric switches, storage/control fabrics, logical
  service pools, and failure domains as addressable resources.
- Extend current per-GPU/per-NIC path labels and bandwidth/latency/GPUDirect
  overrides into first-class direct, same-PCIe-switch, same-NUMA-socket,
  cross-socket, host-staged, degraded, and unavailable topology resources.
- Model paths such as `GPU -> NVSwitch/PCIe -> NIC -> rail -> switch fabric ->
  NIC -> PCIe/NVLink -> GPU`.
- Add per-link bandwidth, latency, duplex mode, startup overhead, protocol
  overhead, sharing scope, failure domain, and degradation state.
- Support asymmetric links, missing links, explicit disconnected islands,
  mixed interconnect generations, and separate inference/storage/control
  fabrics.
- Model rail pinning, rail striping, adaptive routing, route hashing, shared
  uplinks, incast/outcast, head-of-line blocking assumptions, and
  oversubscription.
- Model route sharing across NICs, rails, PCIe links, NVLink/NVSwitch links,
  rack switches, fabric switches, and copy engines.
- Add disabled, reserved, degraded, maintenance-mode, and draining states for
  GPUs, NICs, links, nodes, rails, and switches.
- Surface topology bottlenecks in output, including cross-socket paths,
  unavailable GPUDirect, overloaded rails, slow host-staged copies, and
  oversubscribed uplinks.
- Extend deterministic topology inventory beyond the current summaries and
  first-pass custom-topology diagnostics into richer diagnostics for GPU-type
  counts, mixed-GPU node contents, node groups, NICs, rails, intra-node
  topology, custom links, locality assumptions, and degraded resources.
- Optionally support MIG or fractional accelerator capacity if multi-tenant
  fractional GPU modeling becomes part of the product.

## Config Expressiveness Gaps

The config layer should let users start with a small TOML file and then opt into
more physical detail without changing solver APIs.

- Generalize endpoint selectors beyond the current node groups, explicit node
  IDs, local GPU IDs, and rail filters to support GPU labels, NIC IDs, rack
  IDs, islands, failure domains, fabrics, and service pools.
- Let interconnect resources connect arbitrary endpoint selectors, not just the
  currently supported node pairs, node groups, local GPU subsets, and rails.
  This is the path to modeling different physical links between GPU subsets,
  NIC subsets, racks, and fabric domains.
- Split topology into local node topology, rack/fabric topology, and scenario
  overlays. Local topology covers sockets, NUMA domains, PCIe switches,
  NVSwitch domains, GPUs, NICs, and copy engines. Fabric topology covers rails,
  switches, uplinks, oversubscription, and inter-rack paths. Scenario overlays
  cover disabled, degraded, reserved, maintenance, or draining resources.
- Add topology templates/generators for common layouts so users do not need to
  hand-write hundreds of links for rail-aligned InfiniBand, NVSwitch islands,
  fat tree, dragonfly, rack-local islands, or oversubscribed racks.
- Support explicit asymmetric links and route policy assumptions, including
  rail pinning, rail striping, adaptive routing, ECMP/hash behavior, and
  whether GPUDirect is available on each path.
- Keep IDs stable and human-readable in parsed configs, JSON output, route
  observations, rejection evidence, and calibration records.
- Validate advanced configs structurally before solving: duplicate IDs,
  missing endpoints, invalid rails, impossible GPU-to-NIC paths, disconnected
  service pools, unsupported dtype/GPU combinations, and scenario overlays that
  remove required resources.

## Placement Search Gaps

The current placement is deterministic and capability-aware. Realistic planning
needs topology-aware placement search.

- Search placements across GPUs, nodes, racks, islands, rails, NUMA domains,
  PCIe domains, NVSwitch domains, and failure domains.
- Keep high-traffic tensor, expert, sequence, and context groups inside fast
  topology domains when possible.
- Prefer pipeline boundaries that minimize activation and KV movement over slow
  links.
- Search phase-aware placements for prefill workers, decode workers, tensor
  groups, pipeline stages, MoE experts, cache owners, and routing replicas.
- The MVP now supports user-supplied explicit rank placements from config files
  for the base solver and serving prefill/decode pools.
- Support richer placement constraints over node sets, GPU sets, racks,
  islands, rails, NUMA domains, service pools, tenants, model IDs, failure
  domains, and partial rank groups.
- Score placements by route cost, HBM headroom, GPU capability, queueing
  pressure, failure-domain spread, topology risk, cost, and calibration
  uncertainty.
- Add pruning so larger heterogeneous clusters remain tractable.
- Explain selected and rejected placements with structured evidence for the
  full physical placement search. The MVP now emits candidate-level rank
  placement evidence, but future work must add topology-domain, tenant,
  service-pool, rack, rail, NUMA, PCIe, and NVSwitch reasons.
- [x] Return a v1 serving Pareto annotation when candidates trade off
  throughput, TTFT, TPOT, ITL, E2EL, memory pressure, and GPU footprint, with
  explicit Pareto dimension metadata in candidate output. Future Pareto
  frontiers should extend this to topology risk, cost, power, uncertainty, and
  richer operational policy dimensions.

## Prefill/Decode Disaggregation Gaps

The current setup handles disaggregation approximately. To model production
disaggregation, prefill and decode need to become separate schedulable services.

- Distinguish colocated, partially disaggregated, and fully disaggregated modes
  with the same production service primitives. The config and solver now expose
  those modes at the pool level; true worker/service/cache state is still
  missing.
- Model prefill and decode as independent services with worker state, queue
  state, routing state, scaling policy, failure domains, cache ownership, and
  topology-aware handoff.
- Move from node-set KV handoff estimates to worker/GPU-level source and
  destination placement.
- Extend the current approximate per-GPU prefill and decode worker slot knobs
  into explicit service-level capacity for workers, KV-transfer workers, cache
  owners, scheduler shards, and admission queues.
- Extend the current aggregate/per-node/per-GPU prefill token caps, observed
  pressure reports, and approximate per-GPU prefill worker slots into exact
  worker-local prefill admission with queue state, per-worker active-token
  accounting, max prefill chunks, and backpressure.
- Model KV-transfer contention with collectives, activation sends, PCIe/NVLink
  copies, RDMA, storage/control traffic, and other point-to-point traffic.
- Replace approximate topology/load-aware routing with an online router that
  considers worker load, queue state, cache affinity, KV ownership, tenant/model
  placement, topology path cost, route contention, and partial-route
  availability.
- Refine chunked prefill into production-like chunk sizing, chunk scheduling,
  priority interaction, first-token deadlines, decode backpressure, and
  cache-hit behavior.
- Track KV-cache ownership by decode worker/GPU, including exact allocation,
  fragmentation, eviction, migration, spill, and reuse. The current block
  accounting is an aggregate capacity approximation, not a real allocator.
- Convert aggregate/per-node/per-GPU decode residency checks into exact
  per-worker KV-block accounting.
- Promote request phase spans into state transitions such as queued for
  prefill, prefilling, transferring KV, queued for decode, decoding, cancelled,
  timed out, evicted, migrated, and completed. The current JSON output now
  includes derived lifecycle events for these core request transitions, with
  eviction and migration still unmodeled.
- Add backpressure from decode to prefill and from KV transfer to prefill.
- Report TTFT as prefill queueing plus prefill execution plus KV handoff plus
  first decode iteration.
- Report TPOT and ITL as steady-state decode iteration latency under contention.
- Report E2EL as queueing plus prefill plus KV handoff plus all decode tokens.

## Serving Scheduler and Queueing Gaps

- Build a true online event loop with arrivals, ready queues, batching windows,
  active decode sets, admission, timeout, cancellation, preemption, and
  scheduling decisions.
- Add worker-local queues and active sequence sets for prefill and decode.
- Refine continuous decode with realistic token batching windows, CUDA graph
  reuse, launch overhead, sampling/logprob overhead, per-token deadlines,
  sequence churn, and iteration-level contention.
- Extend admission control beyond queue-delay and decode/KV capacity caps to
  max prefill tokens, max decode batch tokens, SLO pressure, worker-local
  service state, traffic class, tenant, and model constraints.
- Add priority and preemption policies.
- Add scheduler tick, batching-window, runtime bookkeeping, RPC/control-plane,
  serialization, metrics collection, and health-check overhead assumptions.
- Track occupancy over time for GPU compute, HBM, copy engines, PCIe, NVLink,
  NVSwitch, NICs, rails, fabric links, and CPU-side resources.
- Extend stochastic arrivals beyond fixed-gap, Poisson, and deterministic
  bursts to diurnal patterns, self-similar traffic, and trace-derived arrival
  models.
- Configurable measurement windows, warmup/cooldown trims, and optional
  steady-state auto-detection are now available; remaining work is richer
  convergence diagnostics and approximation evidence.

## Compute Model Gaps

The current model is still too coarse for calibrated production predictions.

- Replace coarse FLOP estimates with layer-aware prefill and decode models.
- Separate attention, MLP, MoE routing, logits, sampling, normalization,
  KV-cache kernels, and backend-specific fused kernels.
- Refine the coarse HBM-bandwidth decode floor into a calibrated shape-aware
  memory-bandwidth model for decode.
- Model tensor-core utilization by dtype, batch shape, sequence length, hidden
  size, backend, and GPU generation.
- Extend executable linear latency fits into efficiency tables, piecewise fits,
  nonlinear fits, and backend-specific fitted functions from benchmark sweeps.
- Model GQA/MQA, KV dtype, quantized KV cache, quantized weights, mixed
  precision, sliding-window attention, paged attention, and cache misses.
- Model speculative decoding, draft/target models, acceptance rate, rollback
  cost, and scheduling interaction with normal decode.
- Model MoE routing skew, expert imbalance, dropped tokens, capacity factors,
  and all-to-all pressure.
- Include CPU-side tokenization, sampling, detokenization, logprob, structured
  output, and tool-call overhead when those affect the served workload.

## Memory and Capacity Gaps

- Replace coarse component memory estimates with exact placement- and
  time-aware accounting for weights, KV cache, activations, temporary buffers,
  communication buffers, runtime reserve, graph capture reserve, allocator
  reserve, page tables, and fragmentation.
- [x] Add a v1 approximate memory-pressure timeline derived from request
  schedule spans, reported separately for prefill, KV transfer, and decode with
  active request/token counts, KV block counts, limiting GPU, capacity-used
  fraction, headroom, dominant component, and component estimates.
- [x] Add a `memory_pressure` serving objective and deterministic
  memory-pressure tie-breakers so TOML configs can explicitly solve for lower
  HBM pressure instead of only observing memory headroom after ranking.
- [x] Add an optional hard `max_memory_pressure_fraction` serving constraint
  with structured rejection evidence, so users can enforce HBM safety margin
  independently of the chosen latency/throughput objective.
- [x] Add an optional hard `max_unique_gpus` serving constraint with structured
  rejection evidence, so v1 configs can cap candidate GPU footprint without
  requiring detailed NUMA/NIC placement modeling.
- Track memory pressure over time separately for prefill, KV transfer, and
  decode phases.
- Model KV-cache block allocation, block size, block table overhead,
  fragmentation, reuse, eviction, migration, spill, and allocator reserve.
- Refine placement-aware per-GPU residency checks into exact worker/GPU
  KV-block ownership.
- Account for tensor, pipeline, expert, data, context, and sequence parallel
  memory effects.
- [x] Add v1 memory-pressure evidence to candidate output and OOM records:
  capacity-used fraction, limiting GPU, dominant memory component, and
  component share breakdowns.
- Add OOM rejection records that identify the dominant component and specific
  worker/GPU.
- Report memory pressure timelines and per-component headroom.
- Add model-family-specific KV formulas, including GQA/MQA, dtype, and sliding
  window behavior.

## Communication and Collective Gaps

- Model collective algorithms explicitly: ring, tree, double binary tree,
  hierarchical, CollNet, SHARP, NVLS, and vendor-specific auto-selection.
- Add message-size thresholds, protocol choices, channels, chunking, startup
  costs, and per-algorithm traffic formulas.
- Distinguish NCCL, RCCL, MPI, UCX, vendor fabric plugins, and reduction
  offload behavior where relevant.
- Model all-reduce, reduce-scatter, all-gather, all-to-all, broadcast,
  pipeline sends, activation transfers, KV handoffs, and point-to-point copies
  as different operations.
- Separate tensor, expert, sequence, context, pipeline, KV-transfer, and
  serving-control traffic.
- Account for contention between collectives, KV transfers, activation sends,
  RDMA copies, PCIe/NVLink copies, copy engines, storage traffic, and
  control-plane traffic.
- Add route sharing when multiple logical operations traverse the same physical
  rail, NIC, PCIe path, NVSwitch path, or fabric switch.
- Add explicit congestion assumptions for oversubscribed fabrics, including
  queueing, incast/outcast, PFC/head-of-line blocking assumptions, or a clear
  approximation record when ignored.

## Workload Realism Gaps

- Expand trace fixtures and trace replay controls for large production-like
  workloads.
- Extend trace schema with request class, cache key, tenant/model routing
  constraints, timeout policy, cancellation, streaming disconnects, and
  lifecycle metadata.
- Add downsampling, warmup trimming, replay transforms, and richer replay modes
  for CSV/JSONL imports.
- Add bursty arrivals, diurnal curves, correlated request sizes, multi-modal
  prompt/decode distributions, and tenant/model mixes.
- Model distinct model shapes, routing constraints, capacity pools, and
  per-model serving stacks for multi-model workloads.
- Add tokenizer/model metadata for realistic prompt/decode shapes.
- Extend prefix-cache modeling with prefix sharing distributions, cache
  admission, eviction, cache residency, cache misses, and sharing across workers.
- Add LoRA adapter mixes, speculative acceptance rates, tool-use expansion,
  multimodal token expansion, and structured-output overhead.
- Model streaming responses and client disconnects as first-class outcomes.

## Metrics and Reporting Gaps

- Extend aggregate, tenant/model, routed prefill/decode node, and routed
  node-set metrics into per-pool, per-worker, per-node, per-GPU, per-NIC,
  per-rail, per-link, per-switch, per-tenant, and per-model views as the model
  gains those resources.
- Add richer SLO breakdowns for TTFT, TPOT, ITL, E2EL, deadline misses, and
  traffic-class misses.
- Keep ranking/objective JSON stable as objective terms expand, including
  nominal objective, SLO penalties, uncertainty penalties, topology-risk
  penalties, cost terms, and final ranking score.
- [x] Add a compact v1 serving `bottleneck_summary` derived from existing
  rejection, topology, memory-pressure, resource-utilization, and
  phase-resource-utilization evidence so candidate JSON/text exposes the most
  important limiting factors without requiring consumers to reconstruct them
  from detailed arrays.
- [x] Add a compact v1 serving `approximation_summary` that rolls detailed
  approximation records into status, category counts, top codes, and flags for
  calibration risk, coarse topology, approximate queueing, and policy rejection
  in candidate JSON and text notes.
- Extend phase/resource utilization into deeper bottleneck attribution for
  queueing, compute, HBM, memory pressure, KV residency, collectives, KV
  transfer, PCIe, NIC, rail, switch, copy engine, and fabric contention.
- Extend current route-coverage and request-path reporting into richer
  placement explanations, route explanations, physical path diagrams, and
  rejected alternative summaries.
- [x] Add v1 serving hardware-footprint reporting for each candidate: unique
  nodes/GPUs, prefill/decode/shared footprint, aggregate HBM, aggregate HBM
  bandwidth, aggregate peak/effective FLOPs, and throughput efficiency per GPU,
  peak TFLOP, and HBM GB.
- Add richer topology/compatibility rejection codes and evidence fields.
- Add CSV artifacts for metrics, utilization, timelines, occupancy, memory
  pressure, request lifecycle, calibration, and route paths.
- Add summary output that remains readable with many candidates.
- Extend scenario sensitivity and rank-sensitivity reports beyond the current
  v1 top-candidate and per-candidate CSV/JSON summaries so they explain when
  traffic, placement, topology, or calibration assumptions dominate the
  ranking.

## Calibration and Validation Gaps

- Add measured benchmark fixtures for compute, collectives, RDMA, PCIe/NVLink
  copies, KV transfer, and end-to-end serving traces.
- Fit profiles by hardware, fabric, model family, dtype, backend, and serving
  stack without changing simulator code.
- [x] Add a v1 per-serving-candidate calibration summary that reports active,
  calibrated, and uncalibrated phase coverage, applied fit counts, uncertainty,
  confidence, extrapolation, and calibration-gate counts so downstream solver
  consumers can distinguish calibrated metrics from fallback estimates.
- [x] Add v1 calibration-policy thresholds for maximum relative and absolute
  fit uncertainty, using the existing warn/reject `fit_uncertainty` gate so
  configs can reject high-uncertainty calibration fits instead of only missing
  uncertainty metadata.
- [x] Add a v1 calibration-policy threshold for minimum serving phase coverage,
  so configs can reject candidates whose active prefill/decode/KV/queue
  components are mostly uncalibrated without requiring perfect coverage.
- Compare collective estimates against NCCL/RCCL tests.
- Compare point-to-point and KV-transfer estimates against measured RDMA and
  PCIe/NVLink copies.
- Compare prefill and decode latency estimates against real serving traces.
- Validate solver rankings against known deployment choices and measured
  serving outcomes.
- Extend provenance with driver, CUDA/ROCm, NCCL/RCCL, backend version, kernel
  settings, benchmark command, source artifact, environment hash, and notes.
- Extend calibration gates with confidence intervals, holdout validation,
  interpolation/extrapolation policies, coverage thresholds, and rank
  sensitivity.
- Add sensitivity tests that show how solver rankings change as uncertain
  calibration parameters move.
- Add regression tests that preserve calibrated behavior across refactors.

## Solver Objectives and Cost Gaps

- Extend objectives beyond latency, throughput, SLO miss rate, and uncertainty
  into cost, power, memory headroom, topology risk, failure-domain risk,
  calibration confidence, and Pareto frontiers.
- Add hard and soft constraints for SLOs, memory headroom, failure domains,
  placement domains, tenant isolation, model placement, reserved capacity, and
  operational risk.
- [x] Add an optional hard `min_throughput_tokens_per_s` serving constraint with
  structured rejection evidence, so users can minimize latency while enforcing
  a throughput floor.
- [x] Add optional hard serving latency ceilings for TTFT, TPOT, ITL, and E2EL
  with structured rejection evidence, so users can enforce basic latency SLOs
  while optimizing another objective.
- [x] Add optional v1 KV-route topology constraints for rail diversity, rail
  metadata, and GPUDirect-like GPU/NIC locality, so users can reject
  topology-risky prefill/decode placements without full physical contention
  modeling.
- [x] Add a v1 configurable serving cost/power model for GPU-hours,
  node-hours, GPU watts, node watts, and energy rates, with candidate-level
  total cost, energy, and per-1k output/request cost estimates. Future work
  should extend this to NIC/fabric cost, reservations, amortization, and
  failure-domain capacity.
- [x] Add v1 solver objectives for configured serving cost, energy, and average
  power, so TOML workloads can rank candidates by economics or power envelope
  without requiring full scheduler or fabric contention simulation.
- [x] Make deterministic serving candidate IDs part of the reusable solver data
  model, not only JSON output, so calibration records, sensitivity reports, and
  downstream exports can join candidate-level data without reimplementing CLI
  ID logic.
- [x] Group serving rejections by root cause in CLI text and JSON output so
  users can see the dominant failure modes across the full candidate set
  without inspecting every rejected prefill/decode placement.
- Add sensitivity analysis across traffic rate, prompt/decode distribution,
  calibration profile, degraded resources, and topology changes.

## Software Stack and Runtime Assumption Gaps

- [x] Represent v1 serving stack assumptions in workload TOML and CLI output,
  and emit calibration-profile mismatch evidence when the workload runtime does
  not match `profile.serving_stack`.
- [x] Represent v1 serving runtime feature assumptions such as paged attention,
  CUDA graphs, prefix cache, speculative decode, or custom backend features in
  workload and calibration-profile TOML/CLI output, and emit approximation
  evidence when requested runtime features are not covered by profile metadata
  so unsupported runtime effects are explicit and policy-gateable.
- Represent richer serving stack assumptions such as backend versions, worker
  runtime modes, scheduler implementations, and custom runtimes.
- Model backend-specific behavior for paged attention, chunked prefill,
  continuous batching, CUDA graphs, speculative decoding, prefix cache, LoRA,
  and KV-transfer implementation.
- Represent collective backend assumptions such as NCCL, RCCL, MPI, UCX, SHARP,
  NVLS, and vendor-specific plugins.
- Track software-version compatibility with hardware features such as FP8,
  NVLink/NVSwitch generation, GPUDirect RDMA, reduction offload, and backend
  kernel support.
- Add explicit approximation flags when ignored runtime effects may dominate
  small-batch serving.

## Reliability and Operations Gaps

- Model failure domains and placement spread across racks, rows, islands,
  availability zones, maintenance pools, power domains, and regions.
- Support node, GPU, NIC, link, rail, switch, rack, and fabric failures or
  degradation as scenario inputs.
- Model reserved capacity, draining nodes, rolling upgrades, anti-affinity,
  failover capacity, and replica spread.
- Add optional power, thermal, and rack-density constraints.
- Report operational risk such as single-rack concentration, single-rail
  dependency, no failover capacity, or concentration on degraded resources.

## Testing and Fixture Gaps

- Add golden tests for homogeneous, heterogeneous, islanded, rail-aware,
  oversubscribed, colocated, and prefill/decode-disaggregated examples.
- Add invalid-config tests for duplicate IDs, disconnected pools, impossible
  ranks, invalid rails, invalid affinity, unsupported dtypes, incompatible
  hardware, missing links, and invalid disaggregation routes.
- Add deterministic simulation tests for fixed arrivals, Poisson arrivals,
  trace replay, batching, routing, SLO misses, admission rejection, timeouts,
  cancellation, and rejection evidence.
- Add topology tests for route selection and contention across NICs, rails,
  PCIe, NVLink/NVSwitch, copy engines, and fabric switches.
- Add calibration fixture tests comparing compute, collective, transfer, and
  serving latency estimates against benchmark points.
- Add CLI smoke tests for table output, JSON output, trace output, rejection
  output, calibration profile loading, scenario sweeps, and large bounded search
  spaces.
- Add documentation tests or checked examples so config snippets do not drift
  from parser behavior.

## Suggested Implementation Phases

### Phase 1: Make the MVP Hard to Misuse

- Add richer traffic-class interaction policies and objective weighting.
- Add pruning search budgets and richer scenario sweeps.
- Extend metric-specific precision gates with per-artifact/reporting scopes if
  future consumers need gates beyond the selected serving objective.
- Tighten cross-file validation for unsupported shapes and invalid
  prefill/decode routes.
- Add structured evidence for unsupported candidates and richer rejection codes.
- Add golden CLI examples and smoke tests for the key topology/serving shapes.

### Phase 2: Make Serving Metrics Credible

- Build the online serving event loop.
- Split prefill and decode into independent services with worker-local queues.
- Add topology/cache/load-aware routing.
- Add richer admission, timeout, cancellation, preemption, and backpressure.
- Replace aggregate decode residency with exact worker/GPU KV-block ownership.
- Keep TTFT, TPOT, ITL, throughput, and E2EL derived from the same event
  timeline.

### Phase 3: Make Topology Matter

- Add explicit PCIe, NUMA, NVSwitch, NIC, rail, rack switch, fabric switch, and
  copy-engine resources.
- Add topology-aware placement search.
- Add route sharing and contention for rails, links, switches, and copy engines.
- Add collective algorithm selection and calibrated communication formulas.
- Add topology-domain-aware reporting and route/placement explanations.

### Phase 4: Make Compute and Memory Realistic

- Add layer-aware prefill and decode models.
- Add calibrated memory-bound decode models.
- Add exact placement- and time-aware memory accounting.
- Refine prefix cache, chunked prefill, paged attention, speculative decoding,
  quantization, MoE, and LoRA modeling.

### Phase 5: Calibrate Against Reality

- Add benchmark import fixtures.
- Fit profiles per hardware, fabric, model family, dtype, backend, and serving
  stack.
- Expand calibration gating with confidence intervals, holdout validation, and
  interpolation/extrapolation policies.
- Validate solver rankings against known deployment choices and measured
  serving outcomes.
- Add calibrated rank-sensitivity regression tests against measured outcomes.

## MVP Completion Criteria

The MVP is credible enough for serious relative planning when:

- Unsupported cluster, placement, routing, workload, and calibration shapes
  fail or warn with structured evidence.
- Heterogeneous node groups, custom links, and prefill/decode pools have golden
  examples and smoke tests.
- TTFT, TPOT, ITL, throughput, and E2EL are all tied to one request timeline and
  clearly report approximation and calibration status.
- Solver rankings show hard constraints, soft SLO penalties, uncertainty
  penalties, and objective components separately.
- Calibration profiles can explain whether each recommended candidate is
  measured, interpolated, extrapolated, or outside coverage.
- JSON output is stable enough for downstream plotting and CI regression tests.
