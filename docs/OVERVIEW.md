# Overview

```yaml
Overview:
  description: >
    inference-sim is a Rust CLI simulator for planning GPU-cluster LLM
    inference deployments. It loads TOML cluster, workload, run, and
    calibration-profile files; searches parallelism (TP/PP/EP/DP) and
    prefill/decode serving placements over homogeneous or heterogeneous
    clusters; schedules an approximate request timeline; and emits ranked
    candidates with structured metrics, approximation, calibration,
    rejection, and bottleneck evidence in JSON/CSV artifacts.
  subsystems:
    config: >
      TOML schema parsing and cross-file validation for cluster, workload,
      run, scenario, and calibration-profile inputs (src/config/, src/config.rs).
    types: >
      Core value types for GPUs, fabrics, topology metadata, collectives,
      and byte/bandwidth units (src/types/).
    topology_graph: >
      Graph over GPU/NIC/intra-node/inter-node resources with Dijkstra
      routing, NUMA penalties, rails, and per-hop bottleneck attribution
      (src/topology_graph.rs).
    solver: >
      Parallelism candidate enumeration, rank placement, the analytical
      compute/memory roofline, collective network costs, and calibration-fit
      application (src/solver.rs, src/solver/).
    scheduler: >
      Greedy list scheduler over named exclusive resources; produces the
      scheduled operation timeline that all latency metrics derive from
      (src/scheduler.rs).
    serving: >
      Serving simulation on top of solver+scheduler: pool search, routing,
      admission, continuous prefill/decode batching approximation, KV
      transfer, lifecycle events, measurement windows, ranking, and
      evidence assembly (src/serving.rs, src/serving/).
    cli: >
      Argument parsing, run orchestration, scenario overlays, and all
      JSON/CSV/markdown artifact emission (src/cli.rs, src/cli/).
  data_flow: >
    CLI loads and validates TOML configs -> solver enumerates parallelism
    candidates and places ranks on cluster GPUs -> compute roofline and
    network-cost models price prefill/decode/KV operations (optionally
    overridden by calibration fits) -> scheduler lays operations onto a
    shared timeline -> serving layer replays traffic through pools,
    batching, and routing over the topology graph -> ranking scores
    candidates with objectives, penalties, Pareto ranks, and uncertainty
    adjustments -> cli emits results.json plus CSV evidence artifacts.
Features Index:
  compute_roofline:
    description: >
      Analytical per-phase latency model: dense parameter FLOPs plus causal
      attention FLOPs for prefill/decode, and weight-read plus KV-cache-read
      HBM bandwidth terms for decode.
    entry_points: [Solver::estimate_compute_latency_s, Solver::decode_compute_latency_s]
    depends_on: []
    doc: docs/features/compute_roofline.md
```
