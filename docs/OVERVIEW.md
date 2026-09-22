```yaml
Overview:
  description: >
    inference-sim is a Rust deployment-planning simulator for LLM inference.  It
    loads TOML cluster, workload, run and calibration-profile files, enumerates
    parallelism and serving-pool candidates, simulates prefill/decode/KV-transfer
    timelines over a topology model, and ranks candidates against latency,
    throughput, cost and SLO objectives.  Every number it reports carries
    approximation, calibration, bottleneck and rejection evidence so a ranking can
    be audited rather than trusted.
  subsystems:
    config: >
      TOML parsing and validation for clusters, workloads, run files, calibration
      profiles and calibration/approximation policies (src/config.rs,
      src/config/).  Owns CalibrationProfileMetadata, CalibrationFittedModel,
      CalibrationBenchmarkPoint and the gate-mode policy types.
    workload: >
      Model and request shapes - ModelSpec, InferenceRequest, InferencePhase,
      DType (src/workload.rs).
    topology_graph: >
      Node, rail, island and fabric graph used for collective and KV-transfer
      routing (src/topology_graph.rs).
    calibration: >
      SimulationCalibration scalars (compute efficiency, phase scales, collective
      and KV-transfer scales, scheduler overhead) applied to analytic phase
      estimates (src/calibration.rs).
    solver: >
      Parallelism search, roofline phase estimates, network cost, and the
      calibration-fit application layer that can override an analytic phase
      estimate with a fitted linear model (src/solver.rs, src/solver/).
    serving: >
      Continuous-batching serving simulation - scheduling, routing, memory
      pressure, metric fits, approximations, Pareto frontiers (src/serving.rs,
      src/serving/).
    scheduler: >
      Request admission and batching timeline primitives (src/scheduler.rs).
    cli: >
      Argument parsing, run-file and scenario execution, JSON/CSV/manifest output
      (src/cli.rs, src/cli/).
    aisimulate_calibration: >
      Offline Python tool that composes NVIDIA AISimulate's measured per-op GPU
      performance tables into a calibration-profile TOML this repo can load
      (tools/aisimulate_calibration/).
  data_flow: >
    cli parses CLI flags and (optionally) a run file, then config loads the
    cluster, workload and calibration profile TOMLs into typed structs.  The
    loaded calibration profile contributes scalar calibration, shape coverage,
    fitted phase/serving models and benchmark residuals.  solver enumerates
    parallelism candidates, computes roofline prefill/decode/KV estimates over
    topology_graph, and lets a compatible CalibrationFittedModel override the
    analytic estimate for its phase; calibration policies decide whether missing
    coverage, extrapolated fits or incomplete provenance warn or reject.  serving
    then simulates traffic over the surviving candidates and produces metrics plus
    evidence, which cli renders as tables, JSON or CSV.  The aisimulate_calibration
    tool runs entirely outside this loop: it reads AISimulate Parquet tables and
    writes a calibration-profile TOML that config later loads like any other.

Features Index:
  aisimulate_calibration:
    description: >
      Compose AISimulate's measured GEMM, context/generation attention and
      custom-allreduce tables into phase-level prefill/decode calibration fits
      with holdout validation stats, honest feature ranges, derived efficiency
      scalars and benchmark residual entries.
    entry_points:
      - tools/aisimulate_calibration/convert.py
      - examples/calibration_h100_vllm_llama31_70b.toml
    depends_on: []
    doc: docs/features/aisimulate_calibration.md
```
