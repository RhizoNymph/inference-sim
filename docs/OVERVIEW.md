```yaml
Overview:
  description: >
    inference-sim is a Rust command-line simulator for planning GPU-cluster LLM
    inference deployments. It loads TOML cluster, workload, run, and calibration
    profiles; searches parallelism and prefill/decode serving placements; and
    emits ranked candidates with latency, throughput, capacity, bottleneck,
    rejection, approximation, and calibration evidence as human-readable text,
    JSON, and CSV artifacts. The target is an explainable planning tool: every
    number carries provenance and applicability status rather than pretending to
    be a cycle-accurate cluster emulator.
  subsystems:
    config:
      role: >
        TOML ingestion and validation for clusters, workloads, run files,
        scenarios, traces, serving policies, approximation policies, and
        calibration profiles. Produces validated in-memory structures and
        typed ConfigError values; no simulation logic lives here.
      key_files:
        - src/config.rs (module root, shared validation helpers)
        - src/config/cluster.rs (cluster/topology parsing)
        - src/config/run_config.rs (run files, scenarios, output artifacts)
        - src/config/calibration_config.rs (calibration profiles, fits, gates)
        - src/config/serving_config.rs (serving search and policy parsing)
        - src/config/sections.rs (raw serde section structs)
        - src/config/trace.rs (request trace ingestion)
    types:
      role: >
        Core value types shared by every other subsystem: Bytes, Bandwidth,
        GPU/NIC/node addressing, operational state, GPU profiles, fabric
        profiles, cluster/node topology, collective descriptions, and
        ParallelismConfig/RankPlacement.
      key_files:
        - src/types/common.rs
        - src/types/gpu.rs
        - src/types/topology.rs
        - src/types/configs.rs
        - src/types/collective.rs
        - src/types/fabric/
    topology_graph:
      role: >
        Builds a resource graph over GPUs, intra-node links, NICs, rails, and
        inter-node links, and routes transfers across it so contention and path
        evidence can be attributed to concrete physical resources.
      key_files:
        - src/topology_graph.rs
    solver:
      role: >
        Parallelism search. Places ranks on GPUs, checks memory feasibility,
        estimates per-phase compute and collective cost from an analytical
        roofline, applies calibration fits that override roofline phases,
        builds an operation trace, and scores/ranks candidate
        ParallelismConfigs.
      key_files:
        - src/solver.rs (Solver, scoring, phase latency, tests)
        - src/solver/placement.rs (rank placement and evidence)
        - src/solver/network_cost.rs (collective/transfer cost, fit lookup)
        - src/solver/operations.rs (operation trace construction)
        - src/solver/calibration_fits.rs (fit feature dictionary and evaluation)
    scheduler:
      role: >
        Resource-constrained scheduling of simulated operations. Turns an
        operation trace with dependencies and resource claims into start/finish
        times, a makespan, and per-resource utilization.
      key_files:
        - src/scheduler.rs
    serving:
      role: >
        Serving search on top of the parallelism solver: colocated, partially
        disaggregated, and fully disaggregated prefill/decode pools, arrivals
        and traffic classes, batching, queueing, KV transfer routing, SLOs,
        capacity and memory pressure, rejections, objective scoring, and
        calibrated serving metrics.
      key_files:
        - src/serving.rs (module root, ServingSolver)
        - src/serving/scheduling/ (queueing and batching)
        - src/serving/metric_fits.rs (serving-metric calibration fits)
        - src/serving/calibration.rs, measurement.rs, observations.rs
        - src/serving/capacity.rs, memory.rs, routing.rs, ranking.rs
    cli:
      role: >
        Argument parsing, run orchestration, scenario sweeps, and all
        presentation: text report, JSON output, and the CSV artifact family.
        Owns no simulation math.
      key_files:
        - src/cli.rs (module root and run orchestration)
        - src/cli/args.rs, scenario.rs, search.rs
        - src/cli/json_output.rs, presentation.rs, csv.rs
        - src/cli/calibration.rs, diagnostics.rs, metrics.rs, policy.rs
    calibration:
      role: >
        SimulationCalibration scalar knobs (compute efficiency, phase scales,
        collective scales, scheduler overhead, serving memory fractions) with
        sanitizing defaults, consumed by solver and serving.
      key_files:
        - src/calibration.rs
    workload:
      role: >
        ModelSpec, DType, ExpertSpec, InferenceRequest, and InferencePhase, the
        workload description every estimate is computed against.
      key_files:
        - src/workload.rs
    aisimulate_calibration:
      role: >
        Offline Python tool, outside the simulation loop, that composes NVIDIA
        AISimulate's measured per-operation GPU performance tables into a
        calibration-profile TOML config can load like any other.
      key_files:
        - tools/aisimulate_calibration/convert.py (uv script entry point)
        - tools/aisimulate_calibration/converter/ (tables, op walk, fitting, emission)
        - tools/aisimulate_calibration/fixtures/ (self-test CSVs and golden profile)
  data_flow: >
    cli parses arguments and loads TOML through config, producing a Cluster
    (types::topology), a ModelSpec/InferenceRequest (workload), a
    SimulationCalibration, an optional CalibrationProfileMetadata, and policy
    structures. For each scenario, cli applies topology overlays and calls
    Solver::rank_configs / ServingSolver, passing the calibration profile inside
    SolverOptions. The solver places ranks, estimates analytical phase latency,
    then asks solver::calibration_fits whether a fitted model overrides that
    phase: calibration_feature_values builds a name->value feature dictionary
    from the model, request, and ParallelismConfig, fit_matches filters the
    profile's fits by model kind, phase, and target, and evaluate_fit computes
    intercept + sum(coefficient * feature) plus range/applicability/uncertainty
    evidence. Collective and transfer costs are priced through
    topology_graph routes. The resulting operations go to scheduler, whose
    makespan becomes estimated_latency_s. serving reuses the solver per pool and
    layers arrivals, batching, queueing, KV transfer, and SLO accounting on top,
    applying its own serving-scope fits through
    Solver::fitted_latency_from_features. cli then ranks, gates (calibration and
    approximation policies), and renders text/JSON/CSV, carrying every fit
    application and its applicability status into the output.
    aisimulate_calibration runs entirely outside that loop: it reads AISimulate
    Parquet op tables, composes phase latencies, fits the basis features
    solver::calibration_fits already evaluates, and writes a profile TOML that a
    later cli run loads through config.

Features Index:
  aisimulate_calibration:
    description: >
      Offline converter that composes AISimulate's measured GEMM,
      context/generation attention, and custom-allreduce tables into
      phase-level prefill/decode calibration fits with holdout validation
      stats, honest feature ranges, derived efficiency scalars, and benchmark
      residual entries.
    entry_points:
      - tools/aisimulate_calibration/convert.py
      - examples/calibration_h100_vllm_llama31_70b.toml
      - examples/llama31_70b_calibrated_workload.toml
    depends_on: [calibration_fits]
    doc: docs/features/aisimulate_calibration.md
  calibration_fits:
    description: >
      Fitted linear models in calibration profiles that override analytical
      roofline phase latencies, including the derived basis-feature dictionary
      that lets linear-in-coefficient fits express quadratic attention and
      context-dependent KV-read curve shapes across a tensor-parallel sweep.
    entry_points:
      - src/solver/calibration_fits.rs::calibration_feature_values
      - src/solver/calibration_fits.rs::fit_matches
      - src/solver/calibration_fits.rs::evaluate_fit
      - src/solver/network_cost.rs::fitted_phase_latency
      - src/config/calibration_config.rs::parse_calibration_fits
    depends_on: [config, solver]
    doc: docs/features/calibration_fits.md
```
