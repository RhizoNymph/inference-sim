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
        - src/config/cluster.rs (cluster/topology parsing, NIC caps, directional links)
        - src/config/collective_curves.rs ([[collective_curves]] parsing)
        - src/config/run_config.rs (run files, scenarios, output artifacts)
        - src/config/calibration_config.rs (calibration profiles, fits, gates)
        - src/config/serving_config.rs (serving search and policy parsing)
        - src/config/sections.rs (raw serde section structs)
        - src/config/trace.rs (request trace ingestion)
    types:
      role: >
        Core value types shared by every other subsystem: Bytes, Bandwidth,
        GPU/NIC/node addressing, operational state, GPU profiles, fabric
        profiles (including one-way NIC caps and asymmetric links),
        cluster/node topology, collective descriptions and their pricing
        evidence, measured collective curves, and
        ParallelismConfig/RankPlacement.
      key_files:
        - src/types/common.rs
        - src/types/gpu.rs
        - src/types/topology.rs
        - src/types/configs.rs
        - src/types/collective.rs
        - src/types/collective_curves/ (MeasuredCurve, CurveTarget, CollectiveCurveSet)
        - src/types/fabric/ (direction.rs: NicDirectionCaps, AsymmetricLink)
    topology_graph:
      role: >
        Builds a directed resource graph over GPUs, intra-node links, NICs,
        rails, and inter-node links (each direction carries its own
        bandwidth/latency, honouring NIC egress/ingress caps and asymmetric
        links), and routes transfers across it so contention and path
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
        - src/solver/collective_pricing.rs (measured-curve pricing and evidence)
        - src/solver/operations.rs (operation trace construction)
        - src/solver/calibration_fits.rs (fit feature dictionary and evaluation)
        - src/solver/step_cost.rs (per-engine-step roofline for the serving engine)
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
        calibrated serving metrics. Continuous-batching candidates run on an
        iteration-level engine (vLLM V1-style steps priced from their
        composition): colocated pools on one worker per GPU set,
        disaggregated and partially disaggregated pools as prefill and
        decode workers joined by decode-initiated KV pulls over FIFO
        directed link queues. Independent batching and data-parallel
        replicas stay on the phase-pipeline scheduler.
      key_files:
        - src/serving.rs (module root, ServingSolver)
        - src/serving/scheduling.rs (request states, scheduler dispatch)
        - src/serving/engine/ (iteration-level serving engine)
        - src/serving/engine/disaggregated.rs, engine/transfer/ (disaggregated workers, KV transfer plans and link queues)
        - src/serving/scheduling/pipeline.rs (phase-pipeline scheduler)
        - src/serving/scheduling/summary.rs (shared metric summary)
        - src/serving/scheduling/ (prefill/decode batching for the pipeline)
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
    lab_harness:
      role: >
        Offline Python tool, outside the simulation loop, that measures vLLM
        on real lab GPUs from TOML experiment specs (generating and running
        every ssh/docker/venv/Ray command), runs the simulator over the same
        spec, fits compute_efficiency and decode_memory_bandwidth_scale from
        static-batch runs, emits and verifies a calibration profile, and
        writes measured-vs-simulated reports into lab-runs/.
      key_files:
        - tools/lab/lab.py (uv script entry point)
        - tools/lab/labharness/ (spec, commands, collective_plan, runner, results, curves, curves_toml, simulate, fitting, evaluate, profile, report)
        - tools/lab/remote/ (bench_latency.py, collective_bench.py, probe_env.py, run on the nodes)
        - tools/lab/labs/, tools/lab/specs/ (lab inventories and experiment matrices)
        - lab-runs/ (raw measurements and derived reports)
        - docs/validation_ledger.md (measured accuracy per regime)
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
    directed topology_graph routes; a collective whose op, node placement,
    and rank count match one of the cluster's measured collective curves is
    priced from that curve instead (solver::collective_pricing), and every
    collective's pricing becomes approximation evidence. The resulting operations go to scheduler, whose
    makespan becomes estimated_latency_s. serving reuses the solver per pool and
    layers arrivals, batching, queueing, KV transfer, and SLO accounting on top,
    applying its own serving-scope fits through
    Solver::fitted_latency_from_features. For continuous-batching
    candidates, serving builds solver IterationCostModels from the placed
    configs and runs the discrete-event engine (src/serving/engine/), which
    prices every engine step from its prefill/decode composition; for
    disaggregated pools it also builds per-request KV transfer plans from
    the topology graph and measured send_recv curves
    (src/serving/engine/transfer/) and moves each request from its prefill
    worker to its decode worker through FIFO link queues. The engine writes
    request lifecycles back into the same request states the phase-pipeline
    scheduler fills; both feed one shared metric summary. cli then ranks, gates (calibration and
    approximation policies), and renders text/JSON/CSV, carrying every fit
    application and its applicability status into the output.
    aisimulate_calibration runs entirely outside that loop: it reads AISimulate
    Parquet op tables, composes phase latencies, fits the basis features
    solver::calibration_fits already evaluates, and writes a profile TOML that a
    later cli run loads through config. lab_harness also runs outside the loop:
    it drives vLLM on lab nodes, invokes the release binary as a subprocess
    (--json) for every measured shape or request rate, fits the two roofline
    scalars from static-batch results, and writes a profile TOML that config
    loads; its reports feed docs/validation_ledger.md.

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
  collective_curves:
    description: >
      Measured latency-vs-size curves in the cluster TOML that price
      collectives and directed point-to-point sends (log-log interpolation,
      floor/bandwidth extrapolation) in place of alpha-beta, with
      extrapolation/derived/absent/suspended evidence; plus one-way NIC caps
      and per-direction custom links so routed transfers use their direction
      and ring collectives are bounded by the slower one.
    entry_points:
      - src/config/collective_curves.rs::parse_collective_curves
      - src/solver/collective_pricing.rs::Solver::curve_pricing
      - src/solver/network_cost.rs::Solver::estimate_collective_with_calibration
      - src/topology_graph.rs::TopologyGraph::from_cluster
      - tools/lab/lab.py curves
      - lab-runs/2026-09-30-tp2/rtx3090_lab_cluster_measured_curves.toml
    depends_on: [config, topology_graph, solver, lab_harness]
    doc: docs/features/collective_curves.md
  compute_roofline:
    description: >
      Analytical per-phase latency model: dense parameter FLOPs plus causal
      attention FLOPs for prefill/decode, and weight-read plus KV-cache-read
      HBM bandwidth terms for decode, sharded across tensor/pipeline ranks
      only for the attention and KV terms.
    entry_points:
      - src/solver.rs::estimate_compute_latency_s
      - src/solver.rs::decode_compute_latency_s
      - src/solver.rs::prefill_baseline_s
    depends_on: [types, workload]
    doc: docs/features/compute_roofline.md
  serving_iteration_engine:
    description: >
      Deterministic discrete-event serving loop modeled on vLLM V1 for
      colocated continuous-batching candidates: per-step decode tokens plus
      budget-filling prefill chunks, KV-gated admission with a waiting queue
      instead of rejection, and per-step latency from the solver roofline for
      the step's actual composition (with TP all-reduces and PP stages).
    entry_points:
      - src/serving/engine.rs::select_scheduler_model
      - src/serving/engine.rs::run_iteration_engine
      - src/serving/engine/core.rs::run_engine
      - src/solver/step_cost.rs::IterationCostModel
    depends_on: [compute_roofline]
    doc: docs/features/serving_iteration_engine.md
  disaggregated_serving_engine:
    description: >
      Disaggregated and partially disaggregated prefill/decode serving on the
      iteration engine, modeled on vLLM's NixlConnector: prefill workers run
      chunked-prefill steps, finished prompts queue on their decode worker,
      decode admission reserves KV and starts a pull of the request's
      TP/PP-sharded prompt KV priced from directed send_recv curves or routed
      alpha-beta, FIFO per directed link, and the decode worker recomputes
      the last prompt token to emit the client's first token (vLLM proxy
      TTFT convention, configurable).
    entry_points:
      - src/serving/engine.rs::select_scheduler_model
      - src/serving/engine/disaggregated.rs::run_disaggregated_engine
      - src/serving/engine/core.rs::run_engine_jobs
      - src/serving/engine/transfer/plan.rs::KvTransferPlanner
      - examples/rtx3090_qwen7b_disaggregated_workload.toml
    depends_on: [serving_iteration_engine, collective_curves]
    doc: docs/features/disaggregated_serving_engine.md
  lab_harness:
    description: >
      Turn-key measurement-to-calibration pipeline: TOML experiment specs
      (static-batch shapes or vllm bench serve rate sweeps) with lab quirks as
      config, a dry-runnable remote runner with JSON-record completion
      detection and manifests, simulator sweeps mirroring each benchmark, a
      two-scalar median-ratio fit with leave-one-shape-out validation,
      calibration-profile emission verified by the simulator, per-regime
      markdown comparisons, and a collective-bench mode (NCCL sweep with
      receiver-timed sends) whose results `lab.py curves` converts into the
      cluster's [[collective_curves]] section.
    entry_points:
      - tools/lab/lab.py
      - tools/lab/specs/rtx3090_qwen7b_static_pp1.toml
      - tools/lab/specs/rtx3090_qwen7b_serving_pp1.toml
      - tools/lab/specs/rtx3090_nccl_curve.toml
      - docs/validation_ledger.md
    depends_on: [compute_roofline, calibration_fits]
    doc: docs/features/lab_harness.md
```
