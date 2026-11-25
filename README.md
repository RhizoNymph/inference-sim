# inference-sim

`inference-sim` is a Rust command-line simulator for planning GPU-cluster LLM
inference deployments. It loads TOML cluster, workload, run, and calibration
profiles; searches parallelism and prefill/decode serving placements; and emits
ranked candidates with metrics and evidence.

The current V1 target is an explainable planning tool, not a production-accurate
cluster emulator. Use it to compare candidate configurations and understand why
they pass, fail, or depend on modeling assumptions. Treat absolute latency and
throughput numbers as only as credible as the calibration profile and coverage
used for the run.

## What It Models

- Homogeneous and heterogeneous GPU clusters with node groups, mixed GPU
  inventories, racks, topology islands, failure domains, NICs, rails, GPU/NIC
  locality, NUMA penalties, disabled resources, degraded resources, and custom
  inter-node links.
- Parallelism search across tensor, pipeline, expert, and data parallel ranks.
- Serving search for colocated, partially disaggregated, and fully
  disaggregated prefill/decode pools.
- Iteration-level serving engine for colocated continuous batching: each
  engine step decodes every running sequence and fills its token budget with
  chunked prefill, and requests wait in a queue until their KV fits.
- Synthetic and trace-backed request traffic, including queue caps, worker
  slots, batching controls, SLOs, deadlines, traffic classes, measurement
  windows, and steady-state reporting.
- Calibration profiles, calibration policies, approximation policies, rejection
  evidence, bottleneck evidence, route evidence, placement evidence, and rank
  sensitivity.

## Quickstart

Run commands from this directory.

```sh
cargo test
```

Start the local GUI server and open the printed URL:

```sh
node gui/server.mjs
```

The GUI builds cluster and workload TOML, visualizes the selected topology,
generates bounded solver-owned searches for parallelism and prefill/decode
pools, runs the simulator through the local server, and shows ranked candidate
configs with key metric estimates and a metric-vs-metric Pareto scatter plot.
The generated rank search is derived from the model shape and topology: tensor
ranks follow valid attention/hidden-size divisors, pipeline ranks span feasible
layer/GPU stage counts, and data ranks span available GPU replica counts.
Serving workloads use synthetic arrivals, request-shape distributions, and
continuous batching controls rather than a single static batch size. It also
renders the serving process from simulator JSON. You can open
`gui/index.html`
directly in a browser for offline editing; in that mode, paste CLI JSON into
the Process tab.

After downloading `cluster.toml` and `workload.toml` from the GUI, the same run
can be reproduced manually:

```sh
cargo run -- --cluster cluster.toml --workload workload.toml \
  --json --trace --request-limit 8
```

Paste the JSON output into the GUI's Process tab to inspect request lifecycle
timelines against the configured topology.

Run the smallest colocated serving example and print JSON:

```sh
cargo run -- --cluster examples/h100_cluster.toml \
  --workload examples/homogeneous_serving_workload.toml \
  --json --top-k 1
```

Run a heterogeneous, fully disaggregated prefill/decode example:

```sh
cargo run -- --cluster examples/heterogeneous_cluster.toml \
  --workload examples/heterogeneous_disaggregated_workload.toml \
  --json --top-k 1 --max-serving-pairs 4
```

Run a calibrated trace example through a run file:

```sh
cargo run -- --run examples/trace_run.toml --top-k 1 --request-limit 5
```

Run a scenario sweep:

```sh
cargo run -- --run examples/run_scenarios.toml --top-k 1
```

Write report artifacts instead of a large stdout payload:

```sh
cargo run -- --cluster examples/h100_cluster.toml \
  --workload examples/homogeneous_serving_workload.toml \
  --output-dir tmp/readme-run --top-k 1
```

When `--output-dir` is set, the default output profile is `all`, so the command
writes a `manifest.txt` plus `summary/`, `compare/`, `calibration/`, and
`audit/` subdirectories.

## Common Workflows

Use `--cluster` and `--workload` for one-off runs. The cluster file describes
hardware and topology; the workload file describes the model, request shape,
search space, calibration, and optional serving deployment.

Use `--run` when a run needs reusable output settings, search budgets, or
scenarios. Paths inside a run file are resolved relative to that run file, so
the examples can refer to sibling TOML and trace files.

Use `--output-dir` for repeatable artifacts. Use `--output-profile` only when
you want one profile:

```sh
cargo run -- --run examples/trace_run.toml \
  --output-dir tmp/trace-audit --output-profile audit
```

Use `--trace`, `--occupancy`, or `--critical-path` when debugging a candidate.
Those flags imply JSON output because they add structured diagnostic sections.

Use `--max-candidates`, `--max-prefill-candidates`, `--max-decode-candidates`,
`--max-serving-pairs`, and `--max-search-runtime-ms` to keep broad searches
bounded.

## Configuration Map

Cluster TOML files contain:

- `[cluster]`: preset and node count for simple homogeneous clusters.
- `[[node_groups]]` or `[[nodes]]`: heterogeneous node definitions, GPU
  inventories, GPU labels, topology metadata, disabled GPUs, and per-GPU
  profile overrides.
- `[nics]` or per-node `nics`: NIC count, bandwidth, rails, GPU-to-NIC
  affinity, NUMA maps, disabled NICs, and per-NIC overrides.
- `[interconnect]`: fabric kind, variant, oversubscription, and optional custom
  links.

Workload TOML files contain:

- `[model]`: model shape, parameter size, dtype, and optional expert spec.
- `[request]`: base request shape used by parallelism scoring and as defaults
  for serving traffic.
- `[search]`: tensor/pipeline/expert/data rank search space.
- `[placement]`: optional explicit rank placement for non-serving runs.
- `[serving]`: optional serving mode, objective, pool candidates/search, metric
  ceilings, SLO penalty weights, cost model, and topology-risk controls.
- `[serving.traffic]`: synthetic traffic, trace inputs, batching, capacity,
  queue, timeout, SLO, measurement-window, and worker-slot controls.
- `[serving.prefill_search]` and `[serving.decode_search]`: separate
  parallelism search spaces for disaggregated serving.
- `[calibration_profile]`, `[calibration]`, `[calibration_policy]`, and
  `[approximation_policy]`: runtime calibration and policy gates.

Run TOML files contain:

- top-level `cluster` and `workload`, or `[run] cluster = ...` and
  `workload = ...`.
- `[output]`: format, top-k, artifact paths, trace/occupancy/critical-path
  options, and output profiles.
- `[search]`: search budgets and rejected-candidate retention.
- `[[scenarios]]`: traffic scaling, shape scaling, calibration overrides,
  calibration-profile overrides, and topology degradation overlays.

All config files currently use `schema_version = 1`.

## Output Profiles

`summary` writes a concise Markdown report in `summary/results.md`.

`compare` writes a text report plus compact CSVs for candidate metrics, metric
breakdowns, rejections, bottlenecks, rank sensitivity, and scenario sensitivity
when scenarios are configured.

`calibration` writes JSON plus calibration-oriented CSVs: candidate metrics,
phase calibration, approximation evidence, calibration residuals, and request
metrics.

`audit` writes full JSON with trace, occupancy, and critical-path diagnostics,
plus CSVs for request metrics, lifecycle events, services, utilization, memory
pressure, timeline, occupancy, placement evidence, worker evidence, rejections,
route paths, KV route resources, bottlenecks, phase calibration,
approximations, calibration residuals, rank sensitivity, and scenario
sensitivity when scenarios are configured.

Individual CSV paths can also be supplied directly, for example:

```sh
cargo run -- --run examples/trace_run.toml \
  --serving-metrics-csv tmp/serving_metrics.csv \
  --rank-sensitivity-csv tmp/rank_sensitivity.csv
```

## Example Files

- `examples/h100_cluster.toml` and
  `examples/homogeneous_serving_workload.toml`: smallest colocated serving
  path.
- `examples/heterogeneous_cluster.toml`: mixed H100/A100 cluster with richer
  topology metadata.
- `examples/heterogeneous_disaggregated_workload.toml`: fully disaggregated
  serving across heterogeneous pools.
- `examples/heterogeneous_partially_disaggregated_workload.toml`: partially
  disaggregated serving.
- `examples/heterogeneous_colocated_workload.toml`: heterogeneous colocated
  serving.
- `examples/heterogeneous_rail_island_cluster.toml` and
  `examples/heterogeneous_rail_island_workload.toml`: rail, island, and fabric
  differences.
- `examples/heterogeneous_oversubscribed_cluster.toml` and
  `examples/heterogeneous_oversubscribed_workload.toml`: route-resource
  contention and oversubscription evidence.
- `examples/trace_workload.toml`, `examples/request_trace.csv`, and
  `examples/request_trace.jsonl`: trace-backed serving traffic.
- `examples/calibration_h100_a100.toml`: reusable calibration profile with
  fit, residual, uncertainty, and runtime provenance metadata.
- `examples/calibration_h100_vllm_llama31_70b.toml` and
  `examples/llama31_70b_calibrated_workload.toml`: calibration profile
  generated by `tools/aisimulate_calibration/convert.py` from AISimulate's
  measured H100 vLLM operation tables, plus a Llama-3.1-70B workload whose
  request shape lands inside the profile's fitted feature ranges so both phase
  fits apply as interpolations.
- `examples/trace_run.toml` and `examples/run_scenarios.toml`: run-file
  workflows for calibrated traces and scenario sweeps.

## Approximation And Calibration Policy

Workload TOML can include an approximation preset:

```toml
[approximation_policy]
preset = "topology_sensitive"
```

Supported presets are `mvp_exploration`, `topology_sensitive`,
`memory_capacity`, `calibration_only`, and `production_recommendation`.
Explicit fields such as `default`, `reject_categories`, `reject_codes`,
`warn_categories`, and `warn_codes` override the selected preset field.

Metric-scoped gates reject or warn only when a specific ranking metric depends
on an approximation:

```toml
[[approximation_policy.metric_gates]]
metrics = ["e2el", "throughput"]
reject_categories = ["topology"]
```

Calibration profiles can provide shape coverage, benchmark residuals, fitted
phase models, aggregate serving metric fits, uncertainty, and runtime
provenance. Calibration policies decide whether missing coverage, extrapolated
fits, weak fit metadata, invalid shapes, or incomplete runtime provenance warn
or reject candidates.

### Generating Profiles From AISimulate Measurements

`tools/aisimulate_calibration/convert.py` composes NVIDIA
[AISimulate](https://github.com/ai-dynamo/AISimulate)'s measured per-operation
GPU tables (GEMM, context/generation attention, custom allreduce) into a
calibration profile for a dense Llama-style decoder, with holdout-validated fit
statistics, honest feature ranges, derived efficiency scalars, and benchmark
residuals:

```sh
uv run tools/aisimulate_calibration/convert.py \
  --aisimulate-dir /path/to/AISimulate \
  --system h100_sxm --backend vllm --version 0.24.0 \
  --model-preset llama-3.1-70b --tp 2,4,8 \
  --output examples/calibration_h100_vllm_llama31_70b.toml
```

`--self-test` replays the pipeline against a checked-in CSV fixture and diffs
the result against a golden profile. `examples/llama31_70b_calibrated_workload.toml`
runs the generated profile end to end:

```sh
cargo run -- --cluster examples/h100_cluster.toml \
  --workload examples/llama31_70b_calibrated_workload.toml --json
```

See `docs/features/aisimulate_calibration.md` for the op-walk composition,
interpolation rules, and limits.

## Modeling Limits

V1 intentionally keeps these approximate:

- Colocated continuous-batching serving runs on an iteration-level engine
  (vLLM V1-style steps priced from their composition, with a waiting queue);
  it reserves KV per whole sequence instead of preempting and does not model
  API-server latency. Disaggregated pools, independent batching, and other
  candidates use an approximate phase-by-phase scheduler
  (`docs/features/serving_iteration_engine.md`).
- No exact KV block allocator or worker-local cache state.
- No full PCIe/NUMA/NVSwitch physical graph.
- No exact KV allocator with eviction, migration, spill, or prefix-cache
  residency.
- No full physical rail, switch, copy-engine, or shared-resource contention
  model.
- Measured calibration covers only a two-scalar static-batch fit on one lab
  (see `docs/validation_ledger.md` for what has and has not been validated).

The simulator emits approximation, calibration, bottleneck, rejection, and
rank-sensitivity evidence so comparisons can be audited. For production
recommendations, prefer calibrated workloads with policy gates that reject the
approximations or coverage gaps you cannot tolerate.

## Lab Measurements And Calibration

`tools/lab/` measures vLLM on real GPUs and calibrates the simulator against
it. An experiment is a TOML spec (`tools/lab/specs/`) naming a lab
(`tools/lab/labs/`, where node quirks live), a model, parallelism, and either
static-batch shapes or a `vllm bench serve` request-rate sweep:

```sh
cargo build --release
python3 tools/lab/lab.py run tools/lab/specs/rtx3090_qwen7b_static_pp1.toml --dry-run
python3 tools/lab/lab.py run tools/lab/specs/rtx3090_qwen7b_static_pp1.toml
python3 tools/lab/lab.py calibrate tools/lab/specs/rtx3090_qwen7b_static_pp1.toml \
  --run-dir lab-runs/<date>-qwen7b-static-pp1
python3 tools/lab/lab.py report --run-dir lab-runs/<date>-qwen7b-static-pp1
```

Results go to `lab-runs/<date>-<name>/` with a manifest, the fitted
calibration profile (which the simulator verifiably loads), and a
measured-vs-simulated report. Measured accuracy per regime is tracked in
`docs/validation_ledger.md`; the design is in `docs/features/lab_harness.md`.

## Development

Useful checks before changing behavior:

```sh
cargo fmt -- --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

The crate has no non-Rust runtime services. Its direct dependencies are
`serde`, `serde_json`, and `toml`.
