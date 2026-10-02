# Lab measurement-to-calibration harness

`tools/lab/` turns a TOML experiment spec into (1) the exact commands that
measure vLLM on real GPUs, (2) matching simulator runs, (3) a fitted
calibration profile the simulator loads, and (4) a markdown comparison of
measured versus simulated numbers per regime. Results land in
`lab-runs/<date>-<name>/` and are summarised in `docs/validation_ledger.md`.

An H100 (or any other) matrix is a new lab file plus new spec files; no code
changes.

## Scope

- Experiment specs: model (HF id + simulator shape), nodes, parallelism
  (tp, pp), engine settings, vLLM version pin, and exactly one workload mode:
  `static-batch` (batch x prompt shapes, decode length, warmup/iters) or
  `serving` (`vllm bench serve` random dataset, fixed input/output lengths,
  Poisson request-rate sweep including `inf`, prompt count, seed).
- Lab specs: per-node ssh host, address, launch method (docker image +
  runtime flags + container user, or native venv), HF_HOME override,
  NCCL/GLOO socket interface, Ray port, and the simulator cluster TOML for the
  same hardware.
- A remote runner that generates every command (sync, stale-process cleanup,
  environment probe, Ray cluster bring-up, detached benchmark launch, polling,
  collection, teardown) and either prints it (`--dry-run`) or runs it.
- Static-batch benchmark (`remote/bench_latency.py`) and environment probe
  (`remote/probe_env.py`) that run inside the vLLM environment on the node.
- Simulator sweeps over the same spec; serving sweeps generate a colocated
  serving workload mirroring the benchmark's arrival process.
- A two-scalar fitter (`compute_efficiency`, `decode_memory_bandwidth_scale`)
  with leave-one-shape-out validation and calibration-profile emission,
  verified by loading the profile in the simulator.
- Markdown/JSON comparison per regime with mean |error| per metric.
- `fit-curve`: a compute-efficiency curve over tokens per forward pass from
  a batch-1 prefill token sweep, plus (optionally) the per-request frontend
  latency from isolated-request `vllm bench serve` runs, written into a copy
  of a base profile and verified by the simulator.
- A vLLM KV-capacity estimate (`kv_estimate.py`) that mirrors vLLM's memory
  profiler, used when no measured capacity is available.

## Non-scope

- Fitting anything beyond the two roofline scalars, the efficiency curve,
  and the frontend latency (no `[[fits]]` linear models;
  `tools/aisimulate_calibration` owns those), and no serving-metric fits.
  Serving is validated, never fitted.
- Fixing simulator physics. The harness reports gaps (for example the serving
  loop's linear decode scaling, below); fixes belong in `src/`.
- Provisioning nodes (installing vLLM, pulling images, downloading weights).
  The probe step detects a wrong vLLM version and aborts.
- Multi-node Docker. Multi-node runs use native venvs and Ray only.
- Scheduling shared GPUs. Whoever runs `lab.py run` is responsible for making
  sure nobody else is using the nodes.

## Data and control flow

```
specs/*.toml ──parse_experiment──> Experiment (spec.py; resolves labs/*.toml)
    │
    ├─ run ──build_plan──> Plan (commands.py, pure) ──Runner / --dry-run──> lab nodes
    │                                                    │
    │                            lab-runs/<date>-<name>/ <┘ spec.toml, lab.toml, probe-*.json,
    │                                                       manifest.json, measured.jsonl | rate_*.json, logs
    │
    ├─ sim ───────> SimRunner (simulate.py) ──inference-sim --json──> sim-<tag>-<calibration>.jsonl|json
    │
    ├─ calibrate ─> evaluate.calibrate_static:
    │                 default sweep → fitting.fit_scalars → fitted sweep
    │                 → leave-one-shape-out (fitting.leave_one_out_folds + one sweep per fold)
    │                 → profile.render_profile → evaluate.verify_profile (simulator loads it)
    │               writes calibration_profile-<tag>.toml, calibration-<tag>.json,
    │                      report-calibration-<tag>.md
    │
    ├─ validate ──> evaluate.evaluate_static | evaluate_serving under a calibration
    │               writes validation-<tag>.json, report-validation-<tag>.md
    │
    ├─ fit-curve ─> evaluate.fit_curve_static: sweeps at efficiency 0.005 and 1.0
    │                 → efficiency_curve.fit_efficiency_curve
    │               [frontend.fit_frontend_latency from --frontend-dir]
    │               → curve_profile.render_curve_profile (base profile + keys)
    │               → evaluate.verify_curve_profile (simulator echoes the curve)
    │               writes calibration_profile-<tag>.toml, efficiency-curve-<tag>.json,
    │                      report-calibration-<tag>.md
    │
    └─ report ────> report.md = concatenated report-*.md fragments
```

### Spec resolution (`spec.py`)

`parse_experiment` loads the experiment TOML, resolves `experiment.lab`
relative to the spec file, looks up each node by name, and resolves the launch
method: `auto` takes each node's `default_launch`; all nodes must resolve to
the same kind, and multi-node must be `venv`. It then validates cross-field
invariants (tp*pp equals the listed GPU count; static shapes fit
`max_model_len`, `max_num_seqs`, and `max_num_batched_tokens` so a static
batch really is one batch; `mode` matches exactly one workload section) and
returns a frozen `Experiment`. Variants are encoded as types: `Launch =
DockerLaunch | VenvLaunch`, `Workload = StaticBatchWorkload | ServingWorkload`,
`ReferenceBatch = LittlesLawBatch | FixedBatch`, `ContainerUser`,
`SimAdmission`.

### Command generation (`commands.py`)

`build_plan` is pure and produces `Plan(actions, teardown)`, where an action
is a `Step` (one shell script on one host, optional capture) or a `Wait` (a
probe polled until it prints `complete=1`). Phases, in order:

1. `sync` - `mkdir` the remote bench and run dirs; `scp` the two remote
   scripts to `$HOME/inference-sim-lab/bench/`.
2. `clean` - kill stale benchmark/server processes, remove the run's
   container, `ray stop --force` wherever a venv exists.
3. `preflight` - run `probe_env.py` in the same environment the benchmark
   uses; the runner stores it as `probe-<node>.json` and aborts if
   `vllm_version` differs from the pin.
4. `cluster` (multi-node only) - `ray start --head` on the first node,
   `ray start --address` on the others, then a Python check that Ray sees
   tp*pp GPUs.
5. `launch` + `wait` - static-batch: one detached `bench_latency.py` run;
   serving: a detached `vllm serve`, a `/health` readiness wait, then per rate
   a detached `vllm bench serve` and a wait for its result JSON.
6. `collect` - dump container logs, `scp` results and logs into the local run
   dir.
7. `teardown` (always runs, even after failures) - the same kills as `clean`,
   plus a chown pass for root containers.

Script conventions (all enforced by tests):

- Remote steps execute as `ssh -o BatchMode=yes HOST bash -l -s` with the
  script on stdin; `--dry-run` renders them as heredocs. Scripts are never
  re-quoted and never appear on any process command line.
- Process matching uses bracketed patterns (`pkill -f '[b]ench_latency.py'`),
  so a pattern never matches the command line that carries it.
- HF_HOME is set explicitly for every vLLM process (the nodes' own defaults
  are wrong). Venv processes get `PATH=<venv>/bin:$PATH` (Ray daemons and the
  driver need the venv's `ninja` for vLLM's JIT kernels), NCCL/GLOO socket
  interface, and `VLLM_HOST_IP`.
- Containers run as the ssh user (`--user "$(id -u):$(id -g)"`, `HOME=/tmp`,
  `VLLM_CACHE_ROOT=/tmp/vllm-cache`, HF cache mounted at `/hf`) so nothing
  root-owned lands in the host cache. `container_user = "root"` adds a chown
  teardown step.
- Completion detection for static batch counts only JSON lines carrying the
  `lab.static_batch.v1` tag in the benchmark's own output file and requires
  the `lab.done.v1` sentinel; vLLM's INFO logs share stdout, which broke a
  naive line count before. A wait fails fast when the process or container is
  gone before completion, and times out after `runner.run_timeout_s`.

The serving commands match the ones validated on node0 on 2026-09-30
(`lab-runs/2026-09-30-serving-baseline/run_serving.sh`): server from the
image's default `vllm serve` entrypoint with `--model ...`, published on
`127.0.0.1:<port>`; client from the node's venv with `--dataset-name random
--random-range-ratio 0 --ignore-eos --disable-tqdm --percentile-metrics
ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result
--result-filename rate_<R>.json`.

### Runner (`runner.py`)

`Runner.run` refuses to reuse an existing run dir, copies `spec.toml` and
`lab.toml` into it, executes actions in order through an injectable
`Executor`, polls waits every `runner.poll_interval_s`, runs collect steps
best-effort after a failure, always runs teardown, and writes
`manifest.json` (`lab.manifest.v1`: run id, status, error, timings, simulator
git sha + dirty flag, node probes, every rendered command).

### Simulation (`simulate.py`)

Static batch: for each shape and each phase (`prefill`, `decode`,
`end_to_end`) one workload with `[request]` = (batch, prompt, decode,
`max_sequence_tokens = prompt + decode`) and `[search]` pinned to the spec's
(tp, pp). Calibration is `DefaultCalibration` (no section),
`ScalarCalibration` (`[calibration]`), or `ProfileCalibration`
(`[calibration_profile] path`). Runs are asyncio subprocesses under a
semaphore; workload files are written atomically and only when changed.

Serving: one colocated workload per request rate on the spec's nodes:

| vLLM benchmark | simulator workload |
|---|---|
| Poisson arrivals at rate R (`burstiness = 1`) | `arrival = "poisson"`, `arrival_rate_per_s = R`, `arrival_seed = seed` |
| `--request-rate inf` (all at t=0) | `arrival = "fixed"`, `arrival_gap_ms = 0` |
| `--num-prompts N` | `request_count = sim_request_count` (default N) |
| `--random-input-len I --random-output-len O --random-range-ratio 0` | `batch_sizes = [1]`, `prompt_tokens = [I]`, `decode_tokens = [O]` |
| chunked prefill, `--max-num-batched-tokens B` | `max_prefill_batch_tokens = B`, `max_prefill_chunk_tokens = B` |
| `--max-num-seqs S` | `max_decode_sequences = S`, `max_decode_batch_tokens = S` |
| GPU KV cache size from `server.log` | `max_resident_tokens`, `max_kv_blocks` (16-token blocks) |
| `--no-enable-prefix-caching` | `prefix_cache_hit_rate = 0` |

KV capacity precedence: `serving.kv_cache_tokens` in the spec, else the
`GPU KV cache size: N tokens` line in the run's `server.log`, else
`utilization * HBM - weights` (over-estimates by ~10% because it ignores
activation and CUDA-graph workspace).

**Iteration engine.** The rendered workload (colocated pool, continuous
prefill and decode batching, one parallelism search for both phases) runs on
the simulator's iteration engine (`docs/features/serving_iteration_engine.md`).
The engine prices every engine step from its own prefill/decode composition
and queues requests past `max_num_seqs` or the KV capacity, as vLLM does.

**Reference batch.** `[request].batch_size` no longer affects serving
metrics; it only sizes the static scores the simulator uses for feasibility
and memory headroom. The checked-in spec uses `sim_reference_batch = 1`.
`sim_reference_batch = "littles_law"` (or `--reference-batch littles_law`)
still works: it sets B to the Little's-law steady-state concurrency, iterating
`B = ceil(rate * (prefill(1) + O * step(B)))` from B = 1 with static
simulations. That only mattered for the phase-pipeline scheduler, which
scaled decode iterations linearly from B. On the engine it returns the same
metrics as B = 1.

**Admission.** `sim_admission = "engine"` passes vLLM's `max_num_seqs` and KV
capacity; the simulator queues past them. `sim_admission = "uncapped"` (or
`--admission uncapped`) lifts both limits to the request count, a what-if
with no admission pressure.

### Fitting (`fitting.py`)

With the simulator at base scalars `e0 = 0.35` (`SimulationCalibration`
default) and `s0 = 1.0`:

- `compute_efficiency = e0 * median_i(sim_prefill_i / real_prefill_i)` -
  prefill is compute-bound, so its latency scales as `1 / compute_efficiency`;
- `decode_memory_bandwidth_scale = s0 * median_i(sim_step_i / real_step_i)`
  with `step = decode_ms / decode_tokens` - decode is HBM-bound, so its
  latency scales as `1 / decode_memory_bandwidth_scale`.

The median resists one noisy shape. Leave-one-shape-out refits on all shapes
but one, re-simulates the held-out shape with that fold's scalars, and reports
mean |error| per metric. Validation always re-simulates rather than assuming
the proportionality, because roofline phases mix compute and memory terms.

### Efficiency curve (`efficiency_curve.py`, `curve_profile.py`)

The simulator prices a prefill pass as `max(compute(e), weight_read)` with
`compute(e) = compute(1) / e`. For each measured prefill, `fit-curve` runs
the simulator at a scalar efficiency of 0.005 (compute dominates every
shape, so `compute(1) = 0.005 x sim`) and at 1.0 (a result above
`compute(1)` is the weight-read floor), both with the base profile's
`decode_memory_bandwidth_scale`. A shape measured below `(1 + margin) x
floor` (default margin 0.25, `--memory-bound-margin`) is weight-read bound
and skipped: its efficiency is not identifiable, and the curve clamps there.
Every other shape gives `efficiency = compute(1) / measured`; shapes with
equal `batch x prompt` take the median. Fewer than two compute-bound token
counts, or an efficiency above 1, is a `FitError`.

`render_curve_profile` inserts `compute_efficiency_curve` (and the frontend
keys) right after the base profile's `[calibration]` header, keeps every
other byte, and prepends a provenance comment; it refuses a base that already
sets those keys. `verify_curve_profile` loads the result in the simulator and
requires the JSON `calibration.compute_efficiency_curve` to echo the fitted
points.

### Frontend latency (`frontend.py`)

`load_frontend_ttfts` reads `median_ttft_ms` from `in<N>_out1.json`
(one-output-token runs; they have no TPOT, so the full serving parser does
not apply). Each prompt length's overhead is that TTFT minus the batch-1
static prefill of the same length from the curve run; a least-squares line
in prompt tokens (exact for two lengths, a constant for one) gives
`frontend_latency_us` and `frontend_latency_per_prompt_token_us`. Negative
terms are a `FitError`.

### KV capacity estimate (`kv_estimate.py`)

Used by `kv_budget` only when neither `serving.kv_cache_tokens` nor a
`server.log` capacity is available. vLLM 0.29 gives the KV cache
`utilization x visible memory - (weights + load overhead) - peak activation`,
where peak activation comes from a profiling pass over
`max_num_batched_tokens`. The model: visible = nominal HBM (GiB) - 0.44
GiB, load overhead 0.24 GiB, peak activation = 0.825 GiB + 1.31 x (2 x ffn
+ hidden) x dtype bytes x max_num_batched_tokens / tp, fitted from the
memory lines vLLM logged in five 3090 runs. It reproduces vLLM's logged
capacity within 10% in all five (82,864 tokens at 2048 batched tokens, 24,896
at 32,768); the previous estimate (utilization x HBM - weights) ignored the
activation reserve and was 9% to 262% high.

### Profile emission (`profile.py`)

Same schema as `examples/calibration_h100_vllm_llama31_70b.toml` and
`src/config/calibration_config.rs`: `[profile]` provenance (name, hardware,
model, dtype, `serving_stack = "vllm"`, `backend_version` and
driver/CUDA/NCCL from the probe when present, `environment_hash =
sha256(measured file)`, source = run dir, date = run date, notes with the
method and LOO errors, `kernel_settings`), `[valid_shape]` from the measured
envelope, `[calibration]` with the two scalars, and one `[[benchmarks]]` per
(shape, phase) with `measured_ms` and the in-sample fitted `predicted_ms`.
`verify_profile` loads it in the simulator, checks the JSON calibration block
reports the profile and its scalars, and requires predictions through the
profile to match the scalar run within 0.01%.

## Files

| file | role | key exports |
|---|---|---|
| `tools/lab/lab.py` | uv script entry point; subcommands `run`, `sim`, `calibrate`, `validate`, `report`, `curves`, `fit-curve`; `--cluster` overrides the lab's simulator cluster | `main`, `build_parser` |
| `tools/lab/labharness/efficiency_curve.py` | curve fit math | `CurveSample`, `curve_sample`, `fit_efficiency_curve`, `EfficiencyCurveFit`, `CurvePoint`, `SkippedSample`, `SkipReason`, `CURVE_BASE_EFFICIENCY`, `DEFAULT_MEMORY_BOUND_MARGIN` |
| `tools/lab/labharness/curve_profile.py` | curve profile TOML | `render_curve_profile`, `CurveProfileInputs` |
| `tools/lab/labharness/frontend.py` | frontend latency fit | `FrontendSample`, `FrontendFit`, `fit_frontend_latency`, `load_frontend_ttfts` |
| `tools/lab/labharness/kv_estimate.py` | vLLM KV-capacity estimate | `estimate_kv_tokens`, `KvCapacityInputs`, `VllmMemoryModel` |
| `tools/lab/labharness/spec.py` | typed specs and parsing | `parse_experiment`, `parse_lab`, `Experiment`, `Lab`, `Node`, `DockerLaunch`, `VenvLaunch`, `StaticBatchWorkload`, `ServingWorkload`, `Shape`, `ReferenceBatch`, `SimAdmission`, `rate_label`, `rate_text` |
| `tools/lab/labharness/commands.py` | pure plan generation | `build_plan`, `Plan`, `Step`, `Wait`, `Phase`, `bench_serve_args` |
| `tools/lab/labharness/runner.py` | plan execution, manifest | `Runner`, `dry_run`, `Executor`, `ProbeStatus`, `parse_probe` |
| `tools/lab/labharness/results.py` | measured-result parsers | `parse_static_lines`, `load_static`, `count_static_results`, `parse_bench_serve`, `load_bench_serve`, `parse_kv_cache_tokens`, `StaticMeasurement`, `ServeMeasurement`, `LatencyStats` |
| `tools/lab/labharness/simulate.py` | workload rendering and simulator runs | `SimRunner`, `sweep_static`, `sweep_serving`, `littles_law_batch` (phase-pipeline reference batch), `static_workload_toml`, `serving_workload_toml`, `kv_budget`, `Calibration` variants |
| `tools/lab/labharness/fitting.py` | fit math and error metrics | `fit_scalars`, `leave_one_out_folds`, `match`, `shape_errors`, `summarize`, `pct_error` |
| `tools/lab/labharness/evaluate.py` | orchestration of sims + fits | `calibrate_static`, `evaluate_static`, `evaluate_serving`, `verify_profile`, `fit_curve_static`, `verify_curve_profile` |
| `tools/lab/labharness/profile.py` | profile TOML | `render_profile`, `Provenance` |
| `tools/lab/labharness/report.py` | markdown tables and JSON | `static_table`, `serving_table`, `static_json`, `serving_json`, `fit_json` |
| `tools/lab/labharness/toml_emit.py` | byte-stable TOML values (mirrors the AISimulate converter) | `fmt_float`, `kv_lines`, `table` |
| `tools/lab/labharness/errors.py` | `LabError` hierarchy with exit codes | `SpecError`, `ResultParseError`, `RemoteCommandError`, `CompletionTimeoutError`, `SimulatorError`, `FitError`, `RunDirError` |
| `tools/lab/labharness/logging_setup.py` | key=value logging | `configure_logging`, `get_logger` |
| `tools/lab/remote/bench_latency.py` | static-batch benchmark (runs on the node) | JSON lines `lab.static_batch.v1`, `lab.done.v1` |
| `tools/lab/remote/probe_env.py` | environment probe (runs on the node) | JSON line `lab.probe.v1` |
| `tools/lab/labs/rtx3090.toml` | the 3-node RTX 3090 lab and its quirks | |
| `tools/lab/labs/rtx3090_cluster.toml` | simulator cluster for that lab | |
| `tools/lab/specs/*.toml` | experiment specs (static PP=1, static PP=2, serving PP=1) | |
| `tools/lab/tests/` | pytest suite, real `vllm bench serve` fixtures, golden dry-run output | |
| `tools/lab/pyproject.toml`, `uv.lock` | dev tooling (pytest, ruff, ty), pinned | |

## lab-runs/

`lab-runs/<date>-<name>/` holds raw measurements and everything derived from
them. Raw files are data: never edit them; add new files next to them.

Commit policy: commit everything except `sim-work/`. Raw results, manifests,
probes, logs, reports, profiles, and JSON outputs are small (a static-batch
run is a few KB; a serving run with logs is ~250 KB because `vllm bench serve`
is run without `--save-detailed`) and are the evidence behind
`docs/validation_ledger.md`. `sim-work/` holds regenerated simulator workload
TOMLs (transient) and is gitignored, as is the tool's `.venv/`.

Files the harness writes into a run dir:

| file | writer |
|---|---|
| `spec.toml`, `lab.toml`, `probe-<node>.json`, `manifest.json` | `run` |
| `measured.jsonl`, `bench.log` (static) / `rate_<R>.json`, `bench_rate_<R>.log`, `server.log` (serving) | `run` (collect) |
| `sim-<tag>-<calibration>.jsonl` / `.json` | `sim` |
| `calibration_profile-<tag>.toml`, `calibration-<tag>.json`, `report-calibration-<tag>.md` | `calibrate` |
| `validation-<tag>.json`, `report-validation-<tag>.md` | `validate` |
| `report.md` | `report` |

The two pre-harness runs keep their original file names (`real_pp1.jsonl`,
`real_pp2.jsonl`, `rate_<R>.json`), which is why `--measured` exists.

## Invariants

- Command generation is pure: same spec, date, and paths give byte-identical
  `--dry-run` output (golden-tested).
- No remote script is ever passed on a command line; every process pattern
  is bracketed.
- Every vLLM process gets an explicit HF_HOME; every venv process gets the
  venv first on PATH.
- A run never starts on a node whose probed vLLM version differs from the
  spec's pin, and teardown always runs.
- Static-batch completion requires `len(shapes)` tagged result records plus
  the done sentinel; untagged legacy records are accepted by the parser only
  when they carry every result field.
- Fit inputs are matched by exact shape and decode length, and the simulator
  must call every matched shape feasible.
- An emitted profile is only reported after the simulator has loaded it and
  reproduced the fitted predictions within 0.01%.
- `DEFAULT_COMPUTE_EFFICIENCY` in `fitting.py` must equal
  `SimulationCalibration::default().compute_efficiency` in
  `src/calibration.rs` (0.35).
- Errors everywhere are `(sim - measured) / measured`: positive means the
  simulator is slower than reality.

## Usage

```sh
cargo build --release
cd tools/lab

# print every command for an experiment (never touches the nodes)
uv run lab.py run specs/rtx3090_qwen7b_static_pp1.toml --dry-run
uv run lab.py run specs/rtx3090_qwen7b_serving_pp1.toml --dry-run

# execute it (coordinate GPU use first); results -> lab-runs/<today>-<name>/
uv run lab.py run specs/rtx3090_qwen7b_static_pp1.toml

# static batch: fit, validate, emit + verify a profile
uv run lab.py calibrate specs/rtx3090_qwen7b_static_pp1.toml --run-dir ../../lab-runs/<run>
# apply it to another run (e.g. PP=2) or to a serving sweep
uv run lab.py validate specs/rtx3090_qwen7b_static_pp2.toml --run-dir ../../lab-runs/<run2> \
    --profile ../../lab-runs/<run>/calibration_profile-<tag>.toml
uv run lab.py validate specs/rtx3090_qwen7b_serving_pp1.toml --run-dir ../../lab-runs/<run3> \
    --profile ../../lab-runs/<run>/calibration_profile-<tag>.toml
uv run lab.py report --run-dir ../../lab-runs/<run>

# tests and checks
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check labharness tests && uv run ty check lab.py --extra-search-path .
```

`lab.py` needs only the standard library, so `python3 tools/lab/lab.py ...`
works from the repo root as well.

### Reproducing the recorded runs

Efficiency curve, frontend latency, and the 2026-10-01 re-validation
(validation ledger entries 14-16):

```sh
R=lab-runs/2026-10-01-qwen7b-prefill-token-sweep
B=lab-runs/2026-09-28-static-batch/calibration_profile-pp1-recalibrated.toml
python3 tools/lab/lab.py fit-curve tools/lab/specs/rtx3090_qwen7b_prefill_token_sweep.toml \
    --run-dir $R --base-profile $B --tag curve-only
python3 tools/lab/lab.py fit-curve tools/lab/specs/rtx3090_qwen7b_prefill_token_sweep.toml \
    --run-dir $R --base-profile $B --frontend-dir lab-runs/2026-09-30-frontend --tag curve
# needs the base-commit binary for the "before" column (see the script's docstring)
python3 lab-runs/2026-10-01-structural-calibration/revalidate.py
```

```sh
R=lab-runs/2026-09-28-static-batch; S=tools/lab/specs
python3 tools/lab/lab.py calibrate $S/rtx3090_qwen7b_static_pp1.toml --run-dir $R --measured real_pp1.jsonl --tag pp1
python3 tools/lab/lab.py validate $S/rtx3090_qwen7b_static_pp2.toml --run-dir $R --measured real_pp2.jsonl \
    --default-calibration --tag pp2-default
python3 tools/lab/lab.py validate $S/rtx3090_qwen7b_static_pp2.toml --run-dir $R --measured real_pp2.jsonl \
    --profile $R/calibration_profile-pp1.toml --tag pp2-pp1-profile
python3 tools/lab/lab.py report --run-dir $R

R2=lab-runs/2026-09-30-serving-baseline
python3 tools/lab/lab.py validate $S/rtx3090_qwen7b_serving_pp1.toml --run-dir $R2 \
    --profile $R/calibration_profile-pp1.toml --tag serving-pp1-profile-iteration-engine
```

The `serving-pp1-profile-littles-law*` and `-ref1-uncapped` fragments in
`$R2` were produced by the phase-pipeline scheduler before the iteration
engine existed. `report.md` there is that pre-engine snapshot, and
`report-iteration-engine.md` holds the before/after comparison. Running
`lab.py report --run-dir $R2` again merges every fragment, old and new, into
`report.md`:

```sh
python3 tools/lab/lab.py report --run-dir $R2
```

## Known limits

- Serving TTFT at low load is under-predicted by 15-19% at 1-4 req/s with
  the curve + frontend profile (was 33-39%, `docs/validation_ledger.md`
  entry 15); the rest is in-load interference (arrivals waiting on in-flight
  steps) the engine does not capture.
- The efficiency curve was fitted on node0; node1 runs large prefills ~3%
  slower, so node1 prefills are 3-6% under-predicted with it.
- Simulated Poisson arrivals use the simulator's RNG, not vLLM's, so arrival
  sample paths differ; compare distributions, not individual requests.
- The static benchmark times whole `generate()` calls, so its prefill includes
  one sampled token and scheduler overhead; decode steps are derived by
  subtraction.
