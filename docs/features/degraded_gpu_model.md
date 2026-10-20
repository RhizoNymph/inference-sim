# Degraded GPU Model (locked SM clocks)

## Scope

A first-class per-GPU clock state that slows a GPU's compute and HBM
bandwidth when its SM (graphics) clock is locked below the clock its
calibrated peaks correspond to, for example `nvidia-smi -lgc 1200,1200`. It
covers:

- Three optional GPU-profile keys in the cluster TOML, accepted wherever the
  other profile keys are (node-wide `[[nodes]]`, `[[node_groups]]`,
  `[[nodes.gpus]]` entries, and `[[nodes.gpu_profile_overrides]]`):
  - `sustained_sm_clock_mhz`: the SM clock the card actually holds under
    sustained load, which is what the calibrated `peak_f16_tflops` x
    `compute_efficiency` corresponds to. On boost-clocked cards this is
    below the maximum boost clock (RTX 3090: about 1835 MHz sustained versus
    a 2100 MHz maximum).
  - `bandwidth_knee_sm_clock_mhz`: the SM clock below which achievable DRAM
    bandwidth starts to fall (see the bandwidth model below). Optional.
  - `locked_sm_clock_mhz`: the per-GPU lock. It requires a sustained clock on
    the same profile.
- Two multipliers derived from the lock:
  - compute factor `min(1, locked / sustained)`;
  - bandwidth factor `min(1, locked / knee)`, or 1 without a knee.
- Per-stage application: a pipeline stage runs as fast as its slowest GPU at
  each term (tensor/expert ranks are in lock-step), and a locked GPU slows
  only the stage it serves. Disaggregated prefill and decode workers have
  their own placements, so a locked decode GPU slows only the decode worker.
- Placement: among GPUs of equal nominal capability, unlocked GPUs are placed
  first (capability ordering uses the clock-adjusted peak and bandwidth).
- Evidence: a scored config whose placement includes a locked GPU carries the
  approximation `gpu_clock_degradation` (category `hardware`, scope
  `gpu_clock`) naming each degraded stage, its locked GPUs and clocks, both
  factors, and the stage's compute-latency multiplier. The JSON cluster dump
  reports a per-GPU `clock` object (`sustained_sm_clock_mhz`,
  `bandwidth_knee_sm_clock_mhz`, `locked_sm_clock_mhz`, `compute_factor`,
  `bandwidth_factor`) or `null`.

### Bandwidth model

Achievable DRAM bandwidth follows a memory-system roofline in the SM clock:

```
bandwidth(f) = min(dram_limit, k * f)        knee f_k = dram_limit / k
factor(locked) = bandwidth(locked) / bandwidth(full) = min(1, locked / f_k)
```

Rationale: the SMs, L2 and crossbar that issue and absorb DRAM requests run
on the graphics clock, so below some clock they cannot keep enough requests
in flight to saturate DRAM, and bandwidth falls in proportion to the clock;
above it DRAM is the limit and the SM clock does not matter. The knee must be
at or below the sustained clock (config error otherwise), because the
calibrated bandwidth was measured at full clock and the factor is relative to
it. A useful property of this form is that it does not need to know which
clock decode actually runs at when unlocked (decode is memory-bound and draws
less power, so it may boost higher than prefill): any full clock above the
knee gives factor 1.

A power law (`factor = (locked / sustained)^a`) fits one degraded point
equally well; the knee form was chosen because it saturates (it never
predicts more bandwidth above the sustained clock) and has a physical reading.
The two data points available cannot distinguish them.

### RTX 3090 values (validation ledger 22)

Both clocks are **fitted** from the single 1200 MHz run:

- `sustained_sm_clock_mhz = 1834` = 1200 x median prefill slowdown (1.528x);
- `bandwidth_knee_sm_clock_mhz = 1564` = 1200 x median decode-step slowdown
  (1.303x).

Mean |error| on the five locked shapes: prefill 13.8% -> 4.8% (1.1% with the
efficiency-curve profile), decode step 22.8% -> 3.0%, end-to-end 9.6% -> 2.6%;
leave-one-shape-out (clocks fitted on four shapes): 4.9% / 4.3% / 3.2%.

## Non-scope

- Thermal or power throttling dynamics, clock ramps, and per-kernel clock
  behaviour: the state is a fixed lock for the whole run.
- Memory-clock locks (`-lmc`) or HBM degradation; use the scenario overlay
  `topology.degraded_gpus` (`hbm_bandwidth_scale`, `compute_scale`) for
  arbitrary scaling instead.
- Clock locks in run-scenario overlays (scenario `topology.degraded_gpus`
  keeps its raw scales; clock locks are set in the cluster TOML).
- Batch-dependent clock sensitivity of decode: one bandwidth factor per GPU.
  Measured decode slowed 1.26x at batch 1 and 1.34x at batch 8, so the model
  is +4-5% at batch 1 and -3% at batch 8 (decode kernels at larger batches
  have clock-bound, non-DRAM work the roofline does not separate).
- Heterogeneous base peaks across stages keep the existing behaviour (the
  placement-wide minimum nominal peak/bandwidth prices every stage); only
  the clock factor is applied per stage.
- Collectives and transfers: a locked SM clock does not change link or NCCL
  timing in the model.

## Data/control flow

1. Config: `src/config/cluster.rs` builds each GPU's profile through
   `gpu_profile_override_from_base_config`, which calls
   `merge_gpu_clock` (`src/config/gpu_clock_config.rs`). Each override layer
   (node-wide fields, then a `gpus` entry or a `gpu_profile_overrides`
   entry) replaces only the clock keys it sets and inherits the rest, then
   the merged keys are validated into a typed `GpuClock` stored in
   `GpuProfile::clock`. Errors (`ConfigError`) name the offending key path.
2. Types: `GpuClock` = `GpuClockModel` (sustained `SmClockMhz` +
   `BandwidthClockSensitivity::{Insensitive, Knee}`) + `GpuClockState::
   {Unlocked, Locked}`. A lock therefore always has a sustained clock;
   `GpuProfile::clock_degradation()` returns `ClockDegradation::NOMINAL`
   without a clock or when unlocked.
3. Static solver (`Solver::score_config_with_options`, `src/solver.rs`):
   after `estimate_compute_latency_s` prices the phase on the placement's
   nominal hardware (including any calibration fit),
   `StageClockDegradation::from_placement` (`src/solver/clock_degradation.rs`)
   takes the slowest factors over each pipeline stage's GPUs, and
   `Solver::stage_latency_scales` computes, per stage,
   `analytical(peak x compute, bandwidth x bandwidth) / analytical(nominal)`
   with the same roofline primitives (`prefill_baseline_s`,
   `decode_compute_latency_s`, no fits). `build_operation_trace`
   (`src/solver/operations.rs`) multiplies each layer's compute duration by
   its stage's scale, so the scheduler sees slow stages only where they are.
   The approximation `gpu_clock_degradation` is appended to the config's
   approximations.
4. Serving step cost (`IterationCostModel::new`, `src/solver/step_cost.rs`):
   `step_stage_clocks` records each stage's layer share and factors (or
   `None` when nominal). `step_latency` then prices one forward pass as
   `sum_s share_s * max(compute / c_s, memory / b_s)` instead of
   `max(compute, memory)`; communication and overhead are unchanged.
5. Placement (`Solver::compare_gpu_addrs`, `src/solver/placement.rs`) ranks
   GPUs by clock-adjusted peak then bandwidth.
6. Output: approximations flow into text/JSON/CSV like every other
   approximation; `src/cli/topology.rs::gpu_clock_json` writes the per-GPU
   clock object in the cluster JSON.

## Related files

- `src/types/gpu_clock.rs`: `SmClockMhz`, `BandwidthClockSensitivity`,
  `GpuClockModel` (`new`, `degradation_at`, `sustained`, `bandwidth_knee`),
  `GpuClockState`, `GpuClock` (`degradation`, `locked_clock`),
  `ClockDegradation` (`NOMINAL`, `compute`, `bandwidth`, `slowest`,
  `is_nominal`), `GpuClockError`.
- `src/types/gpu.rs`: `GpuProfile::clock`, `GpuProfile::clock_degradation`.
- `src/config/gpu_clock_config.rs`: `GpuClockOverrideConfig`,
  `merge_gpu_clock`; `src/config/sections.rs` (the three keys on node, node
  group, gpu entry and override sections); `src/config/cluster.rs`
  (wiring).
- `src/solver/clock_degradation.rs`: `StageClockDegradation`,
  `Solver::stage_latency_scales`, `GPU_CLOCK_DEGRADATION`.
- `src/solver/operations.rs`: per-stage compute scaling in
  `build_operation_trace`.
- `src/solver/step_cost.rs`: `StepStageClock`, `step_stage_clocks`,
  `degraded_forward_s`.
- `src/solver/placement.rs`: clock-aware `compare_gpu_addrs`.
- `src/cli/topology.rs`: `gpu_clock_json`.
- Tests: `src/types/gpu_clock.rs` (unit), `src/config/gpu_clock_config_tests.rs`,
  `src/solver/clock_degradation_tests.rs`.
- Validation: `lab-runs/2026-10-01-degraded-gpu-sim/` (`predict.py`,
  `results.json`, `summary.md`), docs/validation_ledger.md entry 22.

## Invariants and constraints

- Without `locked_sm_clock_mhz` on any placed GPU, scores, operation traces,
  approximations and step latencies are bit-identical to a cluster without
  clock keys (tested; the validation script also checks it on the release
  binary).
- Both factors lie in `(0, 1]`; a lock at or above the sustained clock is
  nominal for compute, and a lock at or above the knee is nominal for
  bandwidth.
- `bandwidth_knee_sm_clock_mhz <= sustained_sm_clock_mhz`; a knee or a lock
  without a sustained clock is a config error; every clock is positive and
  finite.
- A stage's factors are the minimum over its GPUs, taken per term.
- A locked GPU never slows another pipeline stage or another serving worker.
- Calibration fits keep their role: the fitted phase latency is scaled by
  the analytical degraded/nominal ratio, never replaced.
- The RTX 3090 clocks are fitted from one locked clock; other clocks and
  other GPUs are unvalidated.
