# Calibration Fits

## Scope

- Parsing `[[fits]]` entries from a calibration profile TOML into
  `CalibrationFittedModel` values.
- The feature dictionary a fit's `features` names are resolved against, built
  per request/config by `calibration_feature_values`, including the derived
  basis features that let a linear-in-coefficients fit express quadratic
  attention cost, mean-decode-context KV-read cost, and tensor-parallel `1/tp`
  shape.
- Selecting an applicable fit for a phase/target and evaluating it
  (`fit_matches`, `evaluate_fit`, `evaluate_value_fit`).
- Overriding the analytical roofline phase latency in the solver with the fit
  prediction, and reporting the fit application (feature values, ranges,
  applicability status, confidence, uncertainty) as evidence.

## Non-scope

- Fitting itself. The simulator never regresses measured data; coefficients are
  produced externally and supplied in the profile.
- Non-linear model kinds. `fit_matches` accepts only `linear`,
  `linear_regression`, `ols`, and `ordinary_least_squares`; anything else is
  ignored. Curvature is expressed through derived basis features, not through a
  non-linear model form.
- Calibration gates, coverage reports, applicability warnings, and residual
  reporting (`src/config/calibration_config.rs` and `src/cli/calibration.rs`),
  which consume fit evidence but do not produce predictions.
- The serving-metric fit feature dictionary in `src/serving/metric_fits.rs`,
  which is built separately from `serving_metric_fit_features` and does not
  include the solver basis features.

## Data and control flow

1. **TOML** — a calibration profile file contains `[[fits]]` tables with
   `target`, optional `phase`, `model`, optional `unit`, optional `intercept`,
   `features`, `coefficients`, optional `feature_ranges`, and optional quality
   and provenance metadata (`r_squared`, `rmse*`, `validation_*`,
   `confidence_interval*`, `sample_count`, `source`, `notes`).
2. **Config parsing** — `load_calibration_profile_path` →
   `parse_calibration_fits` (`src/config/calibration_config.rs`) validates that
   `target` and `model` are non-empty, that `features` and `coefficients` are
   non-empty and equal length, and that numeric metadata is finite (and
   non-negative where required). The result is a `CalibrationProfileMetadata`
   carrying `fits: Vec<CalibrationFittedModel>`.
3. **Options** — `cli` passes the profile through `SolverOptions`
   (`calibration_profile`) into `Solver::score_config_with_options`, and through
   `ServingSolverOptions` into the serving search.
4. **Feature dictionary** — `Solver::fitted_phase_latency`
   (`src/solver/network_cost.rs`) calls `calibration_feature_values`
   (`src/solver/calibration_fits.rs`) with the `ModelSpec`, `InferenceRequest`,
   `ParallelismConfig`, and optional phase-specific extra features (for example
   KV transfer features). Every name is normalized through `normalize_fit_name`
   before insertion, and non-finite values are dropped.
5. **Selection** — `fitted_phase_latency` iterates `profile.fits` with
   `find_map`: the **first** fit for which `fit_matches(fit, phase, targets)`
   holds and `evaluate_fit` returns `Some` wins. `fit_matches` requires a linear
   model kind, a matching `phase` when the fit declares one, and a `target` in
   the phase's accepted target list (for example
   `["prefill_ms", "prefill_latency_ms", "prefill_s"]` for prefill,
   `["decode_ms", "decode_latency_ms", "decode_s"]` for decode).
6. **Evaluation** — `evaluate_fit` computes
   `prediction = intercept + Σ coefficient_i · feature_i`. A feature name that
   is absent from the dictionary makes the whole evaluation return `None`, so
   the next fit (if any) is tried and otherwise the analytical baseline stands.
   A non-finite or non-positive prediction is also rejected. The prediction is
   converted to seconds by `fit_value_to_seconds`, which honours `unit`
   (`us`/`ms`/`s`) and otherwise infers the unit from the target name suffix.
7. **Evidence** — each feature gets a `CalibrationFitFeatureValue` with its
   value, coefficient, declared range, and `in_range`/`extrapolated`/`unbounded`
   status. The fit-level `applicability_status` is `extrapolated` if any feature
   is out of range, `interpolated` if every feature has a range and none is out,
   `partially_bounded` if only some features are ranged, and `unbounded` if none
   is. `fit_confidence_score` and `fit_uncertainty` derive confidence and
   uncertainty, preferring confidence-interval metadata, then validation error,
   then training-fit error.
8. **Phase override** — `Solver::estimate_compute_latency_s` (`src/solver.rs`)
   uses the fit's seconds in place of the roofline phase estimate and records
   the `CalibrationFitApplication` in `ScoredParallelismConfig::calibration_fits`.
   The phase latency then feeds `build_operation_trace` and the scheduler, whose
   makespan is `estimated_latency_s`. Serving reaches the same evaluator through
   `Solver::fitted_latency_from_features` /
   `Solver::fitted_value_from_features` with a pre-built feature map.

## Basis feature dictionary

Let `b = batch_size`, `s = prompt_tokens`, `d = decode_tokens`,
`tp = tensor_ranks` (each clamped with `.max(1)`), and
`ctx = s + (d + 1) / 2` the mean decode context length over the request's
decode steps.

Existing linear terms: `batch_size`/`batch`, `prompt_tokens`/`prefill_tokens`/
`effective_prefill_tokens`, `decode_tokens`/`output_tokens`,
`sequence_tokens`/`max_sequence_tokens`, `batch_tokens`/`prefill_batch_tokens`
(`b·s`), `decode_batch_tokens` (`b·d`), `tensor_ranks`/`tp`,
`pipeline_ranks`/`pp`, `expert_ranks`/`ep`, `data_ranks`/`dp`, `total_ranks`,
and model-shape fields (`parameters_gb`, `parameter_count_billion`, `layers`,
`hidden_size`, `attention_heads`, `kv_heads`, `dtype_bytes`, `kv_dtype_bytes`).

Derived basis features (v1):

| Feature | Definition | Physical term it carries |
| --- | --- | --- |
| `prompt_tokens_squared` | `s²` | per-sequence causal attention work |
| `batch_prompt_tokens_squared` | `b·s²` | batched prefill attention work |
| `decode_context_tokens` | `ctx` | mean KV context read per decode step |
| `batch_decode_context_tokens` | `b·ctx` | batched context per decode step |
| `decode_batch_context_tokens` | `b·d·ctx` | total decode KV-cache reads |
| `inv_tensor_ranks` | `1/tp` | per-rank share of a fixed cost |
| `batch_prompt_tokens_per_tensor_rank` | `b·s/tp` | prefill dense GEMM work |
| `batch_prompt_tokens_squared_per_tensor_rank` | `b·s²/tp` | prefill causal attention |
| `decode_tokens_per_tensor_rank` | `d/tp` | decode weight reads |
| `decode_batch_tokens_per_tensor_rank` | `b·d/tp` | decode GEMM work |
| `decode_batch_context_tokens_per_tensor_rank` | `b·d·ctx/tp` | decode KV-cache reads and attention |

The `_per_tensor_rank` composites give a single fit the physically correct
`1/tp` shape, so one fitted model can span a tensor-parallel sweep while
`model = "linear"` stays true: the prediction remains linear in its
coefficients even though it is quadratic in prompt length.

## Related files

| File | Role |
| --- | --- |
| `src/solver/calibration_fits.rs` | Feature dictionary (`calibration_feature_values`), fit matching (`fit_matches`), evaluation (`evaluate_fit`, `evaluate_value_fit`), range/confidence/uncertainty helpers, `normalize_fit_name`. |
| `src/solver/network_cost.rs` | `fitted_phase_latency`, `fitted_latency_from_features`, `fitted_value_from_features`: first-matching-fit lookup via `find_map`. |
| `src/solver.rs` | `estimate_compute_latency_s` and `decode_compute_latency_s` apply phase overrides; `ScoredParallelismConfig::calibration_fits` carries the evidence; `mod tests` holds the fit tests. |
| `src/config/calibration_config.rs` | `parse_calibration_fits`, profile loading, gates, coverage, applicability warnings. |
| `src/config/sections.rs` | Raw serde sections for `[[fits]]` and `feature_ranges`. |
| `src/solver/operations.rs` | Consumes the (possibly fitted) phase latency when building the operation trace. |
| `src/serving/metric_fits.rs` | Serving-scope fits; builds its own feature map and reuses the same evaluator. |
| `src/cli/json_output.rs`, `src/cli/csv.rs`, `src/cli/presentation.rs` | Render fit applications, feature values, statuses, and uncertainty. |
| `examples/calibration_h100_a100.toml` | Example profile with prefill, decode, and KV-transfer fits. |

## Invariants and constraints

- **Linear in coefficients only.** `fit_matches` rejects any `model` other than
  `linear`, `linear_regression`, `ols`, `ordinary_least_squares`. Curvature must
  be expressed by a derived basis feature, never by a new model form.
- **Prediction form is fixed**: `intercept + Σ coefficient_i · feature_i`, with
  `features[i]` paired positionally with `coefficients[i]`; config parsing
  enforces equal, non-empty lengths.
- **Name normalization**: every feature, target, phase, unit, and model name is
  compared after `normalize_fit_name` (ASCII alphanumerics lowercased, every
  other character mapped to `_`), so `Prompt Tokens`, `prompt-tokens`, and
  `prompt_tokens` are the same name.
- **Unknown feature name means the fit is silently not applied.** `evaluate_fit`
  returns `None` when a name is missing from the dictionary; evaluation falls
  through to the next matching fit and otherwise to the analytical baseline. No
  error is raised, so feature names in profiles must match this dictionary
  exactly.
- **Feature names are a frozen contract.** Profiles and external fitting tools
  depend on them; names may be added, but not renamed or redefined.
- **First matching fit wins** (`find_map` order = profile order). Later fits for
  the same phase/target are unreachable.
- **Clamping**: `batch_size`, `prompt_tokens`, `decode_tokens`,
  `max_sequence_tokens`, and every parallel-rank count are `.max(1)`-clamped
  before deriving features, so `1/tp` and the composites are always finite and
  positive.
- **Non-finite features are dropped** by `insert_feature`, which turns a NaN or
  infinite derived value into a missing name (and therefore an unapplied fit)
  rather than a poisoned prediction.
- **Non-positive or non-finite predictions are rejected**, both before and after
  unit conversion, so a fit can never produce a zero or negative latency.
- **Extra features win ties**: caller-supplied `extra_features` are inserted
  last and overwrite same-named dictionary entries.
