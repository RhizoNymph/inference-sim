"""Byte-stable calibration-profile TOML emission."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from converter.errors import FitError
from converter.fitting import PhaseFit
from converter.opwalk import Sample
from converter.spec import GpuSpec, ModelSpec, Provenance

# ---------------------------------------------------------------------------
# TOML emission
# ---------------------------------------------------------------------------


def fmt_float(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        raise FitError(f"refusing to emit a non-finite number: {value}")
    text = f"{value:.{digits}g}"
    if "e" in text or "E" in text:
        mantissa, _, exponent = text.partition("e")
        if "." not in mantissa:
            mantissa = f"{mantissa}.0"
        return f"{mantissa}e{exponent}"
    if "." not in text:
        text = f"{text}.0"
    return text


def toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _kv_lines(pairs: Iterable[tuple[str, object]]) -> list[str]:
    lines: list[str] = []
    for key, value in pairs:
        if value is None:
            continue
        match value:
            case bool():
                lines.append(f"{key} = {'true' if value else 'false'}")
            case int():
                lines.append(f"{key} = {value}")
            case float():
                lines.append(f"{key} = {fmt_float(value)}")
            case str():
                lines.append(f"{key} = {toml_string(value)}")
            case _:
                raise FitError(f"cannot emit TOML for {key}: {type(value)!r}")
    return lines


def render_fit(fit: PhaseFit, source: str) -> list[str]:
    lines = ["[[fits]]"]
    lines += _kv_lines(
        [
            ("name", fit.name),
            ("target", fit.target),
            ("phase", fit.phase),
            ("kind", "latency"),
            ("model", "linear"),
            ("unit", "ms"),
            ("intercept", fit.intercept),
        ]
    )
    lines.append("features = [" + ", ".join(toml_string(f) for f in fit.features) + "]")
    lines.append(
        "coefficients = [" + ", ".join(fmt_float(c) for c in fit.coefficients) + "]"
    )
    lines.append("feature_ranges = [")
    for feature, low, high in fit.feature_ranges:
        lines.append(
            f"  {{ feature = {toml_string(feature)}, "
            f"min = {fmt_float(low)}, max = {fmt_float(high)} }},"
        )
    lines.append("]")
    lines += _kv_lines(
        [
            ("r_squared", fit.train.r_squared),
            ("adjusted_r_squared", fit.train.adjusted_r_squared),
            ("rmse", fit.train.rmse),
            ("rmse_pct", fit.train.rmse_pct),
            ("mean_abs_pct_error", fit.train.mean_abs_pct_error),
            ("max_abs_pct_error", fit.train.max_abs_pct_error),
            ("validation_rmse", fit.holdout.rmse),
            ("validation_rmse_pct", fit.holdout.rmse_pct),
            ("validation_mean_abs_pct_error", fit.holdout.mean_abs_pct_error),
            ("validation_max_abs_pct_error", fit.holdout.max_abs_pct_error),
            ("sample_count", fit.train.sample_count),
            ("validation_sample_count", fit.holdout.sample_count),
            ("source", source),
        ]
    )
    return lines


def render_benchmarks(
    fits: Sequence[PhaseFit],
    model: ModelSpec,
    hardware: str,
    source: str,
    limit: int,
) -> list[str]:
    entries: list[tuple[PhaseFit, Sample, float]] = []
    for fit in fits:
        for sample, predicted in zip(
            fit.holdout_samples, fit.holdout_predictions, strict=True
        ):
            entries.append((fit, sample, predicted))
    entries.sort(
        key=lambda e: (e[0].phase, e[1].tp, e[1].batch, e[1].prompt, e[1].decode)
    )
    per_phase = max(1, limit // max(1, len(fits)))
    selected: list[tuple[PhaseFit, Sample, float]] = []
    for fit in fits:
        phase_entries = [e for e in entries if e[0] is fit]
        stride = max(1, len(phase_entries) // per_phase)
        selected.extend(phase_entries[::stride][:per_phase])

    lines: list[str] = []
    for fit, sample, predicted in selected:
        name = (
            f"{fit.phase}-tp{sample.tp}-b{sample.batch}"
            f"-p{sample.prompt}-d{sample.decode}"
        )
        lines.append("")
        lines.append("[[benchmarks]]")
        lines += _kv_lines(
            [
                ("name", name),
                ("kind", "compute"),
                ("phase", fit.phase),
                ("hardware", hardware),
                ("model", model.name),
                ("dtype", model.dtype),
                ("batch_size", sample.batch),
                ("prompt_tokens", sample.prompt),
                ("decode_tokens", sample.decode),
                ("sequence_tokens", sample.prompt + sample.decode),
                ("tensor_ranks", sample.tp),
                ("pipeline_ranks", 1),
                ("expert_ranks", 1),
                ("data_ranks", 1),
                ("measured_ms", sample.target_ms),
                ("predicted_ms", max(predicted, 0.0)),
                ("source", source),
            ]
        )
    return lines


def render_profile(
    *,
    model: ModelSpec,
    gpu: GpuSpec,
    provenance: Provenance,
    efficiency: float,
    fits: Sequence[PhaseFit],
    benchmark_limit: int,
) -> str:
    hardware = f"{provenance.system} ({gpu.num_gpus_per_node} GPUs per node)"
    source = (
        f"aisimulate@{provenance.commit} {provenance.system}/{provenance.backend}/"
        f"{provenance.version}; composed from measured op tables"
    )
    lines = ["schema_version = 1", "", "[profile]"]
    lines += _kv_lines(
        [
            ("name", f"{provenance.system}-{provenance.backend}-{model.name}"),
            ("hardware", hardware),
            ("model", model.name),
            ("dtype", model.dtype),
            ("serving_stack", provenance.backend),
            ("backend_version", provenance.version),
            ("nccl_version", gpu.nccl_version),
            ("environment_hash", provenance.environment_hash),
            ("source", source),
            ("date", provenance.date),
            (
                "notes",
                "Phase latencies composed from AISimulate measured GEMM, "
                "context/generation attention and custom-allreduce tables; "
                "dense decoder, pipeline ranks pinned to 1.",
            ),
        ]
    )
    lines += ["", "[calibration]"]
    lines += _kv_lines(
        [
            ("compute_efficiency", efficiency),
            (
                "decode_memory_bandwidth_scale",
                gpu.mem_bw_empirical_scaling_factor,
            ),
        ]
    )
    for fit in fits:
        lines.append("")
        lines += render_fit(fit, source)
    lines += render_benchmarks(fits, model, hardware, source, benchmark_limit)
    return "\n".join(lines) + "\n"
