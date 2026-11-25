"""Calibration-profile TOML emission for a static-batch scalar fit.

Schema: the one src/config/calibration_config.rs loads and
examples/calibration_h100_vllm_llama31_70b.toml demonstrates - `[profile]`
provenance, `[valid_shape]`, `[calibration]` scalars, and one
`[[benchmarks]]` entry per measured (shape, phase) with `measured_ms` and the
fitted in-sample `predicted_ms`. No `[[fits]]`: this profile calibrates the
analytical roofline through scalars rather than replacing phases.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from labharness.fitting import ErrorSummary, ScalarFit
from labharness.results import StaticMeasurement
from labharness.simulate import StaticSimRow
from labharness.spec import Experiment
from labharness.toml_emit import kv_lines, table


@dataclass(frozen=True, slots=True)
class Provenance:
    run_dir: str  # repo-relative lab-runs directory
    date: str  # measurement date, ISO
    backend_version: str
    driver_version: str | None
    cuda_version: str | None
    nccl_version: str | None
    environment_hash: str | None


def _benchmark(
    exp: Experiment,
    measured: StaticMeasurement,
    predicted: StaticSimRow,
    phase: str,
    source: str,
) -> list[str]:
    shape = measured.shape
    match phase:
        case "prefill":
            decode, measured_ms, predicted_ms = 1, measured.prefill_ms, predicted.prefill_ms
        case "decode":
            decode, measured_ms, predicted_ms = (
                measured.decode_tokens,
                measured.decode_ms,
                predicted.decode_ms,
            )
        case _:
            decode, measured_ms, predicted_ms = (
                measured.decode_tokens,
                measured.end_to_end_ms,
                predicted.end_to_end_ms,
            )
    par = exp.parallelism
    return table(
        "benchmarks",
        [
            ("name", f"{phase}-tp{par.tp}-pp{par.pp}-b{shape.batch}-p{shape.prompt}-d{decode}"),
            ("kind", "compute"),
            ("phase", phase),
            ("hardware", exp.lab.hardware),
            ("model", exp.model.name),
            ("dtype", exp.model.dtype),
            ("batch_size", shape.batch),
            ("prompt_tokens", shape.prompt),
            ("decode_tokens", decode),
            ("sequence_tokens", shape.prompt + decode),
            ("tensor_ranks", par.tp),
            ("pipeline_ranks", par.pp),
            ("expert_ranks", 1),
            ("data_ranks", 1),
            ("measured_ms", measured_ms),
            ("predicted_ms", predicted_ms),
            ("source", source),
        ],
        array=True,
    )


def render_profile(
    *,
    exp: Experiment,
    fit: ScalarFit,
    measured: Sequence[StaticMeasurement],
    fitted_rows: Sequence[StaticSimRow],
    loo: ErrorSummary,
    provenance: Provenance,
) -> str:
    par = exp.parallelism
    source = f"{provenance.run_dir}; vllm {provenance.backend_version} static-batch medians"
    lines = ["schema_version = 1", ""]
    lines += table(
        "profile",
        [
            ("name", f"{exp.lab.name}-vllm-{exp.model.name}-tp{par.tp}-pp{par.pp}"),
            ("hardware", exp.lab.hardware),
            ("model", exp.model.name),
            ("dtype", exp.model.dtype),
            ("serving_stack", "vllm"),
            ("backend_version", provenance.backend_version),
            ("driver_version", provenance.driver_version),
            ("cuda_version", provenance.cuda_version),
            ("nccl_version", provenance.nccl_version),
            ("environment_hash", provenance.environment_hash),
            ("source", source),
            ("date", provenance.date),
            (
                "notes",
                "Two-scalar fit by tools/lab: compute_efficiency = "
                f"{fit.base_compute_efficiency:g} * median(sim/real prefill), "
                "decode_memory_bandwidth_scale = "
                f"{fit.base_decode_memory_bandwidth_scale:g} * median(sim/real decode step) over "
                f"{len(fit.shapes)} shapes; leave-one-shape-out mean |error| "
                f"prefill {loo.prefill_mean_abs_pct:.2f}%, decode step "
                f"{loo.decode_step_mean_abs_pct:.2f}%, end-to-end {loo.end_to_end_mean_abs_pct:.2f}%.",
            ),
        ],
    )
    kernel_settings = [
        f"prefix_caching={'on' if exp.engine.enable_prefix_caching else 'off'}",
        f"cuda_graphs={'off' if exp.engine.enforce_eager else 'on'}",
        f"tensor_ranks={par.tp}",
        f"pipeline_ranks={par.pp}",
        f"max_num_batched_tokens={exp.engine.max_num_batched_tokens}",
    ]
    lines += kv_lines([("kernel_settings", kernel_settings)])

    shapes = [m.shape for m in measured]
    decodes = [m.decode_tokens for m in measured]
    lines += [
        "",
        *table(
            "valid_shape",
            [
                ("min_batch_size", min(s.batch for s in shapes)),
                ("max_batch_size", max(s.batch for s in shapes)),
                ("min_prompt_tokens", min(s.prompt for s in shapes)),
                ("max_prompt_tokens", max(s.prompt for s in shapes)),
                ("min_decode_tokens", min(decodes)),
                ("max_decode_tokens", max(decodes)),
                ("min_sequence_tokens", min(s.prompt for s in shapes) + min(decodes)),
                ("max_sequence_tokens", max(s.prompt for s in shapes) + max(decodes)),
            ],
        ),
    ]
    lines += [
        "",
        *table(
            "calibration",
            [
                ("compute_efficiency", fit.compute_efficiency),
                ("decode_memory_bandwidth_scale", fit.decode_memory_bandwidth_scale),
            ],
        ),
    ]
    rows = {row.shape: row for row in fitted_rows}
    for m in sorted(measured, key=lambda item: item.shape):
        for phase in ("prefill", "decode", "end_to_end"):
            lines += ["", *_benchmark(exp, m, rows[m.shape], phase, source)]
    return "\n".join(lines) + "\n"
