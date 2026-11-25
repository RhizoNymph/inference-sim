"""Orchestration: simulate a measured run, fit, validate, and verify a profile.

This is the only module that combines simulator runs with fitting; the math
lives in fitting.py and the TOML in profile.py.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from labharness.errors import FitError, SimulatorError
from labharness.fitting import (
    ErrorSummary,
    ScalarFit,
    ShapeErrors,
    fit_scalars,
    leave_one_out_folds,
    match,
    shape_errors,
    summarize,
)
from labharness.logging_setup import get_logger
from labharness.profile import Provenance, render_profile
from labharness.results import ServeMeasurement, StaticMeasurement, number_or_none
from labharness.simulate import (
    Calibration,
    DefaultCalibration,
    ProfileCalibration,
    ScalarCalibration,
    ServeSimResult,
    SimPhase,
    SimRunner,
    StaticSimRow,
    static_workload_toml,
    sweep_serving,
    sweep_static,
)
from labharness.spec import Experiment, ServingWorkload, StaticBatchWorkload

# ---------------------------------------------------------------------------
# Static batch
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StaticEvaluation:
    label: str
    calibration: str
    pairs: tuple[tuple[StaticMeasurement, StaticSimRow], ...]
    errors: tuple[ShapeErrors, ...]
    summary: ErrorSummary


def _evaluation(
    label: str, calibration: str, pairs: Sequence[tuple[StaticMeasurement, StaticSimRow]]
) -> StaticEvaluation:
    ordered = tuple(sorted(pairs, key=lambda pair: pair[0].shape))
    errors = tuple(shape_errors(m, s) for m, s in ordered)
    return StaticEvaluation(label, calibration, ordered, errors, summarize(errors))


def _static_workload(exp: Experiment) -> StaticBatchWorkload:
    if not isinstance(exp.workload, StaticBatchWorkload):
        raise FitError("experiment is not a static-batch experiment", context={"spec": str(exp.path)})
    return exp.workload


def check_measured_matches_spec(exp: Experiment, measured: Sequence[StaticMeasurement]) -> None:
    wl = _static_workload(exp)
    for m in measured:
        if (m.tp, m.pp) != (exp.parallelism.tp, exp.parallelism.pp):
            raise FitError(
                "measured parallelism differs from the spec",
                context={"shape": m.shape.label, "measured": f"tp{m.tp}pp{m.pp}"},
            )
        if m.decode_tokens != wl.decode_tokens:
            raise FitError("measured decode tokens differ from the spec", context={"shape": m.shape.label})


async def evaluate_static(
    runner: SimRunner,
    exp: Experiment,
    measured: Sequence[StaticMeasurement],
    calibration: Calibration,
    label: str,
) -> StaticEvaluation:
    wl = _static_workload(exp)
    rows = await sweep_static(
        runner, exp.model, exp.parallelism, [m.shape for m in measured], wl.decode_tokens, calibration
    )
    matched = match(measured, rows)
    return _evaluation(label, calibration.label, [(m.measured, m.simulated) for m in matched])


@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    fit: ScalarFit
    folds: tuple[tuple[str, ScalarFit], ...]
    default: StaticEvaluation
    fitted: StaticEvaluation
    loo: StaticEvaluation
    profile_text: str


async def calibrate_static(
    runner: SimRunner,
    exp: Experiment,
    measured: Sequence[StaticMeasurement],
    provenance: Provenance,
    tag: str,
) -> CalibrationOutcome:
    check_measured_matches_spec(exp, measured)
    log = get_logger()
    default = await evaluate_static(runner, exp, measured, DefaultCalibration(), f"{tag} default calibration")
    matched = match(measured, [s for _, s in default.pairs])
    fit = fit_scalars(matched)
    log.info(
        "fitted scalars",
        extra={
            "tag": tag,
            "compute_efficiency": round(fit.compute_efficiency, 5),
            "decode_memory_bandwidth_scale": round(fit.decode_memory_bandwidth_scale, 5),
        },
    )
    fitted = await evaluate_static(runner, exp, measured, _scalars(fit), f"{tag} fitted (in-sample)")

    folds = leave_one_out_folds(matched)
    by_shape = {m.shape: m for m in measured}
    loo_pairs: list[tuple[StaticMeasurement, StaticSimRow]] = []
    async with asyncio.TaskGroup() as group:
        tasks = [
            (
                shape,
                group.create_task(
                    sweep_static(
                        runner, exp.model, exp.parallelism, [shape],
                        by_shape[shape].decode_tokens, _scalars(fold),
                    )
                ),
            )
            for shape, fold in folds
        ]  # fmt: skip
    for shape, task in tasks:
        loo_pairs.append((by_shape[shape], task.result()[0]))
    loo = _evaluation(f"{tag} leave-one-shape-out", "per-fold scalars", loo_pairs)
    log.info(
        "leave-one-shape-out",
        extra={
            "tag": tag,
            "prefill_pct": round(loo.summary.prefill_mean_abs_pct, 3),
            "decode_step_pct": round(loo.summary.decode_step_mean_abs_pct, 3),
            "end_to_end_pct": round(loo.summary.end_to_end_mean_abs_pct, 3),
        },
    )
    profile_text = render_profile(
        exp=exp,
        fit=fit,
        measured=measured,
        fitted_rows=[s for _, s in fitted.pairs],
        loo=loo.summary,
        provenance=provenance,
    )
    return CalibrationOutcome(
        fit=fit,
        folds=tuple((shape.label, fold) for shape, fold in folds),
        default=default,
        fitted=fitted,
        loo=loo,
        profile_text=profile_text,
    )


def _scalars(fit: ScalarFit) -> ScalarCalibration:
    return ScalarCalibration(fit.compute_efficiency, fit.decode_memory_bandwidth_scale)


@dataclass(frozen=True, slots=True)
class ProfileCheck:
    profile_name: str
    compute_efficiency: float
    decode_memory_bandwidth_scale: float
    applicability_status: str
    max_prediction_drift_pct: float


async def verify_profile(
    runner: SimRunner,
    exp: Experiment,
    profile_path: Path,
    fitted: StaticEvaluation,
) -> ProfileCheck:
    """Load the emitted profile in the simulator and confirm it reproduces the fit.

    Checks that (a) the simulator accepts the file, (b) it reports the profile
    and its scalars in the JSON calibration block, and (c) predictions through
    the profile match the in-sample scalar predictions (the profile rounds to
    6 significant digits, so drift must stay below 0.01%).
    """
    wl = _static_workload(exp)
    calibration = ProfileCalibration(profile_path)
    first_shape = fitted.pairs[0][0].shape
    payload = await runner.run(
        f"{calibration.label}-check",
        static_workload_toml(
            exp.model, exp.parallelism, first_shape, wl.decode_tokens, SimPhase.END_TO_END, calibration
        ),
        asyncio.Semaphore(1),
    )
    block = payload.get("calibration")
    if not isinstance(block, dict) or not isinstance(block.get("profile"), dict):
        raise SimulatorError("simulator did not report the calibration profile", workload=str(profile_path))
    through_profile = await evaluate_static(
        runner, exp, [m for m, _ in fitted.pairs], calibration, "profile check"
    )
    drift = 0.0
    for (_, expected), (_, got) in zip(fitted.pairs, through_profile.pairs, strict=True):
        for a, b in (
            (expected.prefill_ms, got.prefill_ms),
            (expected.decode_ms, got.decode_ms),
            (expected.end_to_end_ms, got.end_to_end_ms),
        ):
            drift = max(drift, abs(b - a) / a * 100.0)
    if not math.isfinite(drift) or drift > 0.01:
        raise SimulatorError(
            f"profile predictions drift {drift:.4f}% from the fitted scalars", workload=str(profile_path)
        )
    profile = block["profile"]
    compute_efficiency = number_or_none(block, "compute_efficiency")
    bandwidth_scale = number_or_none(block, "decode_memory_bandwidth_scale")
    if compute_efficiency is None or bandwidth_scale is None:
        raise SimulatorError("calibration block lacks the fitted scalars", workload=str(profile_path))
    return ProfileCheck(
        profile_name=str(profile.get("name")),
        compute_efficiency=compute_efficiency,
        decode_memory_bandwidth_scale=bandwidth_scale,
        applicability_status=str(block.get("applicability_status")),
        max_prediction_drift_pct=drift,
    )


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServingPair:
    measured: ServeMeasurement
    simulated: ServeSimResult


@dataclass(frozen=True, slots=True)
class ServingEvaluation:
    label: str
    calibration: str
    pairs: tuple[ServingPair, ...]


async def evaluate_serving(
    runner: SimRunner,
    exp: Experiment,
    measured: Sequence[ServeMeasurement],
    calibration: Calibration,
    label: str,
    *,
    measured_kv_tokens: int | None = None,
) -> ServingEvaluation:
    if not isinstance(exp.workload, ServingWorkload):
        raise FitError("experiment is not a serving experiment", context={"spec": str(exp.path)})
    wl = exp.workload
    by_rate = {m.request_rate: m for m in measured}
    missing = [r for r in wl.request_rates if r not in by_rate]
    if missing:
        raise FitError("measured serving results are missing rates", context={"rates": missing})
    simulated = await sweep_serving(runner, exp, wl, calibration, measured_kv_tokens=measured_kv_tokens)
    pairs = tuple(ServingPair(by_rate[s.request_rate], s) for s in simulated)
    return ServingEvaluation(label, calibration.label, pairs)
