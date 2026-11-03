"""Orchestration of the KV-read bandwidth fit: simulate, decompose, fit, hold out.

Datasets are static-batch decode runs (for example the decode batch sweep and
the long-context run). For every dataset the simulator is run at the base
scales and at two probes (kv_read_fit.ProbeScales) through the base profile
with inline overrides; the decode-step terms are fitted over all datasets and
once per held-out dataset; every reported error comes from re-simulating
with the fitted scales, never from the linear model alone.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from labharness.errors import FitError, SimulatorError
from labharness.evaluate import StaticEvaluation, check_measured_matches_spec, evaluate_static
from labharness.kv_read_fit import (
    DecodeTerms,
    ErrorTrend,
    KvFitMode,
    KvReadFit,
    ProbeScales,
    decode_terms,
    error_trend,
    fit_kv_read,
    holdout_folds,
    mean_kv_context_tokens,
)
from labharness.kv_read_profile import HoldoutSummary
from labharness.logging_setup import get_logger
from labharness.results import StaticMeasurement
from labharness.simulate import ProfileCalibration, ProfileOverrideCalibration, SimRunner, sweep_static
from labharness.spec import Experiment, Shape, StaticBatchWorkload

# Re-simulated fitted steps must match the linear decomposition this closely,
# or some step is not memory-bound and the decomposition does not hold.
MAX_LINEARITY_DRIFT_PCT: Final = 0.1
# Decode batches above this are dominated by KV-exhaustion preemption on the
# 3090 lab (validation ledger entry 16), which the KV-read term cannot explain.
DEFAULT_MAX_BATCH: Final = 32


@dataclass(frozen=True, slots=True)
class KvDataset:
    name: str
    exp: Experiment
    measured: tuple[StaticMeasurement, ...]
    excluded: tuple[Shape, ...]
    runner: SimRunner
    source: str


def select_measurements(
    measured: Sequence[StaticMeasurement], *, max_batch: int
) -> tuple[tuple[StaticMeasurement, ...], tuple[Shape, ...]]:
    """Keep shapes with batch <= max_batch; return (kept, excluded shapes)."""
    if max_batch < 1:
        raise FitError("max batch must be at least 1", context={"max_batch": max_batch})
    kept = tuple(m for m in measured if m.shape.batch <= max_batch)
    excluded = tuple(m.shape for m in measured if m.shape.batch > max_batch)
    if not kept:
        raise FitError("no measured shapes at or below the max batch", context={"max_batch": max_batch})
    return kept, excluded


def _overrides(profile: Path, weight_scale: float, kv_scale: float) -> ProfileOverrideCalibration:
    return ProfileOverrideCalibration(
        profile, (("decode_memory_bandwidth_scale", weight_scale), ("kv_read_bandwidth_scale", kv_scale))
    )


def _decode_tokens(exp: Experiment) -> int:
    if not isinstance(exp.workload, StaticBatchWorkload):
        raise FitError("KV-read fit needs static-batch specs", context={"spec": str(exp.path)})
    return exp.workload.decode_tokens


async def dataset_terms(dataset: KvDataset, profile: Path, probes: ProbeScales) -> list[DecodeTerms]:
    check_measured_matches_spec(dataset.exp, dataset.measured)
    exp = dataset.exp
    shapes = [m.shape for m in dataset.measured]
    decode = _decode_tokens(exp)
    base, kv_probe, weight_probe = await asyncio.gather(
        *(
            sweep_static(
                dataset.runner, exp.model, exp.parallelism, shapes, decode, _overrides(profile, w, k)
            )
            for w, k in (
                (probes.weight_scale, probes.kv_scale),
                (probes.weight_scale, probes.kv_probe),
                (probes.weight_probe, probes.kv_scale),
            )
        )
    )
    by_shape = [{row.shape: row for row in rows} for rows in (base, kv_probe, weight_probe)]
    return [
        decode_terms(
            dataset.name,
            m,
            base=by_shape[0][m.shape],
            kv_probe=by_shape[1][m.shape],
            weight_probe=by_shape[2][m.shape],
            probes=probes,
        )
        for m in dataset.measured
    ]


@dataclass(frozen=True, slots=True)
class DatasetResult:
    name: str
    source: str
    excluded: tuple[Shape, ...]
    before: StaticEvaluation
    fitted: StaticEvaluation
    held_out: StaticEvaluation
    held_out_fit: KvReadFit

    def trend(self, evaluation: StaticEvaluation) -> ErrorTrend:
        return error_trend(
            [
                (mean_kv_context_tokens(m.shape, m.decode_tokens), e.decode_step_pct)
                for (m, _), e in zip(evaluation.pairs, evaluation.errors, strict=True)
            ]
        )


@dataclass(frozen=True, slots=True)
class KvFitOutcome:
    fit: KvReadFit
    probes: ProbeScales
    datasets: tuple[DatasetResult, ...]
    max_linearity_drift_pct: float

    def holdout_summaries(self) -> tuple[HoldoutSummary, ...]:
        return tuple(
            HoldoutSummary(
                held_out=d.name,
                kv_read_bandwidth_scale=d.held_out_fit.kv_read_bandwidth_scale,
                decode_step_mean_abs_pct=d.held_out.summary.decode_step_mean_abs_pct,
            )
            for d in self.datasets
        )


async def fit_kv_read_datasets(
    datasets: Sequence[KvDataset], base_profile: Path, base_weight_scale: float, mode: KvFitMode
) -> KvFitOutcome:
    names = [d.name for d in datasets]
    if len(set(names)) != len(names):
        raise FitError("dataset names must be unique", context={"datasets": names})
    log = get_logger()
    probes = ProbeScales.halving(base_weight_scale, base_weight_scale)
    per_dataset = await asyncio.gather(*(dataset_terms(d, base_profile, probes) for d in datasets))
    samples = [term for terms in per_dataset for term in terms]
    fit = fit_kv_read(samples, mode, weight_scale=base_weight_scale)
    folds = dict(holdout_folds(samples, mode, weight_scale=base_weight_scale))
    log.info(
        "fitted kv read scale",
        extra={
            "mode": mode.value,
            "kv_read_bandwidth_scale": round(fit.kv_read_bandwidth_scale, 5),
            "decode_memory_bandwidth_scale": round(fit.decode_memory_bandwidth_scale, 5),
            **{f"held_out_{held}": round(f.kv_read_bandwidth_scale, 5) for held, f in folds.items()},
        },
    )

    def scaled(f: KvReadFit) -> ProfileOverrideCalibration:
        return _overrides(base_profile, f.decode_memory_bandwidth_scale, f.kv_read_bandwidth_scale)

    async def evaluate(d: KvDataset) -> DatasetResult:
        before, fitted, held_out = await asyncio.gather(
            evaluate_static(
                d.runner, d.exp, d.measured, ProfileCalibration(base_profile), f"{d.name} before"
            ),
            evaluate_static(d.runner, d.exp, d.measured, scaled(fit), f"{d.name} fitted (all datasets)"),
            evaluate_static(d.runner, d.exp, d.measured, scaled(folds[d.name]), f"{d.name} held out"),
        )
        return DatasetResult(d.name, d.source, d.excluded, before, fitted, held_out, folds[d.name])

    results = await asyncio.gather(*(evaluate(d) for d in datasets))
    terms_by_key = {(t.dataset, t.shape): t for t in samples}
    drift = 0.0
    for result in results:
        for _, row in result.fitted.pairs:
            predicted = terms_by_key[(result.name, row.shape)].predict(
                fit.decode_memory_bandwidth_scale, fit.kv_read_bandwidth_scale
            )
            drift = max(drift, abs(row.decode_ms_per_step - predicted) / predicted * 100.0)
    if drift > MAX_LINEARITY_DRIFT_PCT:
        raise SimulatorError(
            f"re-simulated decode steps drift {drift:.3f}% from the linear decomposition; "
            "some step is not memory-bound",
            workload=str(base_profile),
        )
    return KvFitOutcome(fit=fit, probes=probes, datasets=tuple(results), max_linearity_drift_pct=drift)


def outcome_json(outcome: KvFitOutcome) -> dict[str, object]:
    def evaluation(d: DatasetResult, e: StaticEvaluation) -> dict[str, object]:
        return {
            "calibration": e.calibration,
            "prefill_mean_abs_pct": round(e.summary.prefill_mean_abs_pct, 4),
            "decode_step_mean_abs_pct": round(e.summary.decode_step_mean_abs_pct, 4),
            "end_to_end_mean_abs_pct": round(e.summary.end_to_end_mean_abs_pct, 4),
            "decode_step_trend": d.trend(e).to_json(),
            "shapes": [
                {
                    "shape": m.shape.label,
                    "kv_tokens_per_step": round(mean_kv_context_tokens(m.shape, m.decode_tokens), 1),
                    "measured_step_ms": m.decode_ms_per_step,
                    "simulated_step_ms": round(s.decode_ms_per_step, 5),
                    "decode_step_pct": round(err.decode_step_pct, 4),
                    "end_to_end_pct": round(err.end_to_end_pct, 4),
                }
                for (m, s), err in zip(e.pairs, e.errors, strict=True)
            ],
        }

    return {
        "fit": outcome.fit.to_json(),
        "probes": {
            "weight_scale": outcome.probes.weight_scale,
            "kv_scale": outcome.probes.kv_scale,
            "weight_probe": outcome.probes.weight_probe,
            "kv_probe": outcome.probes.kv_probe,
        },
        "max_linearity_drift_pct": round(outcome.max_linearity_drift_pct, 6),
        "datasets": [
            {
                "name": d.name,
                "measured": d.source,
                "excluded_shapes": [s.label for s in d.excluded],
                "held_out_fit": d.held_out_fit.to_json(),
                "before": evaluation(d, d.before),
                "fitted": evaluation(d, d.fitted),
                "held_out": evaluation(d, d.held_out),
            }
            for d in outcome.datasets
        ],
    }


def outcome_markdown(outcome: KvFitOutcome, tag: str, profile: str, base_profile: str) -> str:
    fit = outcome.fit
    lines = [
        f"## KV-read bandwidth scale: {tag}",
        "",
        f"Mode `{fit.mode.value}`; base profile `{base_profile}`; profile `{profile}` "
        "(loaded and checked by the simulator).",
        "",
        f"Fitted on all datasets ({fit.sample_count} decode steps): `kv_read_bandwidth_scale = "
        f"{fit.kv_read_bandwidth_scale:.4f}`, `decode_memory_bandwidth_scale = "
        f"{fit.decode_memory_bandwidth_scale:.4f}` (base {fit.base_decode_memory_bandwidth_scale:.4f}). "
        f"Re-simulated steps match the linear decomposition within {outcome.max_linearity_drift_pct:.4f}%.",
        "",
        "| dataset | held-out fit kv scale | decode step before | fitted (all) | held out | "
        "trend before | trend fitted | trend held out | e2e before | e2e held out |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in outcome.datasets:
        cells = [
            d.name,
            f"{d.held_out_fit.kv_read_bandwidth_scale:.4f}",
            f"{d.before.summary.decode_step_mean_abs_pct:.1f}%",
            f"{d.fitted.summary.decode_step_mean_abs_pct:.1f}%",
            f"{d.held_out.summary.decode_step_mean_abs_pct:.1f}%",
            f"{d.trend(d.before).slope_pct_per_10k_tokens:+.2f}",
            f"{d.trend(d.fitted).slope_pct_per_10k_tokens:+.2f}",
            f"{d.trend(d.held_out).slope_pct_per_10k_tokens:+.2f}",
            f"{d.before.summary.end_to_end_mean_abs_pct:.1f}%",
            f"{d.held_out.summary.end_to_end_mean_abs_pct:.1f}%",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Trend: OLS slope of the signed decode-step error, in percentage points per 10k KV tokens per step. "
        "Held out: scales fitted on the other datasets only.",
        "",
    ]
    for d in outcome.datasets:
        excluded = ", ".join(s.label for s in d.excluded) or "none"
        lines += [
            f"### {d.name}",
            "",
            f"Measurements `{d.source}`; excluded (batch above the max): {excluded}.",
            "",
            "| shape | KV tokens/step | real ms | before ms | err | fitted ms | err | held-out ms | err |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        rows = zip(
            d.before.pairs,
            d.before.errors,
            d.fitted.pairs,
            d.fitted.errors,
            d.held_out.pairs,
            d.held_out.errors,
            strict=True,
        )
        for (m, b), be, (_, f), fe, (_, h), he in rows:
            cells = [
                m.shape.label,
                f"{mean_kv_context_tokens(m.shape, m.decode_tokens):,.0f}",
                f"{m.decode_ms_per_step:.2f}",
                f"{b.decode_ms_per_step:.2f}",
                f"{be.decode_step_pct:+.1f}%",
                f"{f.decode_ms_per_step:.2f}",
                f"{fe.decode_step_pct:+.1f}%",
                f"{h.decode_ms_per_step:.2f}",
                f"{he.decode_step_pct:+.1f}%",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
