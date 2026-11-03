"""Fit a separate KV-read bandwidth scale from memory-bound decode steps.

The simulator prices a memory-bound static decode step as

    step = fixed + weight_ms_at_full / s_w + kv_ms_at_full / s_kv

where `s_w` is `decode_memory_bandwidth_scale` (streaming weight reads),
`s_kv` is `kv_read_bandwidth_scale` (paged KV-cache reads), and the two
`*_at_full` terms are the read times at 100% of peak HBM bandwidth (see
docs/features/compute_roofline.md). `fixed` collects whatever does not scale
with either bandwidth (scheduler overhead spread over the decode, zero for a
pure roofline).

`decode_terms` recovers the three coefficients per shape from three simulator
runs that differ only in one scale each: the base `(s_w, s_kv0)`, a KV probe
`(s_w, s_kv1)`, and a weight probe `(s_w1, s_kv0)`. Probes lower a scale, which
makes the step only more memory-bound, so the decomposition is exact whenever
the base is memory-bound; orchestration re-simulates the fitted scales to
confirm it.

`fit_kv_read` then solves for `1 / s_kv` (mode `after`: `s_w` kept) or for
`1 / s_w` and `1 / s_kv` together (mode `joint`) by least squares on the
relative step error, which is linear in the inverse scales. Least squares
rather than the median-ratio fit of fitting.py, because a slope (KV growth)
is what is being fitted, not a uniform ratio.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from labharness.errors import FitError
from labharness.results import StaticMeasurement
from labharness.simulate import StaticSimRow
from labharness.spec import Shape

_RELATIVE_TOLERANCE: Final = 1e-9
_TREND_TOKENS: Final = 10_000.0


class KvFitMode(StrEnum):
    AFTER = "after"  # keep decode_memory_bandwidth_scale, fit kv_read_bandwidth_scale
    JOINT = "joint"  # fit both scales together


def mean_kv_context_tokens(shape: Shape, decode_tokens: int) -> float:
    """KV tokens one decode step reads on average: batch x (prompt + (decode + 1) / 2).

    This is the static decode roofline's mean context (`average_decode_context_tokens`).
    """
    return shape.batch * (shape.prompt + (decode_tokens + 1) / 2)


@dataclass(frozen=True, slots=True)
class ProbeScales:
    """Scales of the base run and of the two probes that isolate each term."""

    weight_scale: float
    kv_scale: float
    weight_probe: float
    kv_probe: float

    def __post_init__(self) -> None:
        values = (self.weight_scale, self.kv_scale, self.weight_probe, self.kv_probe)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise FitError("probe scales must be finite and positive", context={"scales": values})
        if math.isclose(self.weight_probe, self.weight_scale) or math.isclose(self.kv_probe, self.kv_scale):
            raise FitError("each probe scale must differ from its base scale", context={"scales": values})

    @classmethod
    def halving(cls, weight_scale: float, kv_scale: float) -> ProbeScales:
        """Probes at half each base scale (more memory-bound than the base)."""
        return cls(weight_scale, kv_scale, weight_scale / 2, kv_scale / 2)


@dataclass(frozen=True, slots=True)
class DecodeTerms:
    dataset: str
    shape: Shape
    kv_tokens: float
    measured_ms: float
    fixed_ms: float
    weight_ms_at_full: float
    kv_ms_at_full: float

    def predict(self, weight_scale: float, kv_scale: float) -> float:
        return self.fixed_ms + self.weight_ms_at_full / weight_scale + self.kv_ms_at_full / kv_scale


def decode_terms(
    dataset: str,
    measured: StaticMeasurement,
    *,
    base: StaticSimRow,
    kv_probe: StaticSimRow,
    weight_probe: StaticSimRow,
    probes: ProbeScales,
) -> DecodeTerms:
    for row in (base, kv_probe, weight_probe):
        if row.shape != measured.shape or row.decode_tokens != measured.decode_tokens:
            raise FitError(
                "simulated rows do not match the measured shape", context={"shape": measured.shape.label}
            )
    if measured.decode_ms_per_step <= 0:
        raise FitError("measured decode step must be positive", context={"shape": measured.shape.label})
    base_ms = base.decode_ms_per_step
    kv_ms = (kv_probe.decode_ms_per_step - base_ms) / (1 / probes.kv_probe - 1 / probes.kv_scale)
    weight_ms = (weight_probe.decode_ms_per_step - base_ms) / (
        1 / probes.weight_probe - 1 / probes.weight_scale
    )
    if kv_ms <= base_ms * _RELATIVE_TOLERANCE:
        raise FitError(
            "decode step has no KV-read term (simulator binary without kv_read_bandwidth_scale?)",
            context={"shape": measured.shape.label},
        )
    if weight_ms <= base_ms * _RELATIVE_TOLERANCE:
        raise FitError("decode step is not weight-read bound", context={"shape": measured.shape.label})
    return DecodeTerms(
        dataset=dataset,
        shape=measured.shape,
        kv_tokens=mean_kv_context_tokens(measured.shape, measured.decode_tokens),
        measured_ms=measured.decode_ms_per_step,
        fixed_ms=base_ms - weight_ms / probes.weight_scale - kv_ms / probes.kv_scale,
        weight_ms_at_full=weight_ms,
        kv_ms_at_full=kv_ms,
    )


@dataclass(frozen=True, slots=True)
class KvReadFit:
    mode: KvFitMode
    decode_memory_bandwidth_scale: float
    kv_read_bandwidth_scale: float
    base_decode_memory_bandwidth_scale: float
    datasets: tuple[str, ...]
    sample_count: int

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "decode_memory_bandwidth_scale": round(self.decode_memory_bandwidth_scale, 6),
            "kv_read_bandwidth_scale": round(self.kv_read_bandwidth_scale, 6),
            "base_decode_memory_bandwidth_scale": self.base_decode_memory_bandwidth_scale,
            "datasets": list(self.datasets),
            "sample_count": self.sample_count,
        }


def fit_kv_read(samples: Sequence[DecodeTerms], mode: KvFitMode, *, weight_scale: float) -> KvReadFit:
    """Least squares on relative step error, linear in the inverse scales."""
    if not samples:
        raise FitError("need at least one decode sample to fit")
    rows = [(s.weight_ms_at_full / s.measured_ms, s.kv_ms_at_full / s.measured_ms, s) for s in samples]
    match mode:
        case KvFitMode.AFTER:
            inv_weight = 1 / weight_scale
            numerator = sum(
                k * (1 - (s.fixed_ms + s.weight_ms_at_full * inv_weight) / s.measured_ms) for _, k, s in rows
            )
            denominator = sum(k * k for _, k, _ in rows)
            inv_kv = numerator / denominator
        case KvFitMode.JOINT:
            targets = [1 - s.fixed_ms / s.measured_ms for _, _, s in rows]
            sww = sum(w * w for w, _, _ in rows)
            skk = sum(k * k for _, k, _ in rows)
            swk = sum(w * k for w, k, _ in rows)
            swt = sum(w * t for (w, _, _), t in zip(rows, targets, strict=True))
            skt = sum(k * t for (_, k, _), t in zip(rows, targets, strict=True))
            determinant = sww * skk - swk * swk
            if determinant <= sww * skk * 1e-9:
                raise FitError(
                    "weight and KV scales are not separately identifiable: "
                    "the samples need different KV sizes",
                    context={"samples": len(rows)},
                )
            inv_weight = (swt * skk - skt * swk) / determinant
            inv_kv = (skt * sww - swt * swk) / determinant
    if not (math.isfinite(inv_kv) and inv_kv > 0 and math.isfinite(inv_weight) and inv_weight > 0):
        raise FitError(
            "fitted bandwidth scales must be positive",
            context={"inverse_kv_scale": inv_kv, "inverse_weight_scale": inv_weight},
        )
    return KvReadFit(
        mode=mode,
        decode_memory_bandwidth_scale=weight_scale if mode is KvFitMode.AFTER else 1 / inv_weight,
        kv_read_bandwidth_scale=1 / inv_kv,
        base_decode_memory_bandwidth_scale=weight_scale,
        datasets=tuple(sorted({s.dataset for s in samples})),
        sample_count=len(samples),
    )


def holdout_folds(
    samples: Sequence[DecodeTerms], mode: KvFitMode, *, weight_scale: float
) -> list[tuple[str, KvReadFit]]:
    """One fit per held-out dataset, trained on every other dataset."""
    datasets = sorted({s.dataset for s in samples})
    if len(datasets) < 2:
        raise FitError("dataset holdout needs at least two datasets", context={"datasets": datasets})
    return [
        (held, fit_kv_read([s for s in samples if s.dataset != held], mode, weight_scale=weight_scale))
        for held in datasets
    ]


@dataclass(frozen=True, slots=True)
class ErrorTrend:
    """Signed decode-step error versus KV tokens per step."""

    slope_pct_per_10k_tokens: float
    mean_abs_pct: float
    count: int

    def to_json(self) -> dict[str, object]:
        return {
            "slope_pct_per_10k_kv_tokens": round(self.slope_pct_per_10k_tokens, 4),
            "mean_abs_pct": round(self.mean_abs_pct, 4),
            "count": self.count,
        }


def error_trend(points: Sequence[tuple[float, float]]) -> ErrorTrend:
    """OLS slope of error (%) on KV tokens, per 10k tokens; 0 with fewer than two KV sizes."""
    if not points:
        raise FitError("no errors to summarize")
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    mean_x = statistics.fmean(xs)
    spread = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0
    if spread > 0:
        mean_y = statistics.fmean(ys)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread * _TREND_TOKENS
    return ErrorTrend(
        slope_pct_per_10k_tokens=slope,
        mean_abs_pct=statistics.fmean(abs(y) for y in ys),
        count=len(points),
    )
