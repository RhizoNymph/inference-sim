"""Fit compute efficiency versus tokens per forward pass from static prefills.

The simulator prices a prefill pass as `max(compute(e), weight_read)` where
`compute(e) = compute(1) / e` and `e` is the compute efficiency (see
docs/features/compute_roofline.md). For each measured prefill the fit needs
`compute(1)` and the weight-read floor, and gets both from two simulator runs
with scalar calibrations:

* at a tiny base efficiency `e0` compute dominates every shape, so
  `compute(1) = e0 * sim(e0)`;
* at `e = 1` the simulator returns `max(compute(1), floor)`, so a result above
  `compute(1)` is the floor, and otherwise the shape is compute-bound at any
  efficiency.

A sample whose measured latency is below `(1 + margin) * floor` is
weight-read bound: its efficiency is not identifiable (only a lower bound), so
it is reported as skipped and the curve clamps there. Every other sample gives
`efficiency = compute(1) / measured`; samples with the same token count
(batch x prompt) are combined by their median.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby
from typing import Final

from labharness.errors import FitError
from labharness.results import StaticMeasurement
from labharness.simulate import StaticSimRow
from labharness.spec import Shape
from labharness.toml_emit import fmt_float

# Base efficiency for the compute-dominated run: low enough that even a
# one-token pass of a small model is compute-bound in the simulator.
CURVE_BASE_EFFICIENCY: Final = 0.005
# A sample within this fraction above the weight-read floor counts as
# memory-bound. Covers the static benchmark's fixed per-pass overhead (one
# sampled token, generate() bookkeeping: about 2 ms over a 19 ms floor on the
# 3090 lab).
DEFAULT_MEMORY_BOUND_MARGIN: Final = 0.25
_RELATIVE_TOLERANCE: Final = 1e-6


@dataclass(frozen=True, slots=True)
class CurveSample:
    shape: Shape
    tokens: int  # batch x prompt: tokens in the prefill forward pass
    measured_ms: float
    compute_ms_at_full_efficiency: float
    memory_floor_ms: float | None  # None: compute-bound even at efficiency 1


class SkipReason(StrEnum):
    MEMORY_BOUND = "memory_bound"


@dataclass(frozen=True, slots=True)
class SkippedSample:
    shape: Shape
    tokens: int
    reason: SkipReason


@dataclass(frozen=True, slots=True)
class CurvePoint:
    tokens: int
    efficiency: float
    shapes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EfficiencyCurveFit:
    points: tuple[CurvePoint, ...]
    skipped: tuple[SkippedSample, ...]
    memory_bound_margin: float

    def toml_value(self) -> str:
        """The `compute_efficiency_curve` TOML array: `[[tokens, efficiency], ...]`."""
        return "[" + ", ".join(f"[{p.tokens}, {fmt_float(p.efficiency)}]" for p in self.points) + "]"

    def to_json(self) -> dict[str, object]:
        return {
            "memory_bound_margin": self.memory_bound_margin,
            "points": [
                {"tokens": p.tokens, "efficiency": round(p.efficiency, 6), "shapes": list(p.shapes)}
                for p in self.points
            ],
            "skipped": [
                {"shape": s.shape.label, "tokens": s.tokens, "reason": s.reason.value} for s in self.skipped
            ],
        }


def curve_sample(
    measured: StaticMeasurement,
    at_base: StaticSimRow,
    at_full: StaticSimRow,
    *,
    base_efficiency: float = CURVE_BASE_EFFICIENCY,
) -> CurveSample:
    """Build one sample from the simulator's prefill at `base_efficiency` and at 1."""
    if at_base.shape != measured.shape or at_full.shape != measured.shape:
        raise FitError(
            "simulated rows do not match the measured shape", context={"shape": measured.shape.label}
        )
    if at_base.prefill_ms <= at_full.prefill_ms * (1.0 + _RELATIVE_TOLERANCE):
        raise FitError(
            "prefill is still memory-bound at the base efficiency; lower it",
            context={"shape": measured.shape.label, "base_efficiency": base_efficiency},
        )
    compute_full = base_efficiency * at_base.prefill_ms
    floor = at_full.prefill_ms if at_full.prefill_ms > compute_full * (1.0 + _RELATIVE_TOLERANCE) else None
    return CurveSample(
        shape=measured.shape,
        tokens=measured.shape.batch * measured.shape.prompt,
        measured_ms=measured.prefill_ms,
        compute_ms_at_full_efficiency=compute_full,
        memory_floor_ms=floor,
    )


def fit_efficiency_curve(
    samples: Sequence[CurveSample], *, memory_bound_margin: float = DEFAULT_MEMORY_BOUND_MARGIN
) -> EfficiencyCurveFit:
    if memory_bound_margin < 0:
        raise FitError("memory_bound_margin must be non-negative", context={"margin": memory_bound_margin})
    kept: list[tuple[CurveSample, float]] = []
    skipped: list[SkippedSample] = []
    for sample in samples:
        if sample.measured_ms <= 0 or sample.compute_ms_at_full_efficiency <= 0:
            raise FitError("latencies must be positive", context={"shape": sample.shape.label})
        floor = sample.memory_floor_ms
        if floor is not None and sample.measured_ms < (1.0 + memory_bound_margin) * floor:
            skipped.append(SkippedSample(sample.shape, sample.tokens, SkipReason.MEMORY_BOUND))
            continue
        efficiency = sample.compute_ms_at_full_efficiency / sample.measured_ms
        if efficiency > 1.0:
            raise FitError(
                "fitted efficiency exceeds 1: measured faster than the peak FLOPs allow",
                context={"shape": sample.shape.label, "efficiency": round(efficiency, 4)},
            )
        kept.append((sample, efficiency))
    kept.sort(key=lambda item: (item[0].tokens, item[0].shape))
    points = tuple(
        CurvePoint(
            tokens=tokens,
            efficiency=statistics.median(e for _, e in group_items),
            shapes=tuple(s.shape.label for s, _ in group_items),
        )
        for tokens, group in groupby(kept, key=lambda item: item[0].tokens)
        for group_items in [list(group)]
    )
    if len(points) < 2:
        raise FitError(
            "need compute-bound samples at at least 2 token counts",
            context={"compute_bound": len(points), "skipped": len(skipped)},
        )
    return EfficiencyCurveFit(
        points=points,
        skipped=tuple(sorted(skipped, key=lambda s: (s.tokens, s.shape))),
        memory_bound_margin=memory_bound_margin,
    )
