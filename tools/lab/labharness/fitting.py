"""Two-scalar calibration fit from matched static-batch data.

Method (reproduces the 2026-09-28 fit):

* Prefill is compute-bound in the roofline, so simulated prefill latency is
  inversely proportional to `compute_efficiency`. With the simulator run at
  base efficiency `e0` (the built-in default 0.35), the efficiency that makes
  shape i match exactly is `e0 * sim_prefill_i / real_prefill_i`. The fit is
  the median of that over shapes.
* Decode is HBM-bandwidth-bound, so simulated decode step latency is inversely
  proportional to `decode_memory_bandwidth_scale`. Likewise the fit is
  `s0 * median(sim_step_i / real_step_i)` with `s0` = 1.0 by default.

The median (not a mean or least squares) keeps one noisy shape from dragging
the fit. Because a roofline phase is a max/sum of compute and memory terms,
the proportionality is exact only where one term dominates; the validation
step re-simulates with the fitted scalars rather than assuming it.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from labharness.errors import FitError
from labharness.results import StaticMeasurement
from labharness.simulate import StaticSimRow
from labharness.spec import Shape

# Must match SimulationCalibration::default() in src/calibration.rs.
DEFAULT_COMPUTE_EFFICIENCY: Final = 0.35
DEFAULT_DECODE_MEMORY_BANDWIDTH_SCALE: Final = 1.0


@dataclass(frozen=True, slots=True)
class MatchedShape:
    measured: StaticMeasurement
    simulated: StaticSimRow

    @property
    def shape(self) -> Shape:
        return self.measured.shape

    @property
    def prefill_ratio(self) -> float:
        return self.simulated.prefill_ms / self.measured.prefill_ms

    @property
    def decode_step_ratio(self) -> float:
        return self.simulated.decode_ms_per_step / self.measured.decode_ms_per_step


@dataclass(frozen=True, slots=True)
class ScalarFit:
    compute_efficiency: float
    decode_memory_bandwidth_scale: float
    base_compute_efficiency: float
    base_decode_memory_bandwidth_scale: float
    shapes: tuple[Shape, ...]
    prefill_ratios: tuple[float, ...]
    decode_step_ratios: tuple[float, ...]


def match(measured: Sequence[StaticMeasurement], simulated: Sequence[StaticSimRow]) -> list[MatchedShape]:
    """Pair measurements and simulation rows by shape; both sides must agree exactly."""
    by_shape = {row.shape: row for row in simulated}
    matched: list[MatchedShape] = []
    for m in measured:
        row = by_shape.get(m.shape)
        if row is None:
            raise FitError(f"no simulated row for shape {m.shape.label}", context={"shape": m.shape.label})
        if row.decode_tokens != m.decode_tokens:
            raise FitError(
                "decode token count differs between measurement and simulation",
                context={"shape": m.shape.label, "measured": m.decode_tokens, "simulated": row.decode_tokens},
            )
        if not row.feasible:
            raise FitError("simulated shape is infeasible", context={"shape": m.shape.label})
        matched.append(MatchedShape(measured=m, simulated=row))
    return sorted(matched, key=lambda item: item.shape)


def fit_scalars(
    matched: Sequence[MatchedShape],
    *,
    base_compute_efficiency: float = DEFAULT_COMPUTE_EFFICIENCY,
    base_decode_memory_bandwidth_scale: float = DEFAULT_DECODE_MEMORY_BANDWIDTH_SCALE,
) -> ScalarFit:
    if not matched:
        raise FitError("need at least one matched shape to fit")
    for item in matched:
        if item.measured.prefill_ms <= 0 or item.measured.decode_ms_per_step <= 0:
            raise FitError("measured latencies must be positive", context={"shape": item.shape.label})
    prefill = tuple(item.prefill_ratio for item in matched)
    decode = tuple(item.decode_step_ratio for item in matched)
    return ScalarFit(
        compute_efficiency=base_compute_efficiency * statistics.median(prefill),
        decode_memory_bandwidth_scale=base_decode_memory_bandwidth_scale * statistics.median(decode),
        base_compute_efficiency=base_compute_efficiency,
        base_decode_memory_bandwidth_scale=base_decode_memory_bandwidth_scale,
        shapes=tuple(item.shape for item in matched),
        prefill_ratios=prefill,
        decode_step_ratios=decode,
    )


def leave_one_out_folds(matched: Sequence[MatchedShape]) -> list[tuple[Shape, ScalarFit]]:
    """One fit per held-out shape, trained on every other shape."""
    if len(matched) < 2:
        raise FitError("leave-one-shape-out needs at least two shapes")
    return [
        (held.shape, fit_scalars([item for item in matched if item.shape != held.shape])) for held in matched
    ]


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def pct_error(predicted: float, measured: float) -> float:
    """Signed relative error in percent: positive means the simulator is slower."""
    if measured <= 0:
        raise FitError("measured value must be positive", context={"measured": measured})
    return 100.0 * (predicted - measured) / measured


@dataclass(frozen=True, slots=True)
class ShapeErrors:
    shape: Shape
    prefill_pct: float
    decode_step_pct: float
    end_to_end_pct: float


def shape_errors(measured: StaticMeasurement, simulated: StaticSimRow) -> ShapeErrors:
    return ShapeErrors(
        shape=measured.shape,
        prefill_pct=pct_error(simulated.prefill_ms, measured.prefill_ms),
        decode_step_pct=pct_error(simulated.decode_ms_per_step, measured.decode_ms_per_step),
        end_to_end_pct=pct_error(simulated.end_to_end_ms, measured.end_to_end_ms),
    )


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    prefill_mean_abs_pct: float
    decode_step_mean_abs_pct: float
    end_to_end_mean_abs_pct: float
    end_to_end_max_abs_pct: float
    count: int


def summarize(errors: Sequence[ShapeErrors]) -> ErrorSummary:
    if not errors:
        raise FitError("no errors to summarize")
    return ErrorSummary(
        prefill_mean_abs_pct=statistics.fmean(abs(e.prefill_pct) for e in errors),
        decode_step_mean_abs_pct=statistics.fmean(abs(e.decode_step_pct) for e in errors),
        end_to_end_mean_abs_pct=statistics.fmean(abs(e.end_to_end_pct) for e in errors),
        end_to_end_max_abs_pct=max(abs(e.end_to_end_pct) for e in errors),
        count=len(errors),
    )
