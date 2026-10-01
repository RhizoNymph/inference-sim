from __future__ import annotations

import pytest

from labharness.errors import FitError
from labharness.fitting import (
    DEFAULT_COMPUTE_EFFICIENCY,
    MatchedShape,
    fit_scalars,
    leave_one_out_folds,
    match,
    pct_error,
    shape_errors,
    summarize,
)
from labharness.results import StaticMeasurement
from labharness.simulate import StaticSimRow
from labharness.spec import Shape

SHAPES = [Shape(1, 512), Shape(1, 2048), Shape(8, 512), Shape(8, 2048), Shape(32, 512)]


def measured(shape: Shape, prefill: float, step: float, e2e: float, decode: int = 128) -> StaticMeasurement:
    return StaticMeasurement(
        shape=shape, decode_tokens=decode, tp=1, pp=1, prefill_ms=prefill, decode_ms=step * decode,
        decode_ms_per_step=step, end_to_end_ms=e2e, prefill_spread_ms=None, e2e_spread_ms=None,
    )  # fmt: skip


def simulated(shape: Shape, prefill: float, step: float, e2e: float, decode: int = 128) -> StaticSimRow:
    return StaticSimRow(
        shape=shape, decode_tokens=decode, prefill_ms=prefill, decode_ms=step * decode,
        end_to_end_ms=e2e, feasible=True,
    )  # fmt: skip


def pair(shape: Shape, prefill_ratio: float, step_ratio: float) -> MatchedShape:
    real = measured(shape, 100.0, 20.0, 2600.0)
    return MatchedShape(real, simulated(shape, 100.0 * prefill_ratio, 20.0 * step_ratio, 2600.0))


def test_uniform_ratio_recovers_scale() -> None:
    fit = fit_scalars([pair(s, 2.4, 0.84) for s in SHAPES])
    assert fit.compute_efficiency == pytest.approx(DEFAULT_COMPUTE_EFFICIENCY * 2.4)
    assert fit.decode_memory_bandwidth_scale == pytest.approx(0.84)
    assert fit.base_compute_efficiency == 0.35 and fit.base_decode_memory_bandwidth_scale == 1.0


def test_median_ignores_one_outlier() -> None:
    ratios = [2.061, 2.431, 2.422, 2.509, 2.503]  # 2026-09-28 PP=1 prefill ratios
    fit = fit_scalars([pair(s, r, 1.0) for s, r in zip(SHAPES, ratios, strict=True)])
    assert fit.compute_efficiency == pytest.approx(0.35 * 2.431)
    wild = fit_scalars([pair(s, r, 1.0) for s, r in zip(SHAPES, [*ratios[:4], 50.0], strict=True)])
    assert wild.compute_efficiency == pytest.approx(0.35 * 2.431)


def test_custom_base_scalars() -> None:
    fit = fit_scalars(
        [pair(s, 0.5, 2.0) for s in SHAPES],
        base_compute_efficiency=0.8,
        base_decode_memory_bandwidth_scale=0.5,
    )
    assert fit.compute_efficiency == pytest.approx(0.4)
    assert fit.decode_memory_bandwidth_scale == pytest.approx(1.0)


def test_reproduces_recorded_fit_from_2026_09_28_numbers() -> None:
    """Measured medians from real_pp1.jsonl vs default-calibration simulator output."""
    real = {
        Shape(1, 512): (114.678, 19.264), Shape(1, 2048): (399.359, 19.3437),
        Shape(8, 512): (780.591, 19.7468), Shape(8, 2048): (3095.53, 20.762),
        Shape(32, 512): (3021.102, 20.7703),
    }  # fmt: skip
    sim = {
        Shape(1, 512): (236.315506, 2087.255906), Shape(1, 2048): (970.668871, 2099.301087),
        Shape(8, 512): (1890.524044, 2118.901953), Shape(8, 2048): (7765.350967, 2215.263398),
        Shape(32, 512): (7562.096176, 2227.402682),
    }  # fmt: skip
    matched = [
        MatchedShape(
            measured(s, real[s][0], real[s][1], 1.0),
            simulated(s, sim[s][0], sim[s][1] / 128, 1.0),
        )
        for s in SHAPES
    ]
    fit = fit_scalars(matched)
    assert fit.compute_efficiency == pytest.approx(0.8507, abs=5e-5)
    assert fit.decode_memory_bandwidth_scale == pytest.approx(0.8383, abs=5e-5)


def test_leave_one_out_excludes_held_out_shape() -> None:
    ratios = [1.0, 2.0, 3.0, 4.0, 5.0]
    matched = [pair(s, r, r) for s, r in zip(SHAPES, ratios, strict=True)]
    folds = dict(leave_one_out_folds(matched))
    assert set(folds) == set(SHAPES)
    # Holding out ratio 1.0 leaves [2,3,4,5] -> median 3.5.
    assert folds[Shape(1, 512)].decode_memory_bandwidth_scale == pytest.approx(3.5)
    assert Shape(1, 512) not in folds[Shape(1, 512)].shapes
    assert folds[Shape(32, 512)].decode_memory_bandwidth_scale == pytest.approx(2.5)
    with pytest.raises(FitError):
        leave_one_out_folds(matched[:1])


def test_match_pairs_by_shape_and_validates() -> None:
    reals = [measured(s, 100.0, 20.0, 2600.0) for s in SHAPES]
    sims = [simulated(s, 200.0, 17.0, 3000.0) for s in reversed(SHAPES)]
    paired = match(reals, sims)
    assert [p.shape for p in paired] == sorted(SHAPES)
    with pytest.raises(FitError, match="no simulated row"):
        match(reals, sims[:-1])
    with pytest.raises(FitError, match="decode token count"):
        match(reals[:1], [simulated(SHAPES[0], 1.0, 1.0, 1.0, decode=64)])
    infeasible = StaticSimRow(SHAPES[0], 128, 1.0, 1.0, 1.0, feasible=False)
    with pytest.raises(FitError, match="infeasible"):
        match(reals[:1], [infeasible])


def test_fit_rejects_empty_and_nonpositive() -> None:
    with pytest.raises(FitError):
        fit_scalars([])
    bad = MatchedShape(measured(SHAPES[0], 0.0, 20.0, 1.0), simulated(SHAPES[0], 1.0, 1.0, 1.0))
    with pytest.raises(FitError, match="positive"):
        fit_scalars([bad])


def test_error_metrics() -> None:
    assert pct_error(110.0, 100.0) == pytest.approx(10.0)
    assert pct_error(90.0, 100.0) == pytest.approx(-10.0)
    with pytest.raises(FitError):
        pct_error(1.0, 0.0)
    errors = [
        shape_errors(measured(SHAPES[0], 100.0, 20.0, 1000.0), simulated(SHAPES[0], 110.0, 19.0, 1020.0)),
        shape_errors(measured(SHAPES[1], 100.0, 20.0, 1000.0), simulated(SHAPES[1], 80.0, 21.0, 970.0)),
    ]
    assert errors[0].prefill_pct == pytest.approx(10.0)
    assert errors[0].decode_step_pct == pytest.approx(-5.0)
    summary = summarize(errors)
    assert summary.prefill_mean_abs_pct == pytest.approx(15.0)
    assert summary.decode_step_mean_abs_pct == pytest.approx(5.0)
    assert summary.end_to_end_mean_abs_pct == pytest.approx(2.5)
    assert summary.end_to_end_max_abs_pct == pytest.approx(3.0)
    assert summary.count == 2
