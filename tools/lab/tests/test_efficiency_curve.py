from __future__ import annotations

import math

import pytest

from labharness.efficiency_curve import (
    CurveSample,
    SkipReason,
    curve_sample,
    fit_efficiency_curve,
)
from labharness.errors import FitError
from labharness.results import StaticMeasurement
from labharness.simulate import StaticSimRow
from labharness.spec import Shape


def measured(shape: Shape, prefill: float) -> StaticMeasurement:
    return StaticMeasurement(
        shape=shape, decode_tokens=128, tp=1, pp=1, prefill_ms=prefill, decode_ms=2500.0,
        decode_ms_per_step=19.5, end_to_end_ms=2600.0, prefill_spread_ms=None, e2e_spread_ms=None,
    )  # fmt: skip


def sim(shape: Shape, prefill: float) -> StaticSimRow:
    return StaticSimRow(
        shape=shape, decode_tokens=128, prefill_ms=prefill, decode_ms=2500.0,
        end_to_end_ms=2600.0, feasible=True,
    )  # fmt: skip


def sample(tokens: int, measured_ms: float, compute_ms: float, floor_ms: float | None) -> CurveSample:
    return CurveSample(
        shape=Shape(1, tokens),
        tokens=tokens,
        measured_ms=measured_ms,
        compute_ms_at_full_efficiency=compute_ms,
        memory_floor_ms=floor_ms,
    )


def test_curve_sample_recovers_full_efficiency_compute_and_memory_floor() -> None:
    shape = Shape(1, 16)
    # At base efficiency 0.005 compute dominates: 0.005 * 400 = 2 ms at e = 1.
    # At e = 1 the simulator reports the 19 ms weight-read floor instead.
    got = curve_sample(measured(shape, 21.7), sim(shape, 400.0), sim(shape, 19.0), base_efficiency=0.005)
    assert got.tokens == 16
    assert math.isclose(got.compute_ms_at_full_efficiency, 2.0)
    assert got.memory_floor_ms == 19.0


def test_curve_sample_has_no_floor_when_compute_bound_at_full_efficiency() -> None:
    shape = Shape(2, 512)
    got = curve_sample(measured(shape, 200.0), sim(shape, 35_000.0), sim(shape, 175.0), base_efficiency=0.005)
    assert got.tokens == 1024
    assert math.isclose(got.compute_ms_at_full_efficiency, 175.0)
    assert got.memory_floor_ms is None


def test_curve_sample_rejects_a_base_efficiency_that_is_still_memory_bound() -> None:
    shape = Shape(1, 1)
    with pytest.raises(FitError, match="memory-bound"):
        curve_sample(measured(shape, 20.0), sim(shape, 19.0), sim(shape, 19.0), base_efficiency=0.005)


def test_fit_divides_full_efficiency_compute_by_measured() -> None:
    fit = fit_efficiency_curve(
        [sample(512, 100.0, 78.0, None), sample(1024, 200.0, 178.0, None), sample(128, 31.5, 22.0, 19.4)]
    )
    assert [(p.tokens, round(p.efficiency, 6)) for p in fit.points] == [
        (128, round(22.0 / 31.5, 6)),
        (512, 0.78),
        (1024, 0.89),
    ]
    assert fit.skipped == ()


def test_fit_skips_memory_bound_samples() -> None:
    fit = fit_efficiency_curve(
        [
            sample(16, 21.7, 2.8, 19.4),  # 21.7 < 1.25 * 19.4: weight-read bound
            sample(64, 21.8, 11.1, 19.4),
            sample(128, 31.5, 22.2, 19.4),  # 31.5 > 24.25: compute-bound
            sample(512, 100.0, 78.0, None),
        ],
        memory_bound_margin=0.25,
    )
    assert [p.tokens for p in fit.points] == [128, 512]
    assert [(s.tokens, s.reason) for s in fit.skipped] == [
        (16, SkipReason.MEMORY_BOUND),
        (64, SkipReason.MEMORY_BOUND),
    ]


def test_fit_takes_the_median_of_samples_with_equal_tokens() -> None:
    fit = fit_efficiency_curve(
        [
            sample(512, 100.0, 70.0, None),
            CurveSample(Shape(2, 256), 512, 100.0, 80.0, None),
            CurveSample(Shape(4, 128), 512, 100.0, 90.0, None),
            sample(1024, 100.0, 89.0, None),
        ]
    )
    assert fit.points[0].tokens == 512
    assert math.isclose(fit.points[0].efficiency, 0.8)
    assert fit.points[0].shapes == ("1x512", "2x256", "4x128")


def test_fit_needs_two_compute_bound_token_counts() -> None:
    with pytest.raises(FitError, match="at least 2"):
        fit_efficiency_curve([sample(512, 100.0, 78.0, None), sample(16, 21.7, 2.8, 19.4)])


def test_fit_rejects_efficiency_above_one() -> None:
    with pytest.raises(FitError, match="exceeds 1"):
        fit_efficiency_curve([sample(512, 70.0, 78.0, None), sample(1024, 200.0, 178.0, None)])


def test_fit_rejects_bad_margin_and_measurements() -> None:
    with pytest.raises(FitError):
        fit_efficiency_curve([sample(512, 100.0, 78.0, None)] * 2, memory_bound_margin=-0.1)
    with pytest.raises(FitError):
        fit_efficiency_curve([sample(512, 0.0, 78.0, None), sample(1024, 200.0, 178.0, None)])


def test_toml_value_is_a_list_of_token_efficiency_pairs() -> None:
    fit = fit_efficiency_curve([sample(512, 100.0, 78.0, None), sample(1024, 200.0, 178.12345, None)])
    assert fit.toml_value() == "[[512, 0.78], [1024, 0.890617]]"
