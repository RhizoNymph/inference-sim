from __future__ import annotations

import pytest

from labharness.errors import FitError
from labharness.kv_read_fit import (
    DecodeTerms,
    KvFitMode,
    ProbeScales,
    decode_terms,
    error_trend,
    fit_kv_read,
    holdout_folds,
    mean_kv_context_tokens,
)
from labharness.results import StaticMeasurement
from labharness.simulate import StaticSimRow
from labharness.spec import Shape

WEIGHT_MS = 19.41 * 0.8383  # weight read at full peak bandwidth, ms
KV_MS_PER_TOKEN = 57344 / 936e9 * 1e3  # one context token's KV at full peak, ms
FIXED_MS = 0.05


def true_step(kv_tokens: float, weight_scale: float, kv_scale: float) -> float:
    return FIXED_MS + WEIGHT_MS / weight_scale + kv_tokens * KV_MS_PER_TOKEN / kv_scale


def terms(dataset: str, shape: Shape, measured_ms: float) -> DecodeTerms:
    kv = mean_kv_context_tokens(shape, 128)
    return DecodeTerms(
        dataset=dataset,
        shape=shape,
        kv_tokens=kv,
        measured_ms=measured_ms,
        fixed_ms=FIXED_MS,
        weight_ms_at_full=WEIGHT_MS,
        kv_ms_at_full=kv * KV_MS_PER_TOKEN,
    )


SWEEP = [Shape(b, 512) for b in (1, 2, 4, 8, 16, 24, 32)]
LONG = [Shape(1, 4096), Shape(1, 8192), Shape(1, 16384), Shape(2, 8192), Shape(4, 4096)]


def synthetic(weight_scale: float, kv_scale: float) -> list[DecodeTerms]:
    return [
        terms(name, s, true_step(mean_kv_context_tokens(s, 128), weight_scale, kv_scale))
        for name, shapes in (("sweep", SWEEP), ("long", LONG))
        for s in shapes
    ]


def test_mean_kv_context_matches_the_simulator_average_context() -> None:
    # batch x (prompt + (decode + 1) / 2): the static decode roofline's mean context.
    assert mean_kv_context_tokens(Shape(32, 512), 128) == pytest.approx(32 * 576.5)


def test_after_mode_recovers_the_kv_scale_with_weights_fixed() -> None:
    fit = fit_kv_read(synthetic(0.8383, 0.45), KvFitMode.AFTER, weight_scale=0.8383)
    assert fit.kv_read_bandwidth_scale == pytest.approx(0.45, rel=1e-9)
    assert fit.decode_memory_bandwidth_scale == 0.8383
    assert fit.mode is KvFitMode.AFTER
    assert fit.datasets == ("long", "sweep")


def test_joint_mode_recovers_both_scales() -> None:
    fit = fit_kv_read(synthetic(0.81, 0.47), KvFitMode.JOINT, weight_scale=0.8383)
    assert fit.decode_memory_bandwidth_scale == pytest.approx(0.81, rel=1e-9)
    assert fit.kv_read_bandwidth_scale == pytest.approx(0.47, rel=1e-9)


def test_joint_mode_needs_kv_variation() -> None:
    flat = [terms("one", Shape(8, 512), 20.0), terms("one", Shape(8, 512), 20.1)]
    with pytest.raises(FitError, match="identif"):
        fit_kv_read(flat, KvFitMode.JOINT, weight_scale=0.8383)


def test_rejects_a_non_positive_fitted_scale() -> None:
    # Measured faster than the weight read alone: no positive KV scale fits.
    fast = [terms("x", s, 10.0) for s in SWEEP]
    with pytest.raises(FitError, match="positive"):
        fit_kv_read(fast, KvFitMode.AFTER, weight_scale=0.8383)


def test_rejects_an_empty_fit() -> None:
    with pytest.raises(FitError):
        fit_kv_read([], KvFitMode.AFTER, weight_scale=0.8383)


def test_predict_uses_both_scales() -> None:
    sample = synthetic(0.8383, 0.45)[0]
    assert sample.predict(0.8383, 0.45) == pytest.approx(sample.measured_ms)
    assert sample.predict(0.8383, 0.8383) < sample.measured_ms


def test_holdout_fits_on_the_other_datasets() -> None:
    samples = synthetic(0.8383, 0.45)
    folds = holdout_folds(samples, KvFitMode.AFTER, weight_scale=0.8383)
    assert [held for held, _ in folds] == ["long", "sweep"]
    for held, fit in folds:
        assert held not in fit.datasets
        assert fit.kv_read_bandwidth_scale == pytest.approx(0.45, rel=1e-9)


def test_holdout_needs_two_datasets() -> None:
    only = [s for s in synthetic(0.8383, 0.45) if s.dataset == "sweep"]
    with pytest.raises(FitError, match="two datasets"):
        holdout_folds(only, KvFitMode.AFTER, weight_scale=0.8383)


def test_error_trend_is_the_slope_per_10k_kv_tokens() -> None:
    trend = error_trend([(0.0, -1.0), (10_000.0, -3.0), (20_000.0, -5.0)])
    assert trend.slope_pct_per_10k_tokens == pytest.approx(-2.0)
    assert trend.mean_abs_pct == pytest.approx(3.0)
    assert error_trend([(5.0, 1.0)]).slope_pct_per_10k_tokens == 0.0


def _measured(shape: Shape, step: float) -> StaticMeasurement:
    return StaticMeasurement(
        shape=shape, decode_tokens=128, tp=1, pp=1, prefill_ms=100.0, decode_ms=step * 128,
        decode_ms_per_step=step, end_to_end_ms=step * 128 + 100.0, prefill_spread_ms=None, e2e_spread_ms=None,
    )  # fmt: skip


def _row(shape: Shape, step: float) -> StaticSimRow:
    return StaticSimRow(
        shape=shape, decode_tokens=128, prefill_ms=100.0, decode_ms=step * 128,
        end_to_end_ms=step * 128 + 100.0, feasible=True,
    )  # fmt: skip


def test_decode_terms_decompose_three_simulator_runs() -> None:
    shape = Shape(16, 512)
    kv = mean_kv_context_tokens(shape, 128)
    probes = ProbeScales(weight_scale=0.8383, kv_scale=0.8383, weight_probe=0.41915, kv_probe=0.41915)

    def sim(w: float, k: float) -> StaticSimRow:
        return _row(shape, true_step(kv, w, k))

    got = decode_terms(
        "sweep",
        _measured(shape, 21.0),
        base=sim(0.8383, 0.8383),
        kv_probe=sim(0.8383, 0.41915),
        weight_probe=sim(0.41915, 0.8383),
        probes=probes,
    )
    assert got.kv_ms_at_full == pytest.approx(kv * KV_MS_PER_TOKEN, rel=1e-9)
    assert got.weight_ms_at_full == pytest.approx(WEIGHT_MS, rel=1e-9)
    assert got.fixed_ms == pytest.approx(FIXED_MS, abs=1e-9)
    assert got.kv_tokens == pytest.approx(kv)


def test_decode_terms_require_a_kv_term() -> None:
    shape = Shape(1, 512)
    probes = ProbeScales(weight_scale=0.8383, kv_scale=0.8383, weight_probe=0.41915, kv_probe=0.41915)
    flat = _row(shape, 19.45)
    with pytest.raises(FitError, match="KV"):
        decode_terms(
            "sweep", _measured(shape, 19.5), base=flat, kv_probe=flat, weight_probe=_row(shape, 38.9),
            probes=probes,
        )  # fmt: skip


def test_probe_scales_must_differ_from_the_base() -> None:
    with pytest.raises(FitError):
        ProbeScales(weight_scale=0.8, kv_scale=0.8, weight_probe=0.8, kv_probe=0.4)
