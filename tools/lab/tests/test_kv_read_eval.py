from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from labharness.errors import FitError
from labharness.kv_read_eval import (
    KvDataset,
    fit_kv_read_datasets,
    outcome_json,
    outcome_markdown,
    select_measurements,
)
from labharness.kv_read_fit import KvFitMode, mean_kv_context_tokens
from labharness.results import StaticMeasurement
from labharness.simulate import SimRunner
from labharness.spec import Shape, parse_experiment
from tests.conftest import BINARY, SPECS

SWEEP = parse_experiment(SPECS / "rtx3090_qwen7b_decode_batch_sweep.toml")
LONG = parse_experiment(SPECS / "rtx3090_qwen7b_static_longctx_node1.toml")

PROFILE = """schema_version = 1
[profile]
name = "kv-test"
[calibration]
compute_efficiency = 0.8494
decode_memory_bandwidth_scale = 0.8383
"""

# Synthetic "real" hardware: weights at 0.8383 of peak, KV reads at 0.45.
WEIGHT_MS = 15.23e9 / 936e9 * 1e3
KV_MS_PER_TOKEN = 57344 / 936e9 * 1e3


def _step(shape: Shape) -> float:
    return WEIGHT_MS / 0.8383 + mean_kv_context_tokens(shape, 128) * KV_MS_PER_TOKEN / 0.45


def _measured(shape: Shape) -> StaticMeasurement:
    step = _step(shape)
    return StaticMeasurement(
        shape=shape, decode_tokens=128, tp=1, pp=1, prefill_ms=100.0, decode_ms=step * 128,
        decode_ms_per_step=step, end_to_end_ms=step * 128 + 100.0, prefill_spread_ms=None, e2e_spread_ms=None,
    )  # fmt: skip


def test_select_measurements_drops_batches_above_the_max() -> None:
    rows = [_measured(Shape(b, 512)) for b in (1, 32, 40, 64)]
    kept, excluded = select_measurements(rows, max_batch=32)
    assert [m.shape.batch for m in kept] == [1, 32]
    assert excluded == (Shape(40, 512), Shape(64, 512))
    with pytest.raises(FitError):
        select_measurements(rows, max_batch=0)
    with pytest.raises(FitError):
        select_measurements([_measured(Shape(64, 512))], max_batch=32)


@pytest.mark.skipif(not BINARY.is_file(), reason="needs cargo build --release")
def test_fit_recovers_a_synthetic_kv_scale_through_the_simulator(tmp_path: Path) -> None:
    profile = tmp_path / "base.toml"
    profile.write_text(PROFILE, encoding="utf-8")

    def dataset(name: str, exp, shapes: list[Shape]) -> KvDataset:
        runner = SimRunner(
            binary=BINARY, cluster=exp.lab.cluster_toml, work_dir=tmp_path / name, concurrency=4
        )
        return KvDataset(name, exp, tuple(_measured(s) for s in shapes), (), runner, f"synthetic/{name}")

    datasets = [
        dataset("sweep", SWEEP, [Shape(b, 512) for b in (1, 8, 32)]),
        dataset("long", LONG, [Shape(1, 4096), Shape(1, 16384), Shape(4, 4096)]),
    ]
    outcome = asyncio.run(fit_kv_read_datasets(datasets, profile, 0.8383, KvFitMode.AFTER))
    assert outcome.fit.kv_read_bandwidth_scale == pytest.approx(0.45, rel=2e-3)
    for result in outcome.datasets:
        assert result.held_out.summary.decode_step_mean_abs_pct < 0.2
        assert result.before.summary.decode_step_mean_abs_pct > 1.0
        assert abs(result.trend(result.held_out).slope_pct_per_10k_tokens) < 0.1
    payload = outcome_json(outcome)
    assert payload["fit"] == outcome.fit.to_json()
    assert outcome.fit.mode is KvFitMode.AFTER
    text = outcome_markdown(outcome, "t", "p.toml", "base.toml")
    assert "| sweep |" in text and "### long" in text
