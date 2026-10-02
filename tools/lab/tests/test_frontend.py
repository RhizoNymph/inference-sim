from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from labharness.errors import FitError, ResultParseError
from labharness.frontend import FrontendSample, fit_frontend_latency, load_frontend_ttfts


def test_two_prompt_lengths_give_an_exact_line() -> None:
    fit = fit_frontend_latency([FrontendSample(16, 27.05, 21.73), FrontendSample(512, 126.88, 114.37)])
    slope_ms = ((126.88 - 114.37) - (27.05 - 21.73)) / (512 - 16)
    assert math.isclose(fit.per_prompt_token_us, slope_ms * 1000.0)
    assert math.isclose(fit.fixed_us, ((27.05 - 21.73) - slope_ms * 16) * 1000.0)
    assert math.isclose(fit.latency_ms(512), 126.88 - 114.37)


def test_one_prompt_length_gives_a_constant() -> None:
    fit = fit_frontend_latency([FrontendSample(512, 126.0, 114.0)])
    assert fit.per_prompt_token_us == 0.0
    assert math.isclose(fit.fixed_us, 12_000.0)


def test_rejects_negative_overhead() -> None:
    with pytest.raises(FitError):
        fit_frontend_latency([FrontendSample(512, 100.0, 114.0)])
    with pytest.raises(FitError):
        # Overhead shrinking with prompt length: negative per-token term.
        fit_frontend_latency([FrontendSample(16, 40.0, 20.0), FrontendSample(512, 120.0, 114.0)])
    with pytest.raises(FitError):
        fit_frontend_latency([])


def _bench(path: Path, median_ttft: float) -> None:
    payload = {
        "request_rate": 0.5, "completed": 40, "duration": 80.0, "total_input_tokens": 100,
        "total_output_tokens": 40, "request_throughput": 0.5, "output_throughput": 0.5,
        "mean_ttft_ms": median_ttft, "median_ttft_ms": median_ttft, "p99_ttft_ms": median_ttft,
        "mean_tpot_ms": None, "median_tpot_ms": None, "p99_tpot_ms": None,
    }  # fmt: skip
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_single_output_token_runs_by_prompt_length(tmp_path: Path) -> None:
    _bench(tmp_path / "in512_out1.json", 126.9)
    _bench(tmp_path / "in16_out1.json", 27.0)
    _bench(tmp_path / "in16_out2.json", 99.0)  # two output tokens: ignored
    assert load_frontend_ttfts(tmp_path) == {16: 27.0, 512: 126.9}


def test_loading_an_empty_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(ResultParseError):
        load_frontend_ttfts(tmp_path)


def test_rejects_a_result_without_ttft(tmp_path: Path) -> None:
    (tmp_path / "in16_out1.json").write_text(json.dumps({"median_ttft_ms": None}), encoding="utf-8")
    with pytest.raises(ResultParseError):
        load_frontend_ttfts(tmp_path)
