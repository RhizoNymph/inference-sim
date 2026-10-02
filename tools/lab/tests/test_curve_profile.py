from __future__ import annotations

import tomllib

import pytest

from labharness.curve_profile import CurveProfileInputs, render_curve_profile
from labharness.efficiency_curve import CurveSample, fit_efficiency_curve
from labharness.errors import FitError
from labharness.frontend import FrontendFit
from labharness.spec import Shape

BASE = """schema_version = 1

[profile]
name = "base"

[calibration]
compute_efficiency = 0.849357
decode_memory_bandwidth_scale = 0.838309

[[benchmarks]]
name = "b"
measured_ms = 1.0
"""


def _fit():
    return fit_efficiency_curve(
        [
            CurveSample(Shape(1, 512), 512, 100.0, 78.0, None),
            CurveSample(Shape(1, 1024), 1024, 200.0, 178.0, None),
        ]
    )


def test_adds_the_curve_and_frontend_latency_to_the_base_calibration() -> None:
    text = render_curve_profile(
        CurveProfileInputs(
            base_text=BASE,
            base_label="lab-runs/x/base.toml",
            curve=_fit(),
            curve_source="lab-runs/y/measured.jsonl",
            frontend=FrontendFit(fixed_us=5070.0, per_prompt_token_us=14.5),
            frontend_source="lab-runs/z",
        )
    )
    parsed = tomllib.loads(text)
    calibration = parsed["calibration"]
    assert calibration["compute_efficiency"] == 0.849357
    assert calibration["compute_efficiency_curve"] == [[512, 0.78], [1024, 0.89]]
    assert calibration["frontend_latency_us"] == 5070.0
    assert calibration["frontend_latency_per_prompt_token_us"] == 14.5
    assert parsed["benchmarks"][0]["name"] == "b"
    assert "lab-runs/x/base.toml" in text and "lab-runs/y/measured.jsonl" in text


def test_frontend_is_optional() -> None:
    text = render_curve_profile(
        CurveProfileInputs(BASE, "base", _fit(), "src", frontend=None, frontend_source=None)
    )
    calibration = tomllib.loads(text)["calibration"]
    assert "frontend_latency_us" not in calibration
    assert len(calibration["compute_efficiency_curve"]) == 2


def test_refuses_a_base_that_already_has_a_curve() -> None:
    base = BASE.replace("[calibration]\n", "[calibration]\ncompute_efficiency_curve = [[1, 0.5], [2, 0.6]]\n")
    with pytest.raises(FitError, match="already"):
        render_curve_profile(CurveProfileInputs(base, "base", _fit(), "src", None, None))


def test_refuses_a_base_without_a_calibration_section() -> None:
    with pytest.raises(FitError, match=r"\[calibration\]"):
        render_curve_profile(
            CurveProfileInputs(
                'schema_version = 1\n[profile]\nname = "b"\n', "base", _fit(), "src", None, None
            )
        )
