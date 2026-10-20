from __future__ import annotations

import tomllib

import pytest

from labharness.errors import FitError
from labharness.kv_read_fit import KvFitMode, KvReadFit
from labharness.kv_read_profile import HoldoutSummary, KvProfileInputs, render_kv_read_profile
from labharness.simulate import ProfileOverrideCalibration, SimPhase, static_workload_toml
from labharness.spec import Shape, parse_experiment
from tests.conftest import SPECS

BASE = """# Base profile: earlier provenance comment
schema_version = 1

[profile]
name = "base"

[calibration]
compute_efficiency_curve = [[128, 0.65], [4096, 0.87]]
compute_efficiency = 0.849357
decode_memory_bandwidth_scale = 0.838309

[[benchmarks]]
name = "b"
measured_ms = 1.0
"""


def _fit(mode: KvFitMode, weight: float = 0.838309) -> KvReadFit:
    return KvReadFit(
        mode=mode,
        decode_memory_bandwidth_scale=weight,
        kv_read_bandwidth_scale=0.452,
        base_decode_memory_bandwidth_scale=0.838309,
        datasets=("long", "sweep"),
        sample_count=17,
    )


HOLDOUT = (
    HoldoutSummary(held_out="sweep", kv_read_bandwidth_scale=0.43, decode_step_mean_abs_pct=0.9),
    HoldoutSummary(held_out="long", kv_read_bandwidth_scale=0.47, decode_step_mean_abs_pct=1.1),
)


def _inputs(base: str, fit: KvReadFit) -> KvProfileInputs:
    return KvProfileInputs(
        base_text=base,
        base_label="lab-runs/x/base.toml",
        fit=fit,
        sources=("lab-runs/a/measured-b1-32.jsonl", "lab-runs/b/measured.jsonl"),
        holdout=HOLDOUT,
    )


def test_after_mode_adds_only_the_kv_scale() -> None:
    text = render_kv_read_profile(_inputs(BASE, _fit(KvFitMode.AFTER)))
    parsed = tomllib.loads(text)
    calibration = parsed["calibration"]
    assert calibration["kv_read_bandwidth_scale"] == 0.452
    assert calibration["decode_memory_bandwidth_scale"] == 0.838309
    assert calibration["compute_efficiency_curve"] == [[128, 0.65], [4096, 0.87]]
    assert parsed["benchmarks"][0]["name"] == "b"
    assert "lab-runs/x/base.toml" in text and "measured-b1-32.jsonl" in text
    # Holdout stats travel with the profile.
    assert "held out sweep" in text and "0.9%" in text
    assert "# Base profile: earlier provenance comment" in text


def test_joint_mode_replaces_the_weight_scale() -> None:
    text = render_kv_read_profile(_inputs(BASE, _fit(KvFitMode.JOINT, weight=0.8301)))
    calibration = tomllib.loads(text)["calibration"]
    assert calibration["decode_memory_bandwidth_scale"] == 0.8301
    assert calibration["kv_read_bandwidth_scale"] == 0.452
    assert text.count("decode_memory_bandwidth_scale =") == 1


def test_refuses_a_base_that_already_has_a_kv_scale() -> None:
    base = BASE.replace("[calibration]\n", "[calibration]\nkv_read_bandwidth_scale = 0.5\n")
    with pytest.raises(FitError, match="already"):
        render_kv_read_profile(_inputs(base, _fit(KvFitMode.AFTER)))


def test_joint_mode_needs_a_weight_scale_line_to_replace() -> None:
    base = BASE.replace("decode_memory_bandwidth_scale = 0.838309\n", "")
    with pytest.raises(FitError, match="decode_memory_bandwidth_scale"):
        render_kv_read_profile(_inputs(base, _fit(KvFitMode.JOINT)))


def test_profile_override_calibration_renders_profile_and_overrides(tmp_path) -> None:
    exp = parse_experiment(SPECS / "rtx3090_qwen7b_decode_batch_sweep.toml")
    profile = tmp_path / "p.toml"
    calibration = ProfileOverrideCalibration(
        profile, (("decode_memory_bandwidth_scale", 0.8383), ("kv_read_bandwidth_scale", 0.45))
    )
    data = tomllib.loads(
        static_workload_toml(exp.model, exp.parallelism, Shape(8, 512), 128, SimPhase.DECODE, calibration)
    )
    assert data["calibration_profile"] == {"path": str(profile.resolve())}
    assert data["calibration"] == {"decode_memory_bandwidth_scale": 0.8383, "kv_read_bandwidth_scale": 0.45}
    assert "kv0.4500" in calibration.label and calibration.label.startswith("profile-p-")
