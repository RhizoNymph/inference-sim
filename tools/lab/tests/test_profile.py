from __future__ import annotations

import asyncio
import re
import tomllib
from pathlib import Path

import pytest

from labharness.fitting import ErrorSummary, ScalarFit
from labharness.profile import Provenance, render_profile
from labharness.simulate import ProfileCalibration, SimPhase, SimRunner, static_workload_toml
from labharness.spec import Shape, parse_experiment
from tests.conftest import BINARY, REPO_ROOT, SPECS
from tests.test_fitting import SHAPES, measured, simulated

EXP = parse_experiment(SPECS / "rtx3090_qwen7b_static_pp1.toml")
FIT = ScalarFit(
    compute_efficiency=0.8507,
    decode_memory_bandwidth_scale=0.838309,
    base_compute_efficiency=0.35,
    base_decode_memory_bandwidth_scale=1.0,
    shapes=tuple(SHAPES),
    prefill_ratios=(2.4,) * 5,
    decode_step_ratios=(0.84,) * 5,
)
LOO = ErrorSummary(5.2, 0.8, 1.3, 2.0, 5)
PROVENANCE = Provenance(
    run_dir="lab-runs/2026-09-28-static-batch",
    date="2026-09-28",
    backend_version="0.29.0",
    driver_version=None,
    cuda_version="13.0",
    nccl_version=None,
    environment_hash="sha256:abc",
)


def _render() -> str:
    reals = [measured(s, 100.0 + i, 19.0 + i, 2600.0 + i) for i, s in enumerate(SHAPES)]
    sims = [simulated(s, 101.0 + i, 19.1 + i, 2610.0 + i) for i, s in enumerate(SHAPES)]
    return render_profile(exp=EXP, fit=FIT, measured=reals, fitted_rows=sims, loo=LOO, provenance=PROVENANCE)


def _struct_fields(struct: str) -> set[str]:
    """Field names of a serde section struct in src/config/sections.rs."""
    source = (REPO_ROOT / "src" / "config" / "sections.rs").read_text(encoding="utf-8")
    body = source.split(f"pub(super) struct {struct} {{", 1)[1].split("\n}", 1)[0]
    return set(re.findall(r"pub\(super\) (\w+):", body))


def test_profile_is_valid_toml_with_expected_sections() -> None:
    data = tomllib.loads(_render())
    assert data["schema_version"] == 1
    assert data["calibration"] == {"compute_efficiency": 0.8507, "decode_memory_bandwidth_scale": 0.838309}
    profile = data["profile"]
    assert profile["serving_stack"] == "vllm"
    assert profile["backend_version"] == "0.29.0"
    assert profile["date"] == "2026-09-28"
    assert profile["cuda_version"] == "13.0"
    assert "driver_version" not in profile  # None values are omitted
    assert "leave-one-shape-out" in profile["notes"]
    assert data["valid_shape"] == {
        "min_batch_size": 1, "max_batch_size": 32, "min_prompt_tokens": 512, "max_prompt_tokens": 2048,
        "min_decode_tokens": 128, "max_decode_tokens": 128, "min_sequence_tokens": 640,
        "max_sequence_tokens": 2176,
    }  # fmt: skip


def test_benchmarks_cover_every_shape_and_phase() -> None:
    benchmarks = tomllib.loads(_render())["benchmarks"]
    assert len(benchmarks) == 15
    names = [b["name"] for b in benchmarks]
    assert names[:3] == [
        "prefill-tp1-pp1-b1-p512-d1",
        "decode-tp1-pp1-b1-p512-d128",
        "end_to_end-tp1-pp1-b1-p512-d128",
    ]
    prefill, decode, e2e = benchmarks[:3]
    assert (prefill["measured_ms"], prefill["predicted_ms"], prefill["decode_tokens"]) == (100.0, 101.0, 1)
    assert decode["measured_ms"] == pytest.approx(19.0 * 128)
    assert e2e["measured_ms"] == 2600.0 and e2e["sequence_tokens"] == 640


def test_emitted_field_names_exist_in_the_rust_schema() -> None:
    data = tomllib.loads(_render())
    assert set(data["profile"]) <= _struct_fields("CalibrationProfileMetadataSection")
    assert set(data["valid_shape"]) <= _struct_fields("CalibrationShapeRangeSection")
    assert set(data["calibration"]) <= _struct_fields("CalibrationSection")
    for benchmark in data["benchmarks"]:
        assert set(benchmark) <= _struct_fields("CalibrationBenchmarkPointSection")


@pytest.mark.skipif(not BINARY.is_file(), reason="needs cargo build --release")
def test_simulator_loads_emitted_profile(tmp_path: Path) -> None:
    profile = tmp_path / "profile.toml"
    profile.write_text(_render(), encoding="utf-8")
    runner = SimRunner(binary=BINARY, cluster=EXP.lab.cluster_toml, work_dir=tmp_path / "work", concurrency=1)
    text = static_workload_toml(
        EXP.model, EXP.parallelism, Shape(8, 512), 128, SimPhase.END_TO_END, ProfileCalibration(profile)
    )
    payload = asyncio.run(runner.run("check", text, asyncio.Semaphore(1)))
    block = payload["calibration"]
    assert isinstance(block, dict)
    assert block["compute_efficiency"] == pytest.approx(0.8507)
    assert block["decode_memory_bandwidth_scale"] == pytest.approx(0.838309)
    assert block["profile"]["name"] == "rtx3090-lab-vllm-qwen2.5-7b-instruct-tp1-pp1"
    assert block["applicability_status"] == "within_valid_shape"
