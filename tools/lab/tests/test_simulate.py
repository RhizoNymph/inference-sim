from __future__ import annotations

import dataclasses
import math
import tomllib
from typing import Any

import pytest

from labharness.simulate import (
    DefaultCalibration,
    KvBudget,
    ScalarCalibration,
    SimPhase,
    kv_budget,
    parse_serving_payload,
    serving_workload_toml,
    static_workload_toml,
)
from labharness.spec import FixedBatch, ServingWorkload, Shape, SimAdmission, parse_experiment
from tests.conftest import SPECS

STATIC = parse_experiment(SPECS / "rtx3090_qwen7b_static_pp2.toml")
SERVING = parse_experiment(SPECS / "rtx3090_qwen7b_serving_pp1.toml")


def test_static_workload_shape_and_calibration() -> None:
    text = static_workload_toml(
        STATIC.model, STATIC.parallelism, Shape(8, 2048), 128, SimPhase.DECODE, ScalarCalibration(0.85, 0.84)
    )
    data = tomllib.loads(text)
    assert data["model"]["layers"] == 28 and data["model"]["ffn_hidden_size"] == 18944
    assert data["request"] == {
        "batch_size": 8, "prompt_tokens": 2048, "decode_tokens": 128,
        "max_sequence_tokens": 2176, "phase": "decode",
    }  # fmt: skip
    assert data["calibration"] == {"compute_efficiency": 0.85, "decode_memory_bandwidth_scale": 0.84}
    assert data["search"] == {
        "tensor_ranks": [1],
        "pipeline_ranks": [2],
        "expert_ranks": [1],
        "data_ranks": [1],
    }
    default = tomllib.loads(
        static_workload_toml(
            STATIC.model, STATIC.parallelism, Shape(1, 512), 128, SimPhase.PREFILL, DefaultCalibration()
        )
    )
    assert "calibration" not in default


def _serving(rate: float, cap: SimAdmission = SimAdmission.ENGINE) -> dict[str, Any]:
    assert isinstance(SERVING.workload, ServingWorkload)
    wl = dataclasses.replace(SERVING.workload, sim_admission=cap)
    text = serving_workload_toml(
        SERVING, wl, rate=rate, reference_batch=6, request_count=200, kv=KvBudget(82864, 5179),
        calibration=DefaultCalibration(),
    )  # fmt: skip
    return tomllib.loads(text)


def test_serving_workload_mirrors_bench_serve() -> None:
    data = _serving(2.0)
    traffic = data["serving"]["traffic"]
    assert traffic["arrival"] == "poisson" and traffic["arrival_rate_per_s"] == 2.0
    assert traffic["request_count"] == 200 and traffic["arrival_seed"] == 0
    assert (
        traffic["prompt_tokens"] == [512]
        and traffic["decode_tokens"] == [128]
        and traffic["batch_sizes"] == [1]
    )
    # vLLM's chunked-prefill budget and sequence cap.
    assert traffic["max_prefill_batch_tokens"] == 2048 and traffic["max_prefill_chunk_tokens"] == 2048
    assert traffic["max_decode_sequences"] == 64 and traffic["max_decode_batch_tokens"] == 64
    assert traffic["max_resident_tokens"] == 82864 and traffic["max_kv_blocks"] == 5179
    assert traffic["prefix_cache_hit_rate"] == 0.0
    assert data["request"]["batch_size"] == 6
    assert data["serving"]["mode"] == "colocated" and data["serving"]["prefill_nodes"] == [0]


def test_infinite_rate_is_a_zero_gap_burst() -> None:
    traffic = _serving(math.inf)["serving"]["traffic"]
    assert traffic["arrival"] == "fixed" and traffic["arrival_gap_ms"] == 0.0
    assert "arrival_rate_per_s" not in traffic


def test_uncapped_admission() -> None:
    traffic = _serving(8.0, SimAdmission.UNCAPPED)["serving"]["traffic"]
    assert traffic["max_decode_sequences"] == 200
    assert traffic["max_resident_tokens"] == 200 * 640 and traffic["max_kv_blocks"] == 200 * 640 // 16


def test_kv_budget() -> None:
    measured = kv_budget(SERVING, 24.0, 82864)
    assert measured == KvBudget(tokens=82864, blocks=5179)
    estimated = kv_budget(SERVING, 24.0)
    # vLLM-profiler estimate (kv_estimate.py); vLLM logged 82,864 tokens for this spec.
    assert estimated.tokens == pytest.approx(82864, rel=0.05)
    assert estimated.tokens % 16 == 0


def test_parse_serving_payload() -> None:
    payload = {
        "results": [
            {
                "feasible": True,
                "status": "ok",
                "metrics": {
                    "ttft_ms": 10.0, "ttft_p50_ms": 9.0, "ttft_p99_ms": 20.0,
                    "tpot_ms": 5.0, "tpot_p50_ms": 5.0, "tpot_p99_ms": 6.0,
                    "itl_ms": None, "itl_p50_ms": None, "itl_p99_ms": None,
                    "e2el_ms": 700.0, "e2el_p50_ms": 690.0, "e2el_p99_ms": 900.0,
                    "throughput_tokens_per_s": 256.0,
                },
            }
        ]
    }  # fmt: skip
    result = parse_serving_payload(payload, rate=2.0, reference_batch=4, request_count=200, output_len=128)
    assert result.ttft is not None and result.ttft.median_ms == 9.0
    assert result.itl is None
    assert result.request_throughput == pytest.approx(2.0)
    assert result.to_json()["request_rate"] == "2"


def test_serving_workload_runs_on_the_iteration_engine() -> None:
    # The simulator's iteration engine needs a colocated pool, continuous
    # prefill and decode batching, and one parallelism config for both phases.
    data = _serving(4.0)
    serving = data["serving"]
    assert serving["mode"] == "colocated" and serving["prefill_nodes"] == serving["decode_nodes"]
    traffic = serving["traffic"]
    assert traffic["prefill_batching"] == "continuous" and traffic["decode_batching"] == "continuous"
    assert serving["prefill_search"] == serving["decode_search"]
    assert serving["prefill_search"]["data_ranks"] == [1]


def test_checked_in_serving_spec_uses_a_unit_reference_batch() -> None:
    # The engine prices steps from their composition; the reference batch only
    # sizes static feasibility scores, so the spec no longer needs Little's law.
    assert isinstance(SERVING.workload, ServingWorkload)
    assert SERVING.workload.sim_reference_batch == FixedBatch(1)
