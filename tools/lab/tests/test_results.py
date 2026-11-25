from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from labharness.errors import ResultParseError
from labharness.results import (
    count_static_results,
    load_bench_serve,
    load_static,
    parse_bench_serve,
    parse_kv_cache_tokens,
    parse_static_lines,
)
from labharness.spec import Shape

VLLM_NOISE = [
    "INFO 09-28 21:01:31 [utils.py:306] Using LBNHC KV cache layout.",
    "(EngineCore pid=191) INFO 09-28 21:01:48 [gpu_worker.py:625] Available KV cache memory: 4.43 GiB",
    "{not json at all",
    '{"event": "some other json object"}',
    "[1, 2, 3]",
    "",
]


def _record(batch: int, prompt: int, tagged: bool = True) -> str:
    record = {
        "batch": batch, "prompt": prompt, "decode": 128, "tp": 1, "pp": 1,
        "prefill_ms": 100.0, "decode_ms": 2000.0, "decode_ms_per_step": 15.625,
        "end_to_end_ms": 2100.0, "prefill_spread_ms": 1.0, "e2e_spread_ms": 2.0,
    }  # fmt: skip
    if tagged:
        record = {"record": "lab.static_batch.v1", **record}
    return json.dumps(record)


def test_static_parser_ignores_vllm_logs() -> None:
    lines = [VLLM_NOISE[0], _record(1, 512), *VLLM_NOISE[1:], _record(8, 512), '{"record": "lab.done.v1"}']
    parsed = parse_static_lines(lines, source="t")
    assert [m.shape for m in parsed] == [Shape(1, 512), Shape(8, 512)]
    assert parsed[0].decode_ms_per_step == 15.625


def test_static_parser_accepts_legacy_untagged_lines(fixtures: Path) -> None:
    parsed = load_static(fixtures / "static_pp1_legacy.jsonl")
    assert len(parsed) == 5
    first = parsed[0]
    assert (first.shape, first.prefill_ms, first.decode_ms_per_step, first.end_to_end_ms) == (
        Shape(1, 512),
        114.678,
        19.264,
        2562.248,
    )


def test_static_parser_rejects_foreign_record_tags() -> None:
    other = json.loads(_record(1, 512))
    other["record"] = "lab.sim_static.v1"
    assert parse_static_lines([json.dumps(other)], source="t") == []


def test_static_parser_rejects_duplicates_and_bad_numbers() -> None:
    with pytest.raises(ResultParseError, match="duplicate"):
        parse_static_lines([_record(1, 512), _record(1, 512)], source="t")
    bad = json.loads(_record(1, 512))
    bad["prefill_ms"] = "fast"
    with pytest.raises(ResultParseError, match="prefill_ms"):
        parse_static_lines([json.dumps(bad)], source="t")


def test_completion_counting() -> None:
    lines = [*VLLM_NOISE, _record(1, 512), VLLM_NOISE[0], _record(8, 512)]
    assert count_static_results(lines) == (2, False)
    assert count_static_results([*lines, '{"record": "lab.done.v1", "shapes": 2}']) == (2, True)
    # A naive line count would say 9 here.
    assert len(lines) == 9


def test_empty_static_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(VLLM_NOISE), encoding="utf-8")
    with pytest.raises(ResultParseError, match="no static-batch result records"):
        load_static(path)


def test_bench_serve_real_output(fixtures: Path) -> None:
    """lab-runs/2026-09-30-serving-baseline/rate_2.json, vLLM 0.29.0."""
    m = load_bench_serve(fixtures / "vllm_bench_serve_rate_2.json")
    assert m.request_rate == 2.0
    assert m.burstiness == 1.0
    assert (m.num_prompts, m.completed, m.failed) == (200, 200, 0)
    assert m.total_input_tokens == 102400 and m.total_output_tokens == 25600
    assert m.duration_s == pytest.approx(102.6639, abs=1e-3)
    assert m.request_throughput == pytest.approx(1.9481, abs=1e-4)
    assert m.output_throughput == pytest.approx(249.357, abs=1e-3)
    assert m.total_token_throughput == pytest.approx(1246.787, abs=1e-3)
    assert m.ttft.median_ms == pytest.approx(168.0437, abs=1e-3)
    assert m.ttft.p99_ms == pytest.approx(384.7903, abs=1e-3)
    assert m.tpot.mean_ms == pytest.approx(25.1819, abs=1e-3)
    assert m.itl.median_ms == pytest.approx(19.7810, abs=1e-3)
    assert m.itl.p99_ms == pytest.approx(129.5319, abs=1e-3)
    assert m.e2el is not None and m.e2el.median_ms == pytest.approx(3308.7776, abs=1e-3)
    assert m.max_concurrent_requests == 19


def test_bench_serve_infinite_rate(fixtures: Path) -> None:
    m = load_bench_serve(fixtures / "vllm_bench_serve_rate_inf.json")
    assert math.isinf(m.request_rate)
    assert m.max_concurrent_requests == 200


def test_bench_serve_missing_fields(fixtures: Path) -> None:
    payload = json.loads((fixtures / "vllm_bench_serve_rate_2.json").read_text(encoding="utf-8"))
    del payload["p99_tpot_ms"]
    with pytest.raises(ResultParseError, match="p99_tpot_ms"):
        parse_bench_serve(payload, source="t")
    payload = json.loads((fixtures / "vllm_bench_serve_rate_2.json").read_text(encoding="utf-8"))
    payload["completed"] = 0
    with pytest.raises(ResultParseError, match="zero requests"):
        parse_bench_serve(payload, source="t")
    payload["completed"] = 200
    payload["request_rate"] = "fast"
    with pytest.raises(ResultParseError, match="request_rate"):
        parse_bench_serve(payload, source="t")


def test_bench_serve_without_e2el(fixtures: Path) -> None:
    payload = json.loads((fixtures / "vllm_bench_serve_rate_2.json").read_text(encoding="utf-8"))
    for key in [k for k in payload if k.endswith("_e2el_ms")]:
        del payload[key]
    assert parse_bench_serve(payload, source="t").e2el is None


def test_kv_cache_tokens_from_server_log() -> None:
    log = (
        "(EngineCore pid=191) INFO 09-30 23:01:48 [kv_cache_utils.py:2032] GPU KV cache size: "
        "82,864 tokens, Maximum concurrency for 4,096 tokens per request: 20.23x\n"
    )
    assert parse_kv_cache_tokens(log) == 82864
    assert parse_kv_cache_tokens("nothing here") is None
