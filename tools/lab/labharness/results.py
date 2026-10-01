"""Parsers for measured results: static-batch JSON lines and `vllm bench serve` JSON.

Both parsers are strict about the data they keep and lenient about noise:
vLLM prints INFO logs to stdout, so a static-batch stream may contain arbitrary
non-JSON lines. Only lines that parse as JSON objects carrying the result
fields (and, when present, the `lab.static_batch.v1` record tag) count.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from labharness.errors import ResultParseError
from labharness.spec import Shape

STATIC_RECORD: Final = "lab.static_batch.v1"
DONE_RECORD: Final = "lab.done.v1"
_STATIC_FIELDS: Final = (
    "batch",
    "prompt",
    "decode",
    "prefill_ms",
    "decode_ms",
    "decode_ms_per_step",
    "end_to_end_ms",
)

# ---------------------------------------------------------------------------
# Static batch
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StaticMeasurement:
    shape: Shape
    decode_tokens: int
    tp: int
    pp: int
    prefill_ms: float
    decode_ms: float
    decode_ms_per_step: float
    end_to_end_ms: float
    prefill_spread_ms: float | None
    e2e_spread_ms: float | None


def _json_object(line: str) -> dict[str, object] | None:
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _num(record: Mapping[str, object], key: str, source: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ResultParseError(f"field {key!r} must be a finite number, got {value!r}", source=source)
    return float(value)


def number_or_none(record: Mapping[str, object], key: str) -> float | None:
    """A finite JSON number as float, else None (bools are not numbers)."""
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _opt_num(record: Mapping[str, object], key: str, source: str) -> float | None:
    return _num(record, key, source) if record.get(key) is not None else None


def is_static_record(record: Mapping[str, object]) -> bool:
    tag = record.get("record")
    if tag is not None and tag != STATIC_RECORD:
        return False
    return all(field in record for field in _STATIC_FIELDS)


def parse_static_lines(lines: Iterable[str], *, source: str) -> list[StaticMeasurement]:
    """Parse a static-batch stream; non-result lines (vLLM logs, sentinels) are skipped."""
    measurements: list[StaticMeasurement] = []
    for line in lines:
        record = _json_object(line)
        if record is None or not is_static_record(record):
            continue
        measurements.append(
            StaticMeasurement(
                shape=Shape(
                    batch=int(_num(record, "batch", source)), prompt=int(_num(record, "prompt", source))
                ),
                decode_tokens=int(_num(record, "decode", source)),
                tp=int(_opt_num(record, "tp", source) or 1),
                pp=int(_opt_num(record, "pp", source) or 1),
                prefill_ms=_num(record, "prefill_ms", source),
                decode_ms=_num(record, "decode_ms", source),
                decode_ms_per_step=_num(record, "decode_ms_per_step", source),
                end_to_end_ms=_num(record, "end_to_end_ms", source),
                prefill_spread_ms=_opt_num(record, "prefill_spread_ms", source),
                e2e_spread_ms=_opt_num(record, "e2e_spread_ms", source),
            )
        )
    shapes = [m.shape for m in measurements]
    if len(set(shapes)) != len(shapes):
        raise ResultParseError("duplicate shapes in static-batch results", source=source)
    return measurements


def count_static_results(lines: Iterable[str]) -> tuple[int, bool]:
    """(result record count, done-sentinel seen) - what completion detection uses."""
    count, done = 0, False
    for line in lines:
        record = _json_object(line)
        if record is None:
            continue
        if record.get("record") == DONE_RECORD:
            done = True
        elif is_static_record(record):
            count += 1
    return count, done


def load_static(path: Path) -> list[StaticMeasurement]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ResultParseError(f"cannot read: {error}", source=str(path)) from error
    measurements = parse_static_lines(lines, source=str(path))
    if not measurements:
        raise ResultParseError("no static-batch result records found", source=str(path))
    return measurements


# ---------------------------------------------------------------------------
# vllm bench serve
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatencyStats:
    mean_ms: float
    median_ms: float
    p99_ms: float


@dataclass(frozen=True, slots=True)
class ServeMeasurement:
    request_rate: float  # math.inf when the benchmark sent everything at t=0
    burstiness: float | None
    num_prompts: int
    completed: int
    failed: int
    duration_s: float
    total_input_tokens: int
    total_output_tokens: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float | None
    ttft: LatencyStats
    tpot: LatencyStats
    itl: LatencyStats
    e2el: LatencyStats | None
    max_concurrent_requests: int | None


SERVE_METRICS: Final = ("ttft", "tpot", "itl", "e2el")


def _stats(payload: Mapping[str, object], metric: str, source: str) -> LatencyStats:
    return LatencyStats(
        mean_ms=_num(payload, f"mean_{metric}_ms", source),
        median_ms=_num(payload, f"median_{metric}_ms", source),
        p99_ms=_num(payload, f"p99_{metric}_ms", source),
    )


def _rate(value: object, source: str) -> float:
    match value:
        case bool():
            raise ResultParseError("request_rate must be a number or 'inf'", source=source)
        case int() | float():
            return float(value)
        case str() if value.lower() in {"inf", "infinity"}:
            return math.inf
        case _:
            raise ResultParseError(f"request_rate must be a number or 'inf', got {value!r}", source=source)


def parse_bench_serve(payload: Mapping[str, object], *, source: str) -> ServeMeasurement:
    """Parse the JSON `vllm bench serve --save-result` writes.

    Requires the `--percentile-metrics ttft,tpot,itl[,e2el]` summary fields and
    `--metric-percentiles` including 99. Per-request arrays (ttfts, itls, ...)
    are ignored.
    """
    completed = int(_num(payload, "completed", source))
    if completed <= 0:
        raise ResultParseError("benchmark completed zero requests", source=source)
    e2el = _stats(payload, "e2el", source) if "mean_e2el_ms" in payload else None
    max_concurrent = payload.get("max_concurrent_requests")
    total_token = payload.get("total_token_throughput")
    num_prompts = payload.get("num_prompts", completed)
    return ServeMeasurement(
        request_rate=_rate(payload.get("request_rate"), source),
        burstiness=_opt_num(payload, "burstiness", source),
        num_prompts=int(num_prompts) if isinstance(num_prompts, int) else completed,
        completed=completed,
        failed=int(_opt_num(payload, "failed", source) or 0),
        duration_s=_num(payload, "duration", source),
        total_input_tokens=int(_num(payload, "total_input_tokens", source)),
        total_output_tokens=int(_num(payload, "total_output_tokens", source)),
        request_throughput=_num(payload, "request_throughput", source),
        output_throughput=_num(payload, "output_throughput", source),
        total_token_throughput=_num(payload, "total_token_throughput", source)
        if total_token is not None
        else None,
        ttft=_stats(payload, "ttft", source),
        tpot=_stats(payload, "tpot", source),
        itl=_stats(payload, "itl", source),
        e2el=e2el,
        max_concurrent_requests=int(max_concurrent) if isinstance(max_concurrent, int) else None,
    )


def load_bench_serve(path: Path) -> ServeMeasurement:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultParseError(f"cannot read bench serve JSON: {error}", source=str(path)) from error
    if not isinstance(payload, dict):
        raise ResultParseError("bench serve JSON must be an object", source=str(path))
    return parse_bench_serve(payload, source=str(path))


_KV_TOKENS_RE: Final = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def parse_kv_cache_tokens(log_text: str) -> int | None:
    """vLLM's logged KV capacity ("GPU KV cache size: 82,864 tokens"), last occurrence."""
    matches = _KV_TOKENS_RE.findall(log_text)
    return int(matches[-1].replace(",", "")) if matches else None
