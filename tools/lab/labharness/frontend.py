"""Per-request API-server (frontend) latency from isolated-request benchmarks.

`vllm bench serve` at a low request rate with one output token measures the
client-observed TTFT of a request that never queues: frontend work
(tokenization, request handling, streaming the first token) plus one prefill
forward pass. Subtracting the engine-only prefill of the same prompt length
(the static-batch benchmark's batch-1 prefill) leaves the frontend latency.
With several prompt lengths the fit is a least-squares line
`fixed_us + per_prompt_token_us * prompt_tokens` (exact for two lengths).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from labharness.errors import FitError, ResultParseError

_SINGLE_TOKEN_FILE: Final = re.compile(r"in(\d+)_out1\.json")


@dataclass(frozen=True, slots=True)
class FrontendSample:
    prompt_tokens: int
    client_ttft_ms: float  # median TTFT of isolated one-token requests
    engine_prefill_ms: float  # batch-1 static prefill of the same prompt length

    @property
    def overhead_ms(self) -> float:
        return self.client_ttft_ms - self.engine_prefill_ms


@dataclass(frozen=True, slots=True)
class FrontendFit:
    fixed_us: float
    per_prompt_token_us: float
    samples: tuple[FrontendSample, ...] = ()

    def latency_ms(self, prompt_tokens: int) -> float:
        return (self.fixed_us + self.per_prompt_token_us * prompt_tokens) / 1000.0

    def to_json(self) -> dict[str, object]:
        return {
            "frontend_latency_us": round(self.fixed_us, 3),
            "frontend_latency_per_prompt_token_us": round(self.per_prompt_token_us, 4),
            "samples": [
                {
                    "prompt_tokens": s.prompt_tokens,
                    "client_ttft_ms": round(s.client_ttft_ms, 3),
                    "engine_prefill_ms": round(s.engine_prefill_ms, 3),
                    "overhead_ms": round(s.overhead_ms, 3),
                }
                for s in self.samples
            ],
        }


def fit_frontend_latency(samples: Sequence[FrontendSample]) -> FrontendFit:
    if not samples:
        raise FitError("need at least one frontend sample")
    xs = [float(s.prompt_tokens) for s in samples]
    ys = [s.overhead_ms for s in samples]
    if len(set(xs)) < 2:
        slope_ms, intercept_ms = 0.0, statistics.fmean(ys)
    else:
        slope_ms, intercept_ms = statistics.linear_regression(xs, ys)
    if intercept_ms < 0 or slope_ms < 0:
        raise FitError(
            "frontend latency fit is negative: client TTFT below the engine-only prefill",
            context={"fixed_ms": round(intercept_ms, 3), "per_token_ms": round(slope_ms, 5)},
        )
    return FrontendFit(
        fixed_us=intercept_ms * 1000.0, per_prompt_token_us=slope_ms * 1000.0, samples=tuple(samples)
    )


def load_frontend_ttfts(run_dir: Path) -> dict[int, float]:
    """Median TTFT per prompt length from `in<N>_out1.json` files in `run_dir`."""
    found: dict[int, float] = {}
    for path in sorted(run_dir.glob("in*_out1.json")):
        match = _SINGLE_TOKEN_FILE.fullmatch(path.name)
        if match is None:
            continue
        found[int(match.group(1))] = _median_ttft_ms(path)
    if not found:
        raise ResultParseError("no in<N>_out1.json frontend results", source=str(run_dir))
    return dict(sorted(found.items()))


def _median_ttft_ms(path: Path) -> float:
    """`median_ttft_ms` of a `vllm bench serve --save-result` file.

    One-output-token runs have no TPOT/ITL, so the full serving parser does
    not apply; only TTFT is read.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultParseError(f"cannot read bench serve JSON: {error}", source=str(path)) from error
    value = payload.get("median_ttft_ms") if isinstance(payload, dict) else None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ResultParseError("median_ttft_ms must be a positive number", source=str(path))
    return float(value)
