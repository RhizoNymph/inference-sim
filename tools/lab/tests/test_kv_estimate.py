"""The vLLM KV-capacity estimate against the capacities vLLM logged on the 3090 lab."""

from __future__ import annotations

import dataclasses

import pytest

from labharness.kv_estimate import KvCapacityInputs, estimate_kv_tokens

QWEN7B = KvCapacityInputs(
    hbm_gib_per_gpu=24.0, gpu_memory_utilization=0.85, max_num_batched_tokens=2048, parameters_gb=15.23,
    layers=28, hidden_size=3584, ffn_hidden_size=18944, kv_heads=4, head_dim=128, dtype_bytes=2,
    kv_dtype_bytes=2, tp=1, pp=1,
)  # fmt: skip
QWEN14B_PP2 = KvCapacityInputs(
    hbm_gib_per_gpu=24.0, gpu_memory_utilization=0.85, max_num_batched_tokens=16384, parameters_gb=29.54,
    layers=48, hidden_size=5120, ffn_hidden_size=13824, kv_heads=8, head_dim=128, dtype_bytes=2,
    kv_dtype_bytes=2, tp=1, pp=2,
)  # fmt: skip

# (run, inputs, "GPU KV cache size" vLLM logged)
LOGGED = [
    ("2026-09-30-serving-baseline", QWEN7B, 82_864),
    (
        "2026-10-01-qwen7b-decode-batch-sweep",
        dataclasses.replace(QWEN7B, max_num_batched_tokens=32768),
        24_896,
    ),
    (
        "2026-10-01-qwen7b-prefill-token-sweep",
        dataclasses.replace(QWEN7B, max_num_batched_tokens=16384),
        63_552,
    ),
    (
        "2026-09-30-qwen7b-static-longctx-node1",
        dataclasses.replace(QWEN7B, max_num_batched_tokens=16384, gpu_memory_utilization=0.90),
        85_568,
    ),
    ("2026-09-30-qwen14b-static-pp2", QWEN14B_PP2, 43_440),
]


@pytest.mark.parametrize(("run", "inputs", "logged"), LOGGED, ids=[run for run, _, _ in LOGGED])
def test_estimate_is_within_ten_percent_of_vllm(run: str, inputs: KvCapacityInputs, logged: int) -> None:
    estimate = estimate_kv_tokens(inputs)
    assert abs(estimate - logged) / logged < 0.10, f"{run}: {estimate} vs {logged}"


def test_large_token_budget_shrinks_the_cache() -> None:
    small = estimate_kv_tokens(QWEN7B)
    large = estimate_kv_tokens(dataclasses.replace(QWEN7B, max_num_batched_tokens=32768))
    assert large < small / 2.5


def test_no_memory_left_gives_zero() -> None:
    assert estimate_kv_tokens(dataclasses.replace(QWEN7B, gpu_memory_utilization=0.5)) == 0


def test_tensor_parallel_shards_weights_and_kv() -> None:
    tp2 = estimate_kv_tokens(dataclasses.replace(QWEN7B, tp=2))
    assert tp2 > 2 * estimate_kv_tokens(QWEN7B)
