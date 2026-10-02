"""vLLM's KV-cache capacity, estimated the way vLLM's memory profiler sizes it.

vLLM 0.29 sets aside `gpu_memory_utilization * visible GPU memory`, subtracts
the memory consumed after loading (weights plus non-torch allocations) and the
peak activation memory of a profiling forward pass over
`max_num_batched_tokens` tokens, and gives the rest to the KV cache. The peak
activation term grows with `max_num_batched_tokens`, so the budget shrinks
sharply at large token budgets: 82,864 tokens at 2048 batched tokens but
24,896 at 32,768 on the same RTX 3090 (Qwen2.5-7B, utilization 0.85).

Constants were fitted from the memory lines vLLM logs at startup
(`gpu_worker.py`: "Actual usage is X GiB for consumed memory (weights +
non-torch), Y GiB for peak activation") in five runs on the 3090 lab
(lab-runs/2026-09-30-serving-baseline, 2026-09-30-qwen7b-static-longctx-node1,
2026-09-30-qwen14b-static-pp2, 2026-10-01-qwen7b-prefill-token-sweep,
2026-10-01-qwen7b-decode-batch-sweep): peak activation is about
`0.825 GiB + 1.31 * (2 * ffn + hidden) * dtype_bytes * batched_tokens / tp`
(the gated MLP's two intermediate projections dominate). The estimate
reproduces vLLM's logged KV capacity within about 10% in all five runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

GIB: Final = float(2**30)


@dataclass(frozen=True, slots=True)
class VllmMemoryModel:
    # Visible memory = nominal HBM minus the CUDA context and driver reserve
    # (3090: 24 GiB nominal, torch reports 23.56 GiB total).
    cuda_context_gib: float = 0.44
    # Consumed after load beyond the raw weight bytes: vLLM buffers plus
    # non-torch allocations (14.42 GiB consumed for 14.18 GiB of weights).
    load_overhead_gib: float = 0.24
    activation_fixed_gib: float = 0.825
    activation_per_token_factor: float = 1.31


DEFAULT_VLLM_MEMORY: Final = VllmMemoryModel()


@dataclass(frozen=True, slots=True)
class KvCapacityInputs:
    hbm_gib_per_gpu: float  # nominal per-GPU HBM (the cluster's hbm_gb, read as GiB)
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    parameters_gb: float  # whole-model weight bytes, decimal GB
    layers: int
    hidden_size: int
    ffn_hidden_size: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int
    kv_dtype_bytes: int
    tp: int
    pp: int


def estimate_kv_tokens(inputs: KvCapacityInputs, model: VllmMemoryModel = DEFAULT_VLLM_MEMORY) -> int:
    """KV-cache tokens per model replica (every GPU holds its shard of each token)."""
    world = inputs.tp * inputs.pp
    usable_gib = inputs.gpu_memory_utilization * (inputs.hbm_gib_per_gpu - model.cuda_context_gib)
    weights_gib = inputs.parameters_gb * 1e9 / GIB / world + model.load_overhead_gib
    per_token_bytes = (
        model.activation_per_token_factor
        * (2 * inputs.ffn_hidden_size + inputs.hidden_size)
        * inputs.dtype_bytes
        / inputs.tp
    )
    activation_gib = model.activation_fixed_gib + per_token_bytes * inputs.max_num_batched_tokens / GIB
    kv_gib = usable_gib - weights_gib - activation_gib
    kv_bytes_per_token_per_gpu = (
        2 * inputs.layers * inputs.kv_heads * inputs.head_dim * inputs.kv_dtype_bytes / world
    )
    return max(0, math.floor(kv_gib * GIB / kv_bytes_per_token_per_gpu))
