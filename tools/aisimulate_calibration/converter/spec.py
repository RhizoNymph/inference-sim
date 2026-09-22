"""Typed model, system, grid and provenance configuration."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Final

from converter.errors import ConfigurationError

# ---------------------------------------------------------------------------
# Typed configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ModelSpec:
    """Dense decoder geometry.  Every field is a per-model constant."""

    name: str
    layers: int
    hidden_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    dtype: str
    kv_dtype: str
    parameters_gb: float

    def __post_init__(self) -> None:
        positive = {
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
        }
        for field, value in positive.items():
            if value <= 0:
                raise ConfigurationError(f"model.{field} must be positive, got {value}")
        if self.attention_heads % self.kv_heads != 0:
            raise ConfigurationError(
                f"attention_heads ({self.attention_heads}) must be a multiple of "
                f"kv_heads ({self.kv_heads})"
            )

    def sharded(self, tp: int) -> ShardedModel:
        # AISimulate `BaseModel.__init__` asserts `num_heads % tp_size == 0` and
        # floor-divides the other widths; refusing an inexact divide keeps the
        # emitted profile from silently describing a truncated model.
        for field, value in (
            ("attention_heads", self.attention_heads),
            ("intermediate_size", self.intermediate_size),
            ("vocab_size", self.vocab_size),
        ):
            if value % tp != 0:
                raise ConfigurationError(
                    f"tensor rank {tp} does not evenly divide model.{field} ({value})"
                )
        return ShardedModel(spec=self, tp=tp)


@dataclasses.dataclass(frozen=True, slots=True)
class ShardedModel:
    """Per-GPU geometry after a vLLM-style tensor-parallel shard."""

    spec: ModelSpec
    tp: int

    @property
    def heads_per_gpu(self) -> int:
        return self.spec.attention_heads // self.tp

    @property
    def kv_heads_per_gpu(self) -> int:
        # AISimulate `BaseModel._num_kv_heads_per_gpu` CEIL-divides so a GQA
        # model with fewer KV heads than ranks still replicates one per GPU.
        return -(-self.spec.kv_heads // self.tp)

    @property
    def qkv_n(self) -> int:
        """Fused QKV output width (AISimulate `context_qkv_gemm`)."""
        return (
            self.spec.attention_heads * self.spec.head_dim // self.tp
            + self.spec.head_dim * self.kv_heads_per_gpu * 2
        )

    @property
    def proj_k(self) -> int:
        return self.spec.attention_heads * self.spec.head_dim // self.tp

    @property
    def gate_ffn1_n(self) -> int:
        """Fused gate+up output width (AISimulate `context_gate_ffn1_gemm`)."""
        return 2 * self.spec.intermediate_size // self.tp

    @property
    def ffn2_k(self) -> int:
        return self.spec.intermediate_size // self.tp

    @property
    def logits_n(self) -> int:
        return self.spec.vocab_size // self.tp


@dataclasses.dataclass(frozen=True, slots=True)
class GpuSpec:
    """The subset of an AISimulate system YAML this tool consumes."""

    mem_bw: float
    mem_bw_empirical_scaling_factor: float
    mem_empirical_constant_latency: float
    tc_flops: Mapping[str, float]
    nccl_version: str
    num_gpus_per_node: int

    def flops(self, dtype: str) -> float:
        value = self.tc_flops.get(dtype)
        if value is None or value <= 0.0:
            raise ConfigurationError(
                f"system spec has no positive {dtype}_tc_flops entry; "
                f"available: {sorted(self.tc_flops)}"
            )
        return value


@dataclasses.dataclass(frozen=True, slots=True)
class GridSpec:
    """The (batch, prompt, decode) sample grid and the tensor ranks to sweep."""

    batch_sizes: tuple[int, ...]
    prompt_tokens: tuple[int, ...]
    decode_tokens: tuple[int, ...]
    tensor_ranks: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class Provenance:
    """Everything that ends up in `[profile]` and the fit `source` strings."""

    system: str
    backend: str
    version: str
    commit: str
    date: str
    environment_hash: str


DTYPE_BYTES: Final[Mapping[str, float]] = {
    "bfloat16": 2.0,
    "float16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
}

MODEL_PRESETS: Final[Mapping[str, Mapping[str, object]]] = {
    "llama-3.1-8b": {
        "layers": 32,
        "hidden_size": 4096,
        "attention_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "parameters_gb": 16.0,
    },
    "llama-3.1-70b": {
        "layers": 80,
        "hidden_size": 8192,
        "attention_heads": 64,
        "kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 28672,
        "vocab_size": 128256,
        "parameters_gb": 141.0,
    },
    "qwen3-32b": {
        "layers": 64,
        "hidden_size": 5120,
        "attention_heads": 64,
        "kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 25600,
        "vocab_size": 151936,
        "parameters_gb": 66.0,
    },
}

PREFILL_FEATURES: Final[tuple[str, ...]] = (
    "batch_prompt_tokens_per_tensor_rank",
    "batch_prompt_tokens_squared_per_tensor_rank",
)
DECODE_FEATURES: Final[tuple[str, ...]] = (
    "decode_tokens_per_tensor_rank",
    "decode_batch_tokens_per_tensor_rank",
    "decode_batch_context_tokens_per_tensor_rank",
)

# AISimulate `gemm_engine_config`: 4 nearest collected (n, k) sites within a
# 2.0-octave log2 gate, inverse-square-distance util transfer.
GEMM_NN_SITES: Final = 4
GEMM_MAX_SITE_DISTANCE: Final = 2.0
# AISimulate `query_generation`: +/-10% 5-sample smoothing of the decode
# context length.
GENERATION_SAMPLE_COUNT: Final = 5
