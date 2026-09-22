"""The dense-decoder op walk and phase-sample composition."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence

import numpy as np

from converter.errors import CoverageError
from converter.spec import (
    DTYPE_BYTES,
    GENERATION_SAMPLE_COUNT,
    GpuSpec,
    GridSpec,
    ShapeBounds,
    ShardedModel,
)
from converter.tables import AllReduceCurve, AttentionTable, GemmTable

# ---------------------------------------------------------------------------
# Op walk
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class RankTables:
    """Every measured table a single tensor rank needs."""

    model: ShardedModel
    gemm: GemmTable
    context_attention: AttentionTable
    generation_attention: AttentionTable
    allreduce: AllReduceCurve
    gpu: GpuSpec

    def mem_op_ms(self, mem_bytes: float) -> float:
        """AISimulate `mem_op_latency_ms` for the non-tabulated memory ops."""
        scaling = max(self.gpu.mem_bw_empirical_scaling_factor, 1e-9)
        return (
            mem_bytes / (self.gpu.mem_bw * scaling)
            + self.gpu.mem_empirical_constant_latency
        ) * 1000.0

    def allreduce_ms(self, num_tokens: int) -> float:
        if self.model.tp <= 1:
            return 0.0
        return self.allreduce.query(float(num_tokens) * self.model.spec.hidden_size)

    def _elementwise_ms(self, num_tokens: int, dim_in: int, dim_out: int) -> float:
        # AISimulate `PyElementWise`: bytes_per_token = 2 * (dim_in + dim_out).
        return self.mem_op_ms(float(num_tokens) * 2.0 * (dim_in + dim_out))

    def _embedding_ms(self, num_tokens: int) -> float:
        spec = self.model.spec
        return self.mem_op_ms(
            float(num_tokens) * spec.hidden_size * DTYPE_BYTES[spec.dtype]
        )

    def _layer_ms(self, num_tokens: int, attention_ms: float) -> float:
        """One decoder layer, minus the phase-specific attention kernel."""
        model = self.model
        spec = model.spec
        h = spec.hidden_size
        total = self._elementwise_ms(num_tokens, 2 * h, 2 * h)  # add_norm_1
        total += self.gemm.query(num_tokens, model.qkv_n, h)  # qkv_gemm
        total += attention_ms
        total += self.gemm.query(num_tokens, h, model.proj_k)  # proj_gemm
        total += self._elementwise_ms(num_tokens, 2 * h, 2 * h)  # add_norm_2
        total += self.gemm.query(num_tokens, model.gate_ffn1_n, h)  # gate_ffn1
        total += self._elementwise_ms(
            num_tokens, model.gate_ffn1_n, model.ffn2_k
        )  # act_gate
        total += self.gemm.query(num_tokens, h, model.ffn2_k)  # ffn2_gemm
        total += 2.0 * self.allreduce_ms(num_tokens)  # ar_1 + ar_2
        return total

    def context_attention_ms(self, batch: int, prompt: int) -> float:
        """Measured FMHA plus AISimulate's fused rope / KV-write extras (x1.1)."""
        model = self.model
        spec = model.spec
        kernel = self.context_attention.query(batch, prompt)
        q_num = model.heads_per_gpu * spec.head_dim
        kv_num = model.kv_heads_per_gpu * spec.head_dim
        fmha_bytes = DTYPE_BYTES[spec.dtype]
        # `apply_rope` defaults to true; Llama sets `use_qk_norm=False`.
        rope = 2.0 * self.mem_op_ms(2.0 * q_num + 2.0 * kv_num)
        kv_write = 2.0 * self.mem_op_ms(kv_num * fmha_bytes)
        return kernel + 1.1 * (rope + kv_write)

    def prefill_ms(self, batch: int, prompt: int) -> float:
        """AISimulate context walk at `x = batch * isl` (logits GEMM at `x = batch`)."""
        model = self.model
        num_tokens = batch * prompt
        attention_ms = self.context_attention_ms(batch, prompt)
        total = self._embedding_ms(num_tokens)
        total += self.allreduce_ms(num_tokens)  # context_embedding_ar
        total += model.spec.layers * self._layer_ms(num_tokens, attention_ms)
        total += self.gemm.query(batch, model.logits_n, model.spec.hidden_size)
        return total

    def decode_step_constant_ms(self, batch: int) -> float:
        """Everything in one decode step that does not depend on context length."""
        model = self.model
        total = self._embedding_ms(batch)
        total += self.allreduce_ms(batch)  # generation_embedding_ar
        total += model.spec.layers * self._layer_ms(batch, 0.0)
        total += self.gemm.query(batch, model.logits_n, model.spec.hidden_size)
        return total

    def generation_attention_ms(self, batch: int, contexts: np.ndarray) -> np.ndarray:
        """Per-step generation attention, with AISimulate's +/-10% smoothing."""
        steps = contexts.astype(np.int64)
        s_min = np.maximum((steps * 0.9).astype(np.int64), 1)
        s_max = np.maximum((steps * 1.1).astype(np.int64), s_min)
        offsets = np.arange(GENERATION_SAMPLE_COUNT, dtype=np.int64)
        samples = s_min[:, None] + (s_max - s_min)[:, None] * offsets[None, :] // (
            GENERATION_SAMPLE_COUNT - 1
        )
        flat = self.generation_attention.query_many(
            batch, samples.reshape(-1).astype(np.float64)
        )
        return flat.reshape(samples.shape).mean(axis=1)

    def decode_series_ms(
        self, batch: int, prompt: int, decode_values: Sequence[int]
    ) -> dict[int, float]:
        """Exact per-step sums over contexts prompt+1 .. prompt+d for each d."""
        longest = max(decode_values)
        contexts = np.arange(prompt + 1, prompt + longest + 1, dtype=np.float64)
        attention = np.cumsum(self.generation_attention_ms(batch, contexts))
        const = self.decode_step_constant_ms(batch)
        layers = self.model.spec.layers
        return {d: d * const + layers * float(attention[d - 1]) for d in decode_values}

    def decode_ms(self, batch: int, prompt: int, decode: int) -> float:
        return self.decode_series_ms(batch, prompt, (decode,))[decode]


# ---------------------------------------------------------------------------
# Sample composition
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Sample:
    tp: int
    batch: int
    prompt: int
    decode: int
    target_ms: float
    features: tuple[float, ...]


def prefill_features(batch: int, prompt: int, tp: int) -> tuple[float, ...]:
    """Values for `PREFILL_FEATURES`, in that exact order."""
    tokens = batch * prompt
    return (float(tokens), tokens / tp, batch * prompt * prompt / tp)


def decode_features(batch: int, prompt: int, decode: int, tp: int) -> tuple[float, ...]:
    """Values for `DECODE_FEATURES`, in that exact order."""
    context = prompt + (decode + 1) / 2.0
    return (
        float(decode),
        decode / tp,
        batch * decode / tp,
        batch * decode * context / tp,
    )


def shape_bounds(decode_samples: Sequence[Sample]) -> ShapeBounds:
    """Bounding box of the request shapes that actually survived composition.

    The decode samples carry the full `(batch, prompt, decode)` triple for every
    surviving `(batch, prompt)`, so they are the superset.
    """
    batches = [sample.batch for sample in decode_samples]
    prompts = [sample.prompt for sample in decode_samples]
    decodes = [sample.decode for sample in decode_samples]
    sequences = [sample.prompt + sample.decode for sample in decode_samples]
    return ShapeBounds(
        min_batch_size=min(batches),
        max_batch_size=max(batches),
        min_prompt_tokens=min(prompts),
        max_prompt_tokens=max(prompts),
        min_decode_tokens=min(decodes),
        max_decode_tokens=max(decodes),
        min_sequence_tokens=min(sequences),
        max_sequence_tokens=max(sequences),
    )


def compose_samples(
    tables: Mapping[int, RankTables],
    grid: GridSpec,
    logger: logging.Logger,
) -> tuple[list[Sample], list[Sample]]:
    prefill: list[Sample] = []
    decode: list[Sample] = []
    skipped = 0
    for tp in grid.tensor_ranks:
        rank = tables[tp]
        m_max = float(rank.gemm.m_values[-1])
        for batch in grid.batch_sizes:
            for prompt in grid.prompt_tokens:
                if batch * prompt > m_max:
                    skipped += 1
                    logger.debug(
                        "prefill sample outside the measured GEMM m sweep",
                        extra={
                            "tp": tp,
                            "batch": batch,
                            "prompt": prompt,
                            "m": batch * prompt,
                            "m_max": int(m_max),
                        },
                    )
                    continue
                value = rank.prefill_ms(batch, prompt)
                prefill.append(
                    Sample(
                        tp=tp,
                        batch=batch,
                        prompt=prompt,
                        decode=1,
                        target_ms=value,
                        features=prefill_features(batch, prompt, tp),
                    )
                )
                series = rank.decode_series_ms(batch, prompt, grid.decode_tokens)
                for decode_tokens in grid.decode_tokens:
                    decode.append(
                        Sample(
                            tp=tp,
                            batch=batch,
                            prompt=prompt,
                            decode=decode_tokens,
                            target_ms=series[decode_tokens],
                            features=decode_features(batch, prompt, decode_tokens, tp),
                        )
                    )
    if not prefill or not decode:
        raise CoverageError(
            "no (batch, prompt) sample survived the measured-coverage filter",
            table="composition",
            query={"prefill": len(prefill), "decode": len(decode)},
        )
    logger.info(
        "composed phase samples",
        extra={"prefill": len(prefill), "decode": len(decode), "skipped": skipped},
    )
    return prefill, decode
