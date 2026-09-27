"""Measured AISimulate op tables and their interpolation rules."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path

import duckdb
import numpy as np

from converter.errors import CoverageError, TableLoadError
from converter.spec import (
    DTYPE_BYTES,
    GEMM_MAX_SITE_DISTANCE,
    GEMM_NN_SITES,
    GpuSpec,
    ShardedModel,
)

# ---------------------------------------------------------------------------
# Measured tables
# ---------------------------------------------------------------------------


def _fetch(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, np.ndarray]:
    raw = con.execute(sql).fetchnumpy()
    out: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        array = value.filled(np.nan) if isinstance(value, np.ma.MaskedArray) else value
        out[key] = np.asarray(array)
    return out


def _relation(path: Path) -> str:
    quoted = str(path).replace("'", "''")
    if path.suffix == ".csv":
        return f"read_csv('{quoted}', header = true, auto_detect = true)"
    return f"read_parquet('{quoted}')"


@dataclasses.dataclass(frozen=True, slots=True)
class GemmTable:
    """Measured GEMM latencies for one quant mode, as scattered (n, k) sites."""

    m_values: np.ndarray  # (M,) ascending
    site_nk: np.ndarray  # (S, 2) int
    site_log: np.ndarray  # (S, 2) log2 of site_nk
    latency: np.ndarray  # (S, M) ms, SOL-clamped
    index: Mapping[tuple[int, int], int]
    dtype: str
    gpu: GpuSpec
    cache: dict[tuple[int, int, int], float] = dataclasses.field(default_factory=dict)

    def sol_ms(self, m: float, n: float, k: float) -> float:
        """AISimulate `gemm_sol_with_flops`: max(math bound, memory bound)."""
        elem = DTYPE_BYTES[self.dtype]
        sol_math = 2.0 * m * n * k / self.gpu.flops(self.dtype) * 1000.0
        sol_mem = elem * (m * n + m * k + n * k) / self.gpu.mem_bw * 1000.0
        return max(sol_math, sol_mem)

    def _curve(self, site: int, m: float) -> float:
        values = self.latency[site]
        lo = float(self.m_values[0])
        hi = float(self.m_values[-1])
        if m < lo or m > hi:
            raise CoverageError(
                f"GEMM m={m:g} outside the measured sweep [{lo:g}, {hi:g}]",
                table="gemm_perf",
                query={"m": m},
            )
        return float(np.interp(m, self.m_values, values))

    def query(self, m: int, n: int, k: int) -> float:
        cached = self.cache.get((m, n, k))
        if cached is not None:
            return cached
        value = self._query_uncached(m, n, k)
        self.cache[(m, n, k)] = value
        return value

    def _query_uncached(self, m: int, n: int, k: int) -> float:
        exact = self.index.get((n, k))
        if exact is not None:
            return self._curve(exact, m)

        q_log = np.array([math.log2(max(n, 1)), math.log2(max(k, 1))])
        distance = np.linalg.norm(self.site_log - q_log, axis=1)
        admissible = np.flatnonzero(distance <= GEMM_MAX_SITE_DISTANCE)
        if admissible.size == 0:
            nearest = int(np.argmin(distance))
            raise CoverageError(
                f"GEMM shape (n={n}, k={k}) has no measured site within "
                f"{GEMM_MAX_SITE_DISTANCE} octaves (nearest is "
                f"(n={self.site_nk[nearest, 0]}, k={self.site_nk[nearest, 1]}) at "
                f"{distance[nearest]:.3f} octaves)",
                table="gemm_perf",
                query={"m": m, "n": n, "k": k},
            )
        order = admissible[np.argsort(distance[admissible], kind="stable")]
        chosen = order[:GEMM_NN_SITES]

        weight_sum = 0.0
        util_acc = 0.0
        for site in chosen:
            site_n = int(self.site_nk[site, 0])
            site_k = int(self.site_nk[site, 1])
            latency = self._curve(int(site), m)
            sol = self.sol_ms(m, site_n, site_k)
            if not (latency > 0.0 and sol > 0.0):
                continue
            weight = 1.0 / (float(distance[site]) ** 2 + 1e-12)
            util_acc += weight * (sol / latency)
            weight_sum += weight
        if weight_sum <= 0.0:
            raise CoverageError(
                f"GEMM shape (n={n}, k={k}) has no usable neighbour site",
                table="gemm_perf",
                query={"m": m, "n": n, "k": k},
            )
        return self.sol_ms(m, n, k) / (util_acc / weight_sum)


def load_gemm_table(
    con: duckdb.DuckDBPyConnection, path: Path, dtype: str, gpu: GpuSpec
) -> GemmTable:
    rows = _fetch(
        con,
        f"SELECT m, n, k, latency FROM {_relation(path)} "
        f"WHERE gemm_dtype = '{dtype}' ORDER BY n, k, m",
    )
    if rows["m"].size == 0:
        raise TableLoadError(
            f"no gemm_perf rows for gemm_dtype='{dtype}'", source=str(path)
        )
    m_values = np.unique(rows["m"]).astype(np.float64)
    pairs = np.stack([rows["n"], rows["k"]], axis=1)
    site_nk, inverse = np.unique(pairs, axis=0, return_inverse=True)
    latency = np.full((site_nk.shape[0], m_values.size), np.nan)
    m_pos = {int(value): idx for idx, value in enumerate(m_values)}
    for site, m, value in zip(inverse, rows["m"], rows["latency"], strict=True):
        latency[int(site), m_pos[int(m)]] = float(value)

    table = GemmTable(
        m_values=m_values,
        site_nk=site_nk.astype(np.int64),
        site_log=np.log2(np.maximum(site_nk.astype(np.float64), 1.0)),
        latency=latency,
        index={(int(n), int(k)): idx for idx, (n, k) in enumerate(site_nk)},
        dtype=dtype,
        gpu=gpu,
    )
    _clamp_gemm_to_sol(table)
    _fill_curve_gaps(table.latency)
    return table


def _clamp_gemm_to_sol(table: GemmTable) -> None:
    """Mirror AISimulate `clamp_gemm_grids_to_sol`: raise measured to SOL."""
    m = table.m_values[None, :]
    n = table.site_nk[:, 0:1].astype(np.float64)
    k = table.site_nk[:, 1:2].astype(np.float64)
    elem = DTYPE_BYTES[table.dtype]
    sol_math = 2.0 * m * n * k / table.gpu.flops(table.dtype) * 1000.0
    sol_mem = elem * (m * n + m * k + n * k) / table.gpu.mem_bw * 1000.0
    sol = np.maximum(sol_math, sol_mem)
    measured = table.latency
    np.copyto(measured, np.maximum(measured, sol), where=np.isfinite(measured))


def _fill_curve_gaps(latency: np.ndarray) -> None:
    """Forward/backward fill NaN holes so every site curve is usable."""
    for row in latency:
        finite = np.flatnonzero(np.isfinite(row))
        if finite.size == 0 or finite.size == row.size:
            continue
        row[:] = np.interp(
            np.arange(row.size, dtype=np.float64),
            finite.astype(np.float64),
            row[finite],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionTable:
    """A `(batch, sequence) -> latency` slice of one attention head geometry."""

    name: str
    lane: str
    batch_values: np.ndarray
    seq_values: np.ndarray
    latency: np.ndarray  # (B, S) with NaN outside the collected staircase
    sqrt_seq_axis: bool

    def query(self, batch: int, seq: float) -> float:
        return float(self.query_many(batch, np.array([seq], dtype=np.float64))[0])

    def query_many(self, batch: int, seqs: np.ndarray) -> np.ndarray:
        b_idx = int(np.searchsorted(self.batch_values, batch))
        if b_idx >= self.batch_values.size or self.batch_values[b_idx] != batch:
            raise CoverageError(
                f"{self.name}: batch_size={batch} is not a collected point "
                f"(collected: {self.batch_values.astype(int).tolist()})",
                table=self.name,
                query={"batch_size": batch},
            )
        row = self.latency[b_idx]
        valid = np.flatnonzero(np.isfinite(row))
        if valid.size == 0:
            raise CoverageError(
                f"{self.name}: no collected sequence points at batch_size={batch}",
                table=self.name,
                query={"batch_size": batch},
            )
        axis = self.seq_values[valid]
        lo = float(axis[0])
        hi = float(axis[-1])
        low = float(seqs.min())
        high = float(seqs.max())
        if low < lo or high > hi:
            raise CoverageError(
                f"{self.name}: sequence range [{low:g}, {high:g}] leaves the "
                f"collected range [{lo:g}, {hi:g}] at batch_size={batch}",
                table=self.name,
                query={"batch_size": batch, "seq_min": low, "seq_max": high},
            )
        if self.sqrt_seq_axis:
            return np.interp(np.sqrt(seqs), np.sqrt(axis), row[valid])
        return np.interp(seqs, axis, row[valid])


def _load_attention_slice(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    name: str,
    seq_expr: str,
    predicate: str,
    sqrt_seq_axis: bool,
) -> AttentionTable:
    # AISimulate never merges points across kernel lanes (`kernel_source`); it
    # picks the first lane whose whole discrete slice exists, ranked by density.
    # Mirror the density tier: the lane with the most rows in this slice wins.
    lanes = _fetch(
        con,
        f"SELECT coalesce(kernel_source, 'default') AS lane, count(*) AS rows "
        f"FROM {_relation(path)} WHERE {predicate} GROUP BY lane "
        f"ORDER BY rows DESC, lane ASC",
    )
    if lanes["lane"].size == 0:
        raise CoverageError(
            f"{name}: no measured rows for the requested head geometry",
            table=name,
            query={"predicate": predicate},
        )
    lane = str(lanes["lane"][0])
    rows = _fetch(
        con,
        f"SELECT batch_size, {seq_expr} AS seq, latency "
        f"FROM {_relation(path)} WHERE {predicate} "
        f"AND coalesce(kernel_source, 'default') = '{lane}' "
        f"ORDER BY batch_size, seq",
    )
    if rows["batch_size"].size == 0:
        raise CoverageError(
            f"{name}: no measured rows for the requested head geometry",
            table=name,
            query={"predicate": predicate},
        )
    batch_values = np.unique(rows["batch_size"]).astype(np.float64)
    seq_values = np.unique(rows["seq"]).astype(np.float64)
    latency = np.full((batch_values.size, seq_values.size), np.nan)
    b_pos = {int(v): i for i, v in enumerate(batch_values)}
    s_pos = {int(v): i for i, v in enumerate(seq_values)}
    for b, s, value in zip(
        rows["batch_size"], rows["seq"], rows["latency"], strict=True
    ):
        latency[b_pos[int(b)], s_pos[int(s)]] = float(value)
    return AttentionTable(
        name=name,
        lane=lane,
        batch_values=batch_values,
        seq_values=seq_values,
        latency=latency,
        sqrt_seq_axis=sqrt_seq_axis,
    )


def load_context_attention(
    con: duckdb.DuckDBPyConnection, path: Path, model: ShardedModel
) -> AttentionTable:
    spec = model.spec
    return _load_attention_slice(
        con,
        path,
        name="context_attention_perf",
        seq_expr="isl",
        predicate=(
            f"num_heads = {model.heads_per_gpu} "
            f"AND num_key_value_heads = {model.kv_heads_per_gpu} "
            f"AND head_dim = {spec.head_dim} "
            f"AND window_size = 0 AND beam_width = 1 "
            f"AND attn_dtype = '{spec.dtype}' AND kv_cache_dtype = '{spec.kv_dtype}'"
        ),
        sqrt_seq_axis=True,
    )


def load_generation_attention(
    con: duckdb.DuckDBPyConnection, path: Path, model: ShardedModel
) -> AttentionTable:
    spec = model.spec
    return _load_attention_slice(
        con,
        path,
        name="generation_attention_perf",
        seq_expr="isl + step",
        predicate=(
            f"num_heads = {model.heads_per_gpu} "
            f"AND num_key_value_heads = {model.kv_heads_per_gpu} "
            f"AND head_dim = {spec.head_dim} "
            f"AND window_size = 0 AND beam_width = 1 "
            f"AND kv_cache_dtype = '{spec.kv_dtype}'"
        ),
        sqrt_seq_axis=False,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class AllReduceCurve:
    """Measured custom-allreduce latency vs element count for one fan-out."""

    message_size: np.ndarray
    latency: np.ndarray
    num_gpus: int

    def query(self, elements: float) -> float:
        lo = float(self.message_size[0])
        hi = float(self.message_size[-1])
        if elements < lo or elements > hi:
            raise CoverageError(
                f"custom_allreduce: message size {elements:g} outside the "
                f"collected range [{lo:g}, {hi:g}] at num_gpus={self.num_gpus}",
                table="custom_allreduce_perf",
                query={"num_gpus": self.num_gpus, "message_size": elements},
            )
        return float(np.interp(elements, self.message_size, self.latency))


def load_allreduce_curve(
    con: duckdb.DuckDBPyConnection, path: Path, tp: int
) -> AllReduceCurve:
    # AISimulate's loader drops every `*_eager` backend row outside b60, so the
    # graph-captured curve is the one that backs vLLM predictions.
    rows = _fetch(
        con,
        f"SELECT message_size, min(latency) AS latency FROM {_relation(path)} "
        f"WHERE num_gpus = {tp} AND NOT (backend LIKE '%\\_eager' ESCAPE '\\') "
        f"GROUP BY message_size ORDER BY message_size",
    )
    if rows["message_size"].size == 0:
        raise CoverageError(
            f"custom_allreduce_perf has no graph-mode rows at num_gpus={tp}",
            table="custom_allreduce_perf",
            query={"num_gpus": tp},
        )
    return AllReduceCurve(
        message_size=rows["message_size"].astype(np.float64),
        latency=rows["latency"].astype(np.float64),
        num_gpus=tp,
    )


# ---------------------------------------------------------------------------
# Derived calibration scalars
# ---------------------------------------------------------------------------


def compute_efficiency(
    con: duckdb.DuckDBPyConnection, path: Path, dtype: str, gpu: GpuSpec
) -> float:
    """90th-percentile measured MFU (2mnk / latency / datasheet peak)."""
    rows = _fetch(
        con,
        f"SELECT m, n, k, latency FROM {_relation(path)} WHERE gemm_dtype = '{dtype}'",
    )
    if rows["m"].size == 0:
        raise TableLoadError(
            f"no gemm_perf rows for gemm_dtype='{dtype}'", source=str(path)
        )
    flops = (
        2.0
        * rows["m"].astype(np.float64)
        * rows["n"].astype(np.float64)
        * rows["k"].astype(np.float64)
    )
    seconds = rows["latency"].astype(np.float64) / 1000.0
    mfu = flops / np.maximum(seconds, 1e-12) / gpu.flops(dtype)
    return float(min(max(np.percentile(mfu, 90.0), 0.01), 1.0))
