"""End-to-end run plan: load tables, compose samples, fit, render."""

from __future__ import annotations

import dataclasses
import logging

import duckdb

from converter.emit import render_profile
from converter.fitting import fit_phase
from converter.opwalk import RankTables, compose_samples, shape_bounds
from converter.sources import TableSources
from converter.spec import (
    DECODE_FEATURES,
    PREFILL_FEATURES,
    GpuSpec,
    GridSpec,
    ModelSpec,
    Provenance,
)
from converter.tables import (
    compute_efficiency,
    load_allreduce_curve,
    load_context_attention,
    load_gemm_table,
    load_generation_attention,
)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class RunPlan:
    sources: TableSources
    gpu: GpuSpec
    model: ModelSpec
    grid: GridSpec
    provenance: Provenance
    holdout_fraction: float
    seed: int
    benchmark_limit: int


def build_rank_tables(plan: RunPlan, logger: logging.Logger) -> dict[int, RankTables]:
    con = duckdb.connect(database=":memory:")
    tables: dict[int, RankTables] = {}
    gemm = load_gemm_table(con, plan.sources.gemm, plan.model.dtype, plan.gpu)
    for tp in plan.grid.tensor_ranks:
        sharded = plan.model.sharded(tp)
        tables[tp] = RankTables(
            model=sharded,
            gemm=gemm,
            context_attention=load_context_attention(
                con, plan.sources.context_attention, sharded
            ),
            generation_attention=load_generation_attention(
                con, plan.sources.generation_attention, sharded
            ),
            allreduce=load_allreduce_curve(con, plan.sources.allreduce, tp),
            gpu=plan.gpu,
        )
        logger.info(
            "loaded measured tables for tensor rank",
            extra={
                "tp": tp,
                "gemm_sites": len(gemm.index),
                "heads_per_gpu": sharded.heads_per_gpu,
                "kv_heads_per_gpu": sharded.kv_heads_per_gpu,
                "context_lane": tables[tp].context_attention.lane,
                "generation_lane": tables[tp].generation_attention.lane,
            },
        )
    con.close()
    return tables


def run(plan: RunPlan, logger: logging.Logger) -> str:
    tables = build_rank_tables(plan, logger)
    prefill, decode = compose_samples(tables, plan.grid, logger)
    prefill_fit = fit_phase(
        name=f"{plan.provenance.system}-{plan.provenance.backend}-prefill-latency-fit",
        target="prefill_ms",
        phase="prefill",
        feature_names=PREFILL_FEATURES,
        samples=prefill,
        holdout_fraction=plan.holdout_fraction,
        seed=plan.seed,
    )
    decode_fit = fit_phase(
        name=f"{plan.provenance.system}-{plan.provenance.backend}-decode-latency-fit",
        target="decode_ms",
        phase="decode",
        feature_names=DECODE_FEATURES,
        samples=decode,
        holdout_fraction=plan.holdout_fraction,
        seed=plan.seed + 1,
    )
    for fit in (prefill_fit, decode_fit):
        logger.info(
            "fitted phase",
            extra={
                "phase": fit.phase,
                "r_squared": round(fit.train.r_squared, 6),
                "rmse_pct": round(fit.train.rmse_pct, 4),
                "validation_rmse_pct": round(fit.holdout.rmse_pct, 4),
                "sample_count": fit.train.sample_count,
                "validation_sample_count": fit.holdout.sample_count,
            },
        )

    con = duckdb.connect(database=":memory:")
    efficiency = compute_efficiency(con, plan.sources.gemm, plan.model.dtype, plan.gpu)
    con.close()
    logger.info(
        "derived compute efficiency", extra={"compute_efficiency": round(efficiency, 6)}
    )

    return render_profile(
        model=plan.model,
        gpu=plan.gpu,
        provenance=plan.provenance,
        efficiency=efficiency,
        shape=shape_bounds(decode),
        fits=(prefill_fit, decode_fit),
        benchmark_limit=plan.benchmark_limit,
    )
