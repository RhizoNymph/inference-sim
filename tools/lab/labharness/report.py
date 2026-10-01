"""Markdown comparison tables and JSON serialisation of evaluations.

Error convention everywhere: signed percent `(sim - measured) / measured`;
positive means the simulator predicts slower than measured. Summary rows
report mean |error|.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from labharness.evaluate import ProfileCheck, ServingEvaluation, StaticEvaluation
from labharness.fitting import ScalarFit, pct_error
from labharness.results import LatencyStats
from labharness.spec import rate_text


def _ms(value: float) -> str:
    return f"{value:,.1f}"


def _pct(value: float) -> str:
    return f"{value:+.1f}%"


# ---------------------------------------------------------------------------
# Static batch
# ---------------------------------------------------------------------------


def static_table(evaluation: StaticEvaluation) -> list[str]:
    lines = [
        f"### {evaluation.label}",
        "",
        f"Calibration: `{evaluation.calibration}`",
        "",
        "| shape | prefill real ms | prefill sim ms | err | decode step real ms "
        "| decode step sim ms | err | e2e real ms | e2e sim ms | err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (m, s), e in zip(evaluation.pairs, evaluation.errors, strict=True):
        lines.append(
            f"| {m.shape.label} | {_ms(m.prefill_ms)} | {_ms(s.prefill_ms)} | {_pct(e.prefill_pct)} "
            f"| {m.decode_ms_per_step:.2f} | {s.decode_ms_per_step:.2f} | {_pct(e.decode_step_pct)} "
            f"| {_ms(m.end_to_end_ms)} | {_ms(s.end_to_end_ms)} | {_pct(e.end_to_end_pct)} |"
        )
    summary = evaluation.summary
    lines.append(
        f"| **mean \\|err\\|** | | | **{summary.prefill_mean_abs_pct:.1f}%** | | | "
        f"**{summary.decode_step_mean_abs_pct:.1f}%** | | | **{summary.end_to_end_mean_abs_pct:.1f}%** |"
    )
    return lines


def fit_lines(fit: ScalarFit, folds: Sequence[tuple[str, ScalarFit]]) -> list[str]:
    lines = [
        f"- `compute_efficiency` = {fit.base_compute_efficiency:g} x median(sim/real prefill) = "
        f"**{fit.compute_efficiency:.4f}** (ratios: "
        + ", ".join(f"{s.label} {r:.3f}" for s, r in zip(fit.shapes, fit.prefill_ratios, strict=True))
        + ")",
        f"- `decode_memory_bandwidth_scale` = {fit.base_decode_memory_bandwidth_scale:g} x "
        f"median(sim/real decode step) = **{fit.decode_memory_bandwidth_scale:.4f}** (ratios: "
        + ", ".join(f"{s.label} {r:.3f}" for s, r in zip(fit.shapes, fit.decode_step_ratios, strict=True))
        + ")",
        "- leave-one-shape-out folds (held-out shape: compute_efficiency / decode_memory_bandwidth_scale): "
        + "; ".join(
            f"{label}: {f.compute_efficiency:.4f} / {f.decode_memory_bandwidth_scale:.4f}"
            for label, f in folds
        ),
    ]
    return lines


def profile_check_lines(check: ProfileCheck, path: str) -> list[str]:
    return [
        f"- `{path}` loads in the simulator as profile `{check.profile_name}` "
        f"(applicability `{check.applicability_status}`), reporting compute_efficiency "
        f"{check.compute_efficiency:g} and decode_memory_bandwidth_scale "
        f"{check.decode_memory_bandwidth_scale:g}; predictions through the profile drift at most "
        f"{check.max_prediction_drift_pct:.4f}% from the in-sample scalar run.",
    ]


def static_json(evaluation: StaticEvaluation) -> dict[str, object]:
    return {
        "label": evaluation.label,
        "calibration": evaluation.calibration,
        "summary": {
            "prefill_mean_abs_pct": evaluation.summary.prefill_mean_abs_pct,
            "decode_step_mean_abs_pct": evaluation.summary.decode_step_mean_abs_pct,
            "end_to_end_mean_abs_pct": evaluation.summary.end_to_end_mean_abs_pct,
            "end_to_end_max_abs_pct": evaluation.summary.end_to_end_max_abs_pct,
            "count": evaluation.summary.count,
        },
        "rows": [
            {
                "shape": m.shape.label,
                "measured": {
                    "prefill_ms": m.prefill_ms,
                    "decode_ms_per_step": m.decode_ms_per_step,
                    "end_to_end_ms": m.end_to_end_ms,
                },
                "simulated": s.to_json(),
                "error_pct": {
                    "prefill": e.prefill_pct,
                    "decode_step": e.decode_step_pct,
                    "end_to_end": e.end_to_end_pct,
                },
            }
            for (m, s), e in zip(evaluation.pairs, evaluation.errors, strict=True)
        ],
    }


def fit_json(fit: ScalarFit) -> dict[str, object]:
    return {
        "compute_efficiency": fit.compute_efficiency,
        "decode_memory_bandwidth_scale": fit.decode_memory_bandwidth_scale,
        "base_compute_efficiency": fit.base_compute_efficiency,
        "base_decode_memory_bandwidth_scale": fit.base_decode_memory_bandwidth_scale,
        "shapes": [s.label for s in fit.shapes],
        "prefill_ratios": list(fit.prefill_ratios),
        "decode_step_ratios": list(fit.decode_step_ratios),
    }


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

_SERVING_METRICS = ("ttft", "tpot", "itl", "e2el")


def _stat_cells(
    real: LatencyStats | None, sim: LatencyStats | None, which: str
) -> tuple[str, str, float | None]:
    if real is None:
        return "-", "-", None
    real_value = getattr(real, which)
    if sim is None:
        return _ms(real_value), "n/a", None
    sim_value = getattr(sim, which)
    return _ms(real_value), _ms(sim_value), pct_error(sim_value, real_value)


def serving_table(evaluation: ServingEvaluation) -> list[str]:
    lines = [
        f"### {evaluation.label}",
        "",
        f"Calibration: `{evaluation.calibration}`. Ref batch = the serving model's "
        "`[request].batch_size`, which the iteration engine does not use (see the feature doc). "
        "`n/a` = the simulator rejected the candidate and reported no latency metrics.",
        "",
        "| rate req/s | ref batch (sim status) | metric | real median | sim median | err "
        "| real p99 | sim p99 | err |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    errors: dict[str, list[float]] = {}
    for pair in evaluation.pairs:
        rate = rate_text(pair.measured.request_rate)
        ref = f"{pair.simulated.reference_batch} ({pair.simulated.status})"
        for metric in _SERVING_METRICS:
            real = getattr(pair.measured, metric)
            sim = getattr(pair.simulated, metric)
            rm, sm, em = _stat_cells(real, sim, "median_ms")
            rp, sp, ep = _stat_cells(real, sim, "p99_ms")
            for key, value in ((f"{metric} median", em), (f"{metric} p99", ep)):
                if value is not None:
                    errors.setdefault(key, []).append(abs(value))
            median_err = _pct(em) if em is not None else "-"
            p99_err = _pct(ep) if ep is not None else "-"
            cells = [rate, ref, metric.upper(), rm, sm, median_err, rp, sp, p99_err]
            lines.append("| " + " | ".join(cells) + " |")
        sim_tput = pair.simulated.output_throughput
        err = pct_error(sim_tput, pair.measured.output_throughput) if sim_tput else None
        if err is not None:
            errors.setdefault("output tok/s", []).append(abs(err))
        lines.append(
            f"| {rate} | {ref} | output tok/s | "
            f"{pair.measured.output_throughput:,.1f} | "
            f"{f'{sim_tput:,.1f}' if sim_tput else 'n/a'} | {_pct(err) if err is not None else '-'} | | | |"
        )
    rejected = [p for p in evaluation.pairs if p.simulated.rejected_reason]
    if rejected:
        lines += ["", "Simulator rejections:", ""]
        lines += [
            f"- rate {rate_text(p.measured.request_rate)}: {p.simulated.rejected_reason}" for p in rejected
        ]
    lines += ["", "Mean |error| per metric across rates:", ""]
    lines += ["| metric | mean \\|err\\| |", "|---|---:|"]
    for key, values in errors.items():
        lines.append(f"| {key} | {statistics.fmean(values):.1f}% |")
    return lines


def serving_summary(evaluation: ServingEvaluation) -> dict[str, float]:
    errors: dict[str, list[float]] = {}
    for pair in evaluation.pairs:
        for metric in _SERVING_METRICS:
            real, sim = getattr(pair.measured, metric), getattr(pair.simulated, metric)
            if real is None or sim is None:
                continue
            for which in ("median_ms", "p99_ms"):
                errors.setdefault(f"{metric}_{which[:-3]}", []).append(
                    abs(pct_error(getattr(sim, which), getattr(real, which)))
                )
        if pair.simulated.output_throughput:
            errors.setdefault("output_throughput", []).append(
                abs(pct_error(pair.simulated.output_throughput, pair.measured.output_throughput))
            )
    return {key: statistics.fmean(values) for key, values in errors.items()}


def serving_json(evaluation: ServingEvaluation) -> dict[str, object]:
    def stats(value: LatencyStats | None) -> dict[str, float] | None:
        if value is None:
            return None
        return {"mean_ms": value.mean_ms, "median_ms": value.median_ms, "p99_ms": value.p99_ms}

    return {
        "label": evaluation.label,
        "calibration": evaluation.calibration,
        "mean_abs_pct": serving_summary(evaluation),
        "rates": [
            {
                "request_rate": rate_text(pair.measured.request_rate),
                "measured": {
                    **{metric: stats(getattr(pair.measured, metric)) for metric in _SERVING_METRICS},
                    "output_throughput": pair.measured.output_throughput,
                    "request_throughput": pair.measured.request_throughput,
                    "max_concurrent_requests": pair.measured.max_concurrent_requests,
                },
                "simulated": pair.simulated.to_json(),
            }
            for pair in evaluation.pairs
        ],
    }
