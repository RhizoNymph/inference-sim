"""Non-negative least squares fitting and honest fit statistics."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

from converter.errors import ConfigurationError, FitError
from converter.opwalk import Sample

# ---------------------------------------------------------------------------
# Non-negative least squares + statistics
# ---------------------------------------------------------------------------


def nnls(design: np.ndarray, target: np.ndarray, *, max_iter: int = 200) -> np.ndarray:
    """Lawson-Hanson active-set NNLS.  Small problems only (<= 4 columns here)."""
    n_cols = design.shape[1]
    passive = np.zeros(n_cols, dtype=bool)
    solution = np.zeros(n_cols, dtype=np.float64)
    residual = target - design @ solution
    gradient = design.T @ residual
    for _ in range(max_iter):
        candidates = np.flatnonzero(~passive & (gradient > 1e-12))
        if candidates.size == 0:
            break
        enter = int(candidates[np.argmax(gradient[candidates])])
        passive[enter] = True
        for _ in range(max_iter):
            columns = np.flatnonzero(passive)
            trial = np.zeros(n_cols, dtype=np.float64)
            trial[columns] = np.linalg.lstsq(design[:, columns], target, rcond=None)[0]
            if np.all(trial[columns] > 0.0):
                solution = trial
                break
            blocked = columns[trial[columns] <= 0.0]
            ratios = solution[blocked] / (solution[blocked] - trial[blocked])
            alpha = float(np.min(ratios)) if ratios.size else 0.0
            solution = solution + alpha * (trial - solution)
            zeroed = np.flatnonzero(passive & (solution <= 1e-15))
            if zeroed.size == 0:
                solution = trial
                break
            passive[zeroed] = False
            solution[zeroed] = 0.0
        else:  # pragma: no cover - guarded by max_iter
            raise FitError("NNLS inner loop failed to converge")
        residual = target - design @ solution
        gradient = design.T @ residual
    return solution


@dataclasses.dataclass(frozen=True, slots=True)
class FitStats:
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    rmse_pct: float
    mean_abs_pct_error: float
    max_abs_pct_error: float
    sample_count: int


def evaluate(actual: np.ndarray, predicted: np.ndarray, n_features: int) -> FitStats:
    residual = predicted - actual
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    n = actual.size
    denominator = n - n_features - 1
    adjusted = (
        1.0 - (1.0 - r_squared) * (n - 1) / denominator
        if denominator > 0
        else r_squared
    )
    rmse = math.sqrt(ss_res / n)
    mean_actual = float(actual.mean())
    pct = np.abs(residual) / np.maximum(np.abs(actual), 1e-12) * 100.0
    return FitStats(
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        rmse=rmse,
        rmse_pct=rmse / mean_actual * 100.0 if mean_actual > 0.0 else 0.0,
        mean_abs_pct_error=float(pct.mean()),
        max_abs_pct_error=float(pct.max()),
        sample_count=n,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class PhaseFit:
    name: str
    target: str
    phase: str
    features: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    feature_ranges: tuple[tuple[str, float, float], ...]
    train: FitStats
    holdout: FitStats
    holdout_samples: tuple[Sample, ...]
    holdout_predictions: tuple[float, ...]


def split_samples(
    samples: Sequence[Sample], holdout_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < holdout_fraction < 0.9:
        raise ConfigurationError(
            f"--holdout-fraction must be in (0, 0.9), got {holdout_fraction}"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    n_holdout = max(1, int(round(len(samples) * holdout_fraction)))
    holdout_idx = sorted(int(i) for i in order[:n_holdout])
    train_idx = sorted(int(i) for i in order[n_holdout:])
    if not train_idx:
        raise FitError("holdout fraction consumed every sample")
    return [samples[i] for i in train_idx], [samples[i] for i in holdout_idx]


def fit_phase(
    *,
    name: str,
    target: str,
    phase: str,
    feature_names: Sequence[str],
    samples: Sequence[Sample],
    holdout_fraction: float,
    seed: int,
) -> PhaseFit:
    train, holdout = split_samples(samples, holdout_fraction, seed)
    n_features = len(feature_names)
    if len(train) <= n_features:
        raise FitError(
            f"{name}: {len(train)} training samples for {n_features} features"
        )

    def design_of(rows: Sequence[Sample]) -> np.ndarray:
        block = np.array([row.features for row in rows], dtype=np.float64)
        return np.hstack([np.ones((len(rows), 1)), block])

    train_design = design_of(train)
    train_target = np.array([row.target_ms for row in train], dtype=np.float64)
    solution = nnls(train_design, train_target)
    intercept = float(solution[0])
    coefficients = tuple(float(v) for v in solution[1:])

    train_stats = evaluate(train_target, train_design @ solution, n_features)
    holdout_design = design_of(holdout)
    holdout_target = np.array([row.target_ms for row in holdout], dtype=np.float64)
    holdout_predicted = holdout_design @ solution
    holdout_stats = evaluate(holdout_target, holdout_predicted, n_features)

    block = np.array([row.features for row in train], dtype=np.float64)
    ranges = tuple(
        (feature_names[i], float(block[:, i].min()), float(block[:, i].max()))
        for i in range(n_features)
    )
    return PhaseFit(
        name=name,
        target=target,
        phase=phase,
        features=tuple(feature_names),
        intercept=intercept,
        coefficients=coefficients,
        feature_ranges=ranges,
        train=train_stats,
        holdout=holdout_stats,
        holdout_samples=tuple(holdout),
        holdout_predictions=tuple(float(v) for v in holdout_predicted),
    )
