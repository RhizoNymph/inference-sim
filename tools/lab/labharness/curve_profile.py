"""Calibration profile carrying a compute-efficiency curve.

The output is the base profile (normally the static-batch scalar fit) with the
fitted `compute_efficiency_curve` and, optionally, the frontend latency keys
inserted into its `[calibration]` table. Everything else (provenance, valid
shape, scalar constants, benchmarks) is kept byte for byte; a comment block
at the top records where the added keys came from.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass

from labharness.efficiency_curve import EfficiencyCurveFit
from labharness.errors import FitError
from labharness.frontend import FrontendFit
from labharness.toml_emit import fmt_float

_ADDED_KEYS = (
    "compute_efficiency_curve",
    "frontend_latency_us",
    "frontend_latency_per_prompt_token_us",
)


@dataclass(frozen=True, slots=True)
class CurveProfileInputs:
    base_text: str
    base_label: str
    curve: EfficiencyCurveFit
    curve_source: str
    frontend: FrontendFit | None
    frontend_source: str | None


def render_curve_profile(inputs: CurveProfileInputs) -> str:
    try:
        base = tomllib.loads(inputs.base_text)
    except tomllib.TOMLDecodeError as error:
        raise FitError(
            f"base profile is not valid TOML: {error}", context={"base": inputs.base_label}
        ) from error
    calibration = base.get("calibration")
    if not isinstance(calibration, dict):
        raise FitError("base profile has no [calibration] table", context={"base": inputs.base_label})
    present = [key for key in _ADDED_KEYS if key in calibration]
    if present:
        raise FitError("base profile already sets the keys to add", context={"keys": ",".join(present)})

    added = [f"compute_efficiency_curve = {inputs.curve.toml_value()}"]
    if inputs.frontend is not None:
        added += [
            f"frontend_latency_us = {fmt_float(inputs.frontend.fixed_us)}",
            f"frontend_latency_per_prompt_token_us = {fmt_float(inputs.frontend.per_prompt_token_us)}",
        ]
    lines = inputs.base_text.splitlines()
    try:
        header = lines.index("[calibration]")
    except ValueError as error:
        raise FitError("base profile's [calibration] header is not on its own line") from error
    lines[header + 1 : header + 1] = added

    skipped = ", ".join(f"{s.shape.label}" for s in inputs.curve.skipped) or "none"
    comment = [
        f"# Base profile: {inputs.base_label}",
        f"# compute_efficiency_curve fitted by `lab.py fit-curve` from {inputs.curve_source}",
        f"# (memory-bound margin {inputs.curve.memory_bound_margin:g}; "
        f"weight-read-bound shapes skipped: {skipped}).",
        "# The curve takes precedence over the scalar compute_efficiency, which is kept for reference.",
    ]
    if inputs.frontend is not None:
        comment.append(
            f"# frontend_latency_* fitted from {inputs.frontend_source} minus the static batch-1 prefill."
        )
    return "\n".join(comment + lines) + "\n"
