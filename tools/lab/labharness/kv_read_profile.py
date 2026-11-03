"""Calibration profile carrying a fitted `kv_read_bandwidth_scale`.

The output is the base profile (normally the curve + frontend profile) with
`kv_read_bandwidth_scale` inserted into its `[calibration]` table and, for a
joint fit, its `decode_memory_bandwidth_scale` value replaced. Everything else
is kept byte for byte; a comment block at the top records the fit's sources,
mode, and dataset-holdout errors.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass

from labharness.errors import FitError
from labharness.kv_read_fit import KvFitMode, KvReadFit
from labharness.toml_emit import fmt_float

_WEIGHT_LINE = re.compile(r"^decode_memory_bandwidth_scale\s*=")


@dataclass(frozen=True, slots=True)
class HoldoutSummary:
    """A fit trained without `held_out`, evaluated on it by re-simulation."""

    held_out: str
    kv_read_bandwidth_scale: float
    decode_step_mean_abs_pct: float

    def to_json(self) -> dict[str, object]:
        return {
            "held_out": self.held_out,
            "kv_read_bandwidth_scale": round(self.kv_read_bandwidth_scale, 6),
            "decode_step_mean_abs_pct": round(self.decode_step_mean_abs_pct, 4),
        }


@dataclass(frozen=True, slots=True)
class KvProfileInputs:
    base_text: str
    base_label: str
    fit: KvReadFit
    sources: Sequence[str]
    holdout: Sequence[HoldoutSummary]


def render_kv_read_profile(inputs: KvProfileInputs) -> str:
    try:
        base = tomllib.loads(inputs.base_text)
    except tomllib.TOMLDecodeError as error:
        raise FitError(
            f"base profile is not valid TOML: {error}", context={"base": inputs.base_label}
        ) from error
    calibration = base.get("calibration")
    if not isinstance(calibration, dict):
        raise FitError("base profile has no [calibration] table", context={"base": inputs.base_label})
    if "kv_read_bandwidth_scale" in calibration:
        raise FitError(
            "base profile already sets kv_read_bandwidth_scale", context={"base": inputs.base_label}
        )

    lines = inputs.base_text.splitlines()
    try:
        header = lines.index("[calibration]")
    except ValueError as error:
        raise FitError("base profile's [calibration] header is not on its own line") from error
    fit = inputs.fit
    if fit.mode is KvFitMode.JOINT:
        end = next((i for i in range(header + 1, len(lines)) if lines[i].startswith("[")), len(lines))
        targets = [i for i in range(header + 1, end) if _WEIGHT_LINE.match(lines[i])]
        if len(targets) != 1:
            raise FitError(
                "joint fit needs exactly one decode_memory_bandwidth_scale line in [calibration] to replace",
                context={"base": inputs.base_label, "found": len(targets)},
            )
        lines[targets[0]] = f"decode_memory_bandwidth_scale = {fmt_float(fit.decode_memory_bandwidth_scale)}"
    lines[header + 1 : header + 1] = [f"kv_read_bandwidth_scale = {fmt_float(fit.kv_read_bandwidth_scale)}"]

    was = fmt_float(fit.base_decode_memory_bandwidth_scale)
    weight_note = (
        f"decode_memory_bandwidth_scale refitted jointly (was {was})."
        if fit.mode is KvFitMode.JOINT
        else "decode_memory_bandwidth_scale kept from the base profile."
    )
    comment = [
        f"# kv_read_bandwidth_scale fitted by `lab.py fit-kv-read` (mode {fit.mode.value}) from "
        f"{fit.sample_count} decode steps: {', '.join(inputs.sources)}.",
        f"# {weight_note}",
        *(
            f"# Dataset holdout: held out {h.held_out}, kv_read_bandwidth_scale "
            f"{fmt_float(h.kv_read_bandwidth_scale, 4)}, "
            f"decode step mean |error| {h.decode_step_mean_abs_pct:.1f}%."
            for h in inputs.holdout
        ),
        f"# Base profile: {inputs.base_label}",
    ]
    return "\n".join(comment + lines) + "\n"
