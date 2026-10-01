"""Byte-stable TOML value rendering.

Mirrors tools/aisimulate_calibration/converter/emit.py so profiles written by
both tools format numbers identically.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from labharness.errors import FitError

type TomlScalar = bool | int | float | str
type TomlValue = TomlScalar | Sequence[int] | Sequence[float] | Sequence[str] | None


def fmt_float(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        raise FitError(f"refusing to emit a non-finite number: {value}")
    text = f"{value:.{digits}g}"
    if "e" in text or "E" in text:
        mantissa, _, exponent = text.partition("e")
        if "." not in mantissa:
            mantissa = f"{mantissa}.0"
        return f"{mantissa}e{exponent}"
    if "." not in text:
        text = f"{text}.0"
    return text


def toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_value(key: str, value: TomlValue) -> str:
    match value:
        case bool():
            return "true" if value else "false"
        case int():
            return str(value)
        case float():
            return fmt_float(value)
        case str():
            return toml_string(value)
        case list() | tuple():
            return "[" + ", ".join(render_value(key, item) for item in value) + "]"
        case _:
            raise FitError(f"cannot emit TOML for {key}: {type(value)!r}")


def kv_lines(pairs: Iterable[tuple[str, TomlValue]]) -> list[str]:
    """Render `key = value` lines, skipping keys whose value is None."""
    return [f"{key} = {render_value(key, value)}" for key, value in pairs if value is not None]


def table(name: str, pairs: Iterable[tuple[str, TomlValue]], *, array: bool = False) -> list[str]:
    header = f"[[{name}]]" if array else f"[{name}]"
    return [header, *kv_lines(pairs)]
