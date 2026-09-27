"""Structured exception hierarchy for the AISimulate calibration converter."""

from __future__ import annotations

from collections.abc import Mapping

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConverterError(Exception):
    """Base class for every error this tool raises deliberately."""

    exit_code: int = 1

    def details(self) -> dict[str, object]:
        return {}


class ConfigurationError(ConverterError):
    """CLI arguments or fixture metadata are inconsistent."""

    exit_code = 2


class TableLoadError(ConverterError):
    """A required measured table is missing or unreadable."""

    exit_code = 3

    def __init__(self, message: str, *, source: str) -> None:
        super().__init__(message)
        self.source = source

    def details(self) -> dict[str, object]:
        return {"source": self.source}


class CoverageError(ConverterError):
    """A required shape is outside the measured envelope."""

    exit_code = 4

    def __init__(
        self, message: str, *, table: str, query: Mapping[str, object]
    ) -> None:
        super().__init__(message)
        self.table = table
        self.query = dict(query)

    def details(self) -> dict[str, object]:
        return {"table": self.table, **{f"query_{k}": v for k, v in self.query.items()}}


class FitError(ConverterError):
    """The regression could not be built from the composed samples."""

    exit_code = 5


class SelfTestError(ConverterError):
    """The self-test output does not match the golden profile."""

    exit_code = 6
