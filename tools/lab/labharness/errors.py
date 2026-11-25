"""Structured exception hierarchy for the lab harness.

Every error the harness raises deliberately derives from `LabError`, carries a
process exit code, and exposes key/value `details()` for structured logging.
"""

from __future__ import annotations

from collections.abc import Mapping


class LabError(Exception):
    """Base class for every error this tool raises deliberately."""

    exit_code: int = 1

    def details(self) -> dict[str, object]:
        return {}


class SpecError(LabError):
    """An experiment or lab spec is missing a field or is inconsistent."""

    exit_code = 2

    def __init__(self, message: str, *, path: str, field: str | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.field = field

    def details(self) -> dict[str, object]:
        return {"spec": self.path, **({"field": self.field} if self.field else {})}


class ResultParseError(LabError):
    """A measured or simulated result file could not be parsed."""

    exit_code = 3

    def __init__(self, message: str, *, source: str) -> None:
        super().__init__(message)
        self.source = source

    def details(self) -> dict[str, object]:
        return {"source": self.source}


class RemoteCommandError(LabError):
    """A remote or local step of an experiment failed."""

    exit_code = 4

    def __init__(self, message: str, *, step: str, returncode: int | None) -> None:
        super().__init__(message)
        self.step = step
        self.returncode = returncode

    def details(self) -> dict[str, object]:
        return {"step": self.step, "returncode": self.returncode}


class CompletionTimeoutError(LabError):
    """A benchmark did not produce its expected results before the deadline."""

    exit_code = 5

    def __init__(self, message: str, *, step: str, observed: int, expected: int) -> None:
        super().__init__(message)
        self.step = step
        self.observed = observed
        self.expected = expected

    def details(self) -> dict[str, object]:
        return {"step": self.step, "observed": self.observed, "expected": self.expected}


class SimulatorError(LabError):
    """The inference-sim binary failed or returned unusable JSON."""

    exit_code = 6

    def __init__(self, message: str, *, workload: str, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.workload = workload
        self.stderr_tail = stderr_tail

    def details(self) -> dict[str, object]:
        return {"workload": self.workload, "stderr_tail": self.stderr_tail}


class FitError(LabError):
    """The calibration fit could not be computed from the matched data."""

    exit_code = 7

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.context = dict(context or {})

    def details(self) -> dict[str, object]:
        return dict(self.context)


class RunDirError(LabError):
    """A lab-runs directory is missing, already exists, or is incomplete."""

    exit_code = 8

    def __init__(self, message: str, *, path: str) -> None:
        super().__init__(message)
        self.path = path

    def details(self) -> dict[str, object]:
        return {"path": self.path}


class CurveError(LabError):
    """Collective benchmark rows cannot be turned into simulator curves."""

    exit_code = 9

    def __init__(self, message: str, *, op: str | None = None, source: str | None = None) -> None:
        super().__init__(message)
        self.op = op
        self.source = source

    def details(self) -> dict[str, object]:
        details: dict[str, object] = {}
        if self.op is not None:
            details["op"] = self.op
        if self.source is not None:
            details["source"] = self.source
        return details
