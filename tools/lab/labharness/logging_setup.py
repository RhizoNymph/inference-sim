"""Structured `level msg key=value ...` logging (mirrors aisimulate_calibration)."""

from __future__ import annotations

import logging
import sys
from typing import Final

LOGGER_NAME: Final = "lab"


class KeyValueFormatter(logging.Formatter):
    """Render records as `level message key=value ...`."""

    _RESERVED: Final = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname.lower():<5} {record.getMessage()}"
        extras = {k: v for k, v in record.__dict__.items() if k not in self._RESERVED}
        if not extras:
            return base
        rendered = " ".join(f"{k}={_render_value(v)}" for k, v in sorted(extras.items()))
        return f"{base} {rendered}"


def _render_value(value: object) -> str:
    text = str(value)
    return f'"{text}"' if " " in text else text


def configure_logging(verbose: bool) -> logging.Logger:
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(KeyValueFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
