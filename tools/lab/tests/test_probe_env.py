"""GPU-state parsing in remote/probe_env.py (importable without torch or a GPU)."""

from __future__ import annotations

import importlib.util
from types import ModuleType

from tests.conftest import TOOL_DIR


def _load() -> ModuleType:
    path = TOOL_DIR / "remote" / "probe_env.py"
    spec = importlib.util.spec_from_file_location("probe_env", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load()


def test_gpu_state_parses_one_line_per_gpu() -> None:
    out = "0, 420.00, 420.00, 2100, 1905, 9751, 0x0000000000000000\n1, 350.00, 420.00, 2100, 210, 405, 0x0000000000000001\n"
    states = probe.parse_gpu_state(out)
    assert states == [
        {
            "index": 0,
            "power_limit_w": 420.0,
            "default_power_limit_w": 420.0,
            "max_sm_clock_mhz": 2100,
            "sm_clock_mhz": 1905,
            "mem_clock_mhz": 9751,
            "clock_event_reasons": "0x0000000000000000",
        },
        {
            "index": 1,
            "power_limit_w": 350.0,
            "default_power_limit_w": 420.0,
            "max_sm_clock_mhz": 2100,
            "sm_clock_mhz": 210,
            "mem_clock_mhz": 405,
            "clock_event_reasons": "0x0000000000000001",
        },
    ]


def test_gpu_state_maps_unavailable_values_to_none() -> None:
    states = probe.parse_gpu_state("0, [N/A], 420.00, [Not Supported], 1905, 9751, [N/A]\n")
    assert states[0]["power_limit_w"] is None
    assert states[0]["max_sm_clock_mhz"] is None
    assert states[0]["clock_event_reasons"] is None
    assert states[0]["default_power_limit_w"] == 420.0


def test_gpu_state_skips_malformed_lines() -> None:
    assert probe.parse_gpu_state("garbage\n\n") == []


def test_compute_apps_parse_pid_name_and_memory() -> None:
    out = "6634, text-embeddings-router, 594\n3171273, /app/.venv/bin/python, 1970\n"
    assert probe.parse_compute_apps(out) == [
        {"pid": 6634, "process_name": "text-embeddings-router", "used_memory_mib": 594},
        {"pid": 3171273, "process_name": "/app/.venv/bin/python", "used_memory_mib": 1970},
    ]


def test_compute_apps_empty_output_is_empty() -> None:
    assert probe.parse_compute_apps("") == []
