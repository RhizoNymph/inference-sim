"""Runner control flow against a fake executor (no ssh, no GPUs)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from labharness.commands import build_plan
from labharness.errors import CompletionTimeoutError, RemoteCommandError, RunDirError
from labharness.runner import ProbeStatus, Runner, parse_probe
from labharness.spec import parse_experiment
from tests.conftest import SPECS

PROBE = {"record": "lab.probe.v1", "vllm_version": "0.29.0", "torch_version": "2.13.0", "gpus": []}


class FakeExecutor:
    """Answers each step by keyword; records every call."""

    def __init__(self, *, vllm: str = "0.29.0", polls: list[str] | None = None) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.vllm = vllm
        self.polls = polls or ["complete=0 progress=2 alive=1", "complete=1 progress=5 alive=1"]

    def __call__(
        self, argv: list[str], stdin: str | None, timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, stdin))
        text = stdin or " ".join(argv)
        out = ""
        if "probe_env.py" in text:
            out = "INFO noise\n" + json.dumps({**PROBE, "vllm_version": self.vllm}) + "\n"
        elif "complete=$c" in text:
            out = self.polls.pop(0) + "\n"
        elif argv[:1] == ["git"]:
            out = "deadbeef\n" if "rev-parse" in argv else ""
        return subprocess.CompletedProcess(argv, 0, out, "")


def _runner(tmp_path: Path, executor: FakeExecutor) -> Runner:
    exp = parse_experiment(SPECS / "rtx3090_qwen7b_static_pp1.toml")
    plan = build_plan(exp, date="2026-10-01", runs_root=tmp_path, remote_src=Path("/src"))
    return Runner(exp, plan, repo=tmp_path, executor=executor, sleep=lambda _: None)


def test_successful_run_writes_manifest(tmp_path: Path) -> None:
    executor = FakeExecutor()
    _runner(tmp_path, executor).run()
    run_dir = tmp_path / "2026-10-01-qwen7b-static-pp1"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["simulator_git_sha"] == "deadbeef"
    assert manifest["simulator_git_dirty"] is False
    assert manifest["probes"]["node0"]["vllm_version"] == "0.29.0"
    assert manifest["commands"][0].startswith("ssh -o BatchMode=yes node0 bash -l -s")
    assert (run_dir / "spec.toml").read_text(encoding="utf-8").startswith("# Static-batch latency")
    assert (run_dir / "lab.toml").is_file()
    assert json.loads((run_dir / "probe-node0.json").read_text(encoding="utf-8"))["record"] == "lab.probe.v1"
    # Teardown always runs last.
    assert "docker rm -f" in (executor.calls[-3][1] or "")


def test_version_mismatch_aborts_but_tears_down(tmp_path: Path) -> None:
    executor = FakeExecutor(vllm="0.28.1")
    with pytest.raises(RemoteCommandError, match=r"spec pins 0\.29\.0"):
        _runner(tmp_path, executor).run()
    manifest = json.loads((tmp_path / "2026-10-01-qwen7b-static-pp1" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RemoteCommandError"
    assert not any("bench_latency.py --model" in (stdin or "") for _, stdin in executor.calls)
    assert "pkill" in (executor.calls[-3][1] or "")


def test_dead_benchmark_fails_fast_and_collects_logs(tmp_path: Path) -> None:
    executor = FakeExecutor(polls=["complete=0 progress=3 alive=0"])
    with pytest.raises(CompletionTimeoutError) as raised:
        _runner(tmp_path, executor).run()
    assert raised.value.observed == 3 and raised.value.expected == 5
    scripts = [stdin or " ".join(argv) for argv, stdin in executor.calls]
    assert any("docker logs" in s for s in scripts)  # best-effort collect after failure


def test_existing_run_dir_is_refused(tmp_path: Path) -> None:
    (tmp_path / "2026-10-01-qwen7b-static-pp1").mkdir()
    with pytest.raises(RunDirError):
        _runner(tmp_path, FakeExecutor()).run()


def test_probe_status_parsing() -> None:
    assert ProbeStatus.parse("noise\ncomplete=1 progress=5 alive=0\n", step="s") == ProbeStatus(
        True, 5, False
    )
    with pytest.raises(RemoteCommandError):
        ProbeStatus.parse("garbage", step="s")
    with pytest.raises(RemoteCommandError):
        ProbeStatus.parse("complete=1 progress=x alive=1", step="s")


def test_parse_probe_skips_logs() -> None:
    assert parse_probe("WARNING x\n" + json.dumps(PROBE) + "\n", step="s")["torch_version"] == "2.13.0"
    with pytest.raises(RemoteCommandError):
        parse_probe('{"other": 1}\n', step="s")


def test_run_logs_every_step_with_cli_logging_enabled(tmp_path: Path) -> None:
    # The CLI enables INFO logging; LogRecords are only built when the level
    # is enabled, so reserved-attribute clashes in `extra` only surface here.
    from labharness.logging_setup import configure_logging

    configure_logging(verbose=True)
    _runner(tmp_path, FakeExecutor()).run()
    manifest = json.loads((tmp_path / "2026-10-01-qwen7b-static-pp1" / "manifest.json").read_text())
    assert manifest["status"] == "complete"
