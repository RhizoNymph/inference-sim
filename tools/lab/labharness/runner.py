"""Execute (or print) a Plan and record the run in lab-runs/<date>-<name>/.

The runner is deliberately sequential and synchronous: an experiment is a
strict sequence of remote steps separated by polling waits. Command execution
goes through an injectable `Executor`, so tests drive the control flow
without ssh.

Run directory contents written here:

* `spec.toml`, `lab.toml`      - verbatim copies of the inputs;
* `probe-<node>.json`          - probe_env.py output per node;
* `manifest.json`              - run id, status, timings, simulator git sha,
                                 node probes, and every rendered command;
* collected results            - `measured.jsonl` (static), `rate_*.json`
                                 (serving), or `collective_curve.jsonl`
                                 (collective-bench) plus logs, from the plan's
                                 collect steps.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from labharness.commands import CaptureKind, Phase, Plan, Step, Wait
from labharness.errors import CompletionTimeoutError, LabError, RemoteCommandError, RunDirError
from labharness.logging_setup import get_logger
from labharness.spec import CollectiveExperiment, Experiment, Spec

STEP_TIMEOUT_S = 900.0


class Executor(Protocol):
    def __call__(
        self, argv: list[str], stdin: str | None, timeout_s: float
    ) -> subprocess.CompletedProcess[str]: ...


def subprocess_executor(
    argv: list[str], stdin: str | None, timeout_s: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout_s, check=False)


@dataclass(frozen=True, slots=True)
class ProbeStatus:
    complete: bool
    progress: int
    alive: bool

    @classmethod
    def parse(cls, text: str, *, step: str) -> ProbeStatus:
        for line in reversed(text.strip().splitlines()):
            fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
            if {"complete", "progress", "alive"} <= fields.keys():
                try:
                    return cls(
                        complete=fields["complete"] == "1",
                        progress=int(fields["progress"]),
                        alive=fields["alive"] == "1",
                    )
                except ValueError:
                    break
        raise RemoteCommandError(f"unparseable probe output: {text[-200:]!r}", step=step, returncode=None)


def git_state(repo: Path, executor: Executor) -> dict[str, object]:
    sha = executor(["git", "-C", str(repo), "rev-parse", "HEAD"], None, 30.0)
    status = executor(["git", "-C", str(repo), "status", "--porcelain"], None, 30.0)
    return {
        "simulator_git_sha": sha.stdout.strip() if sha.returncode == 0 else None,
        "simulator_git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def parse_probe(stdout: str, *, step: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("record") == "lab.probe.v1":
            return record
    raise RemoteCommandError("probe produced no lab.probe.v1 record", step=step, returncode=0)


class Runner:
    def __init__(
        self,
        exp: Spec,
        plan: Plan,
        *,
        repo: Path,
        executor: Executor = subprocess_executor,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.exp = exp
        self.plan = plan
        self.repo = repo
        self.executor = executor
        self.sleep = sleep
        self.monotonic = monotonic
        self.log = get_logger()
        self.probes: dict[str, dict[str, object]] = {}

    # -- execution ---------------------------------------------------------

    def _run_step(self, step: Step) -> str:
        self.log.info(
            "step", extra={"phase": step.phase.value, "step_name": step.name, "host": step.host or "local"}
        )
        try:
            result = self.executor(step.argv(), step.stdin(), STEP_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            raise RemoteCommandError(
                f"step timed out after {STEP_TIMEOUT_S:g}s", step=step.name, returncode=None
            ) from error
        if result.returncode != 0 and step.check:
            raise RemoteCommandError(
                f"step failed: {result.stderr.strip()[-400:]}", step=step.name, returncode=result.returncode
            )
        if step.capture is not None:
            path = self.plan.local_run_dir / step.capture.filename
            match step.capture.kind:
                case CaptureKind.PROBE:
                    probe = parse_probe(result.stdout, step=step.name)
                    self.probes[step.capture.node] = probe
                    path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
                    self._check_probe(probe, node=step.capture.node, step=step.name)
                case CaptureKind.TEXT:
                    path.write_text(result.stdout, encoding="utf-8")
        return result.stdout

    def _check_probe(self, probe: dict[str, object], *, node: str, step: str) -> None:
        """vLLM modes need the pinned vLLM; collective-bench needs torch with NCCL."""
        match self.exp:
            case Experiment() as exp:
                if probe.get("vllm_version") != exp.vllm_version:
                    raise RemoteCommandError(
                        f"{node} runs vllm {probe.get('vllm_version')}, spec pins {exp.vllm_version}",
                        step=step,
                        returncode=None,
                    )
            case CollectiveExperiment():
                missing = [key for key in ("torch_version", "nccl_version") if not probe.get(key)]
                if missing:
                    raise RemoteCommandError(
                        f"{node} probe reports no {' or '.join(missing)}; collective-bench needs torch "
                        "with CUDA and NCCL in the node's venv",
                        step=step,
                        returncode=None,
                    )

    def _wait(self, wait: Wait) -> None:
        deadline = self.monotonic() + wait.timeout_s
        status = ProbeStatus(complete=False, progress=0, alive=True)
        while self.monotonic() < deadline:
            status = ProbeStatus.parse(self._run_step(wait.as_step()), step=wait.name)
            self.log.info(
                "poll",
                extra={
                    "wait": wait.name,
                    "progress": status.progress,
                    "expected": wait.expected,
                    "alive": status.alive,
                },
            )
            if status.complete:
                return
            if not status.alive:
                raise CompletionTimeoutError(
                    "process exited before completing",
                    step=wait.name,
                    observed=status.progress,
                    expected=wait.expected,
                )
            self.sleep(self.exp.runner.poll_interval_s)
        raise CompletionTimeoutError(
            f"no completion within {wait.timeout_s:g}s",
            step=wait.name,
            observed=status.progress,
            expected=wait.expected,
        )

    def _best_effort(self, steps: list[Step]) -> None:
        for step in steps:
            try:
                self._run_step(Step(step.phase, step.name, step.host, step.script, check=False))
            except LabError as error:
                self.log.warning("best-effort step failed", extra={"step": step.name, "error": str(error)})

    # -- entry points ------------------------------------------------------

    def _prepare_dir(self) -> None:
        run_dir = self.plan.local_run_dir
        if run_dir.exists():
            raise RunDirError("run directory already exists; pick another --date or name", path=str(run_dir))
        run_dir.mkdir(parents=True)
        (run_dir / "spec.toml").write_text(self.exp.source_text, encoding="utf-8")
        (run_dir / "lab.toml").write_text(self.exp.lab.path.read_text(encoding="utf-8"), encoding="utf-8")

    def _manifest(self, status: str, started: str, error: LabError | None) -> dict[str, object]:
        return {
            "record": "lab.manifest.v1",
            "run_id": self.plan.run_id,
            "experiment": self.exp.name,
            "mode": self.exp.mode.value,
            "vllm_version_pin": self.exp.vllm_version if isinstance(self.exp, Experiment) else None,
            "nodes": [node.name for node in self.exp.nodes],
            "launch": self.exp.launch.value,
            "remote_run_dir": self.plan.remote_run_dir,
            "status": status,
            "error": None
            if error is None
            else {"type": type(error).__name__, "message": str(error), **error.details()},
            "started_at": started,
            "finished_at": _now(),
            **git_state(self.repo, self.executor),
            "probes": self.probes,
            "spec_file": "spec.toml",
            "lab_file": "lab.toml",
            "commands": [action.render() for action in (*self.plan.actions, *self.plan.teardown)],
        }

    def run(self) -> None:
        self._prepare_dir()
        started = _now()
        collect = [a for a in self.plan.actions if isinstance(a, Step) and a.phase is Phase.COLLECT]
        failure: LabError | None = None
        try:
            for action in self.plan.actions:
                match action:
                    case Wait():
                        self._wait(action)
                    case Step():
                        self._run_step(action)
        except LabError as error:
            failure = error
            self.log.error(str(error), extra={"error": type(error).__name__, **error.details()})
            self._best_effort(collect)
        finally:
            self._best_effort(list(self.plan.teardown))
            manifest = self._manifest("failed" if failure else "complete", started, failure)
            (self.plan.local_run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        if failure is not None:
            raise failure


def dry_run(plan: Plan, out: TextIO) -> None:
    out.write(plan.render())
