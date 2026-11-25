"""Pure command generation for `collective-bench` experiments.

Same conventions as `commands.py` (scripts on stdin, bracketed process
patterns, detached processes polled by a `Wait`), but no vLLM, Ray, or
containers: one detached `collective_bench.py` process per GPU, wired together
by torch.distributed's env:// rendezvous on the head node.

Phases: sync (bench + probe scripts), clean, preflight (probe; the runner
requires torch and NCCL), launch (one process per rank, rank 0 first), wait
(counts `lab.collective.v1` lines plus `lab.done.v1` in rank 0's output),
collect (`collective_curve.jsonl` and every rank's log), teardown.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Final

from labharness.commands import (
    DONE_RECORD,
    Action,
    Capture,
    CaptureKind,
    Phase,
    Plan,
    Step,
    Wait,
    _collect,
    _env_prefix,
    _Layout,
    _nohup,
    _process_alive,
    _remote_rel,
    run_id_for,
)
from labharness.errors import SpecError
from labharness.spec import CollectiveExperiment, GpuRank, Node, VenvLaunch

COLLECTIVE_RECORD: Final = "lab.collective.v1"
COLLECTIVE_SCRIPTS: Final = ("collective_bench.py", "probe_env.py")
RESULTS_NAME: Final = "collective_curve.jsonl"
BENCH_PATTERN: Final = "[c]ollective_bench.py"


def rank_log_name(rank: int) -> str:
    return f"collective_rank{rank}.log"


def _venv(exp: CollectiveExperiment, node: Node) -> VenvLaunch:
    match node.launch(exp.launch):
        case VenvLaunch() as launch:
            return launch
        case _:
            raise SpecError("collective-bench requires venv launch", path=str(exp.path), field="launch")


def _base_env(exp: CollectiveExperiment, node: Node) -> dict[str, str]:
    env = {
        "PATH": f"{_venv(exp, node).venv}/bin:$PATH",
        "NCCL_SOCKET_IFNAME": exp.lab.socket_ifname,
        "GLOO_SOCKET_IFNAME": exp.lab.socket_ifname,
        "PYTHONUNBUFFERED": "1",
    }
    env.update(exp.lab.extra_env)
    return env


def rank_env(exp: CollectiveExperiment, rank: GpuRank) -> dict[str, str]:
    """torch.distributed env:// rendezvous variables plus the lab's socket interface."""
    env = _base_env(exp, rank.node)
    env |= {
        "MASTER_ADDR": exp.head.address,
        "MASTER_PORT": str(exp.workload.master_port),
        "RANK": str(rank.rank),
        "WORLD_SIZE": str(exp.world_size),
        "LOCAL_RANK": str(rank.local_rank),
    }
    return env


def bench_args(exp: CollectiveExperiment, rank: GpuRank, output: str) -> list[str]:
    wl = exp.workload
    return [
        "--min-bytes", str(wl.min_bytes),
        "--max-bytes", str(wl.max_bytes),
        "--iters", str(wl.iters),
        "--warmup", str(wl.warmup),
        "--ops", ",".join(op.value for op in wl.ops),
        "--sim-node-id", str(rank.node.sim_node_id),
        "--output", output,
    ]  # fmt: skip


def _sync_steps(exp: CollectiveExperiment, layout: _Layout, remote_src: Path) -> list[Step]:
    sources = " ".join(shlex.quote(str(remote_src / name)) for name in COLLECTIVE_SCRIPTS)
    steps: list[Step] = []
    for node in exp.nodes:
        steps += [
            Step(
                Phase.SYNC,
                f"mkdir {node.name}",
                node.ssh_host,
                f'mkdir -p "{layout.bench_dir}" "{layout.run_dir}"',
            ),
            Step(
                Phase.SYNC,
                f"copy bench scripts to {node.name}",
                None,
                f"scp -q {sources} {node.ssh_host}:{_remote_rel(layout.bench_dir)}/",
            ),
        ]
    return steps


def _kill_steps(exp: CollectiveExperiment, phase: Phase, *, check: bool) -> list[Step]:
    return [
        Step(
            phase,
            f"stop stale processes on {node.name}",
            node.ssh_host,
            f"pkill -f '{BENCH_PATTERN}' || true",
            check=check,
        )
        for node in exp.nodes
    ]


def _probe_steps(exp: CollectiveExperiment, layout: _Layout) -> list[Step]:
    return [
        Step(
            Phase.PREFLIGHT,
            f"probe {node.name}",
            node.ssh_host,
            f'{_env_prefix(_base_env(exp, node))} python "{layout.bench_dir}/probe_env.py"',
            capture=Capture(CaptureKind.PROBE, f"probe-{node.name}.json", node.name),
        )
        for node in exp.nodes
    ]


def _launch_steps(exp: CollectiveExperiment, layout: _Layout) -> list[Step]:
    results = f'"{layout.run_dir}/{RESULTS_NAME}"'
    steps: list[Step] = []
    for rank in exp.ranks():
        args = shlex.join(bench_args(exp, rank, "@OUTPUT@")).replace("@OUTPUT@", results)
        steps.append(
            Step(
                Phase.LAUNCH,
                f"start rank {rank.rank} on {rank.node.name}",
                rank.node.ssh_host,
                _nohup(
                    rank_env(exp, rank),
                    layout,
                    f'python "{layout.bench_dir}/collective_bench.py" {args}',
                    rank_log_name(rank.rank),
                ),
            )
        )
    return steps


def _wait(exp: CollectiveExperiment, layout: _Layout) -> Wait:
    results = f"{layout.run_dir}/{RESULTS_NAME}"
    expected = exp.workload.expected_rows(exp.world_size)
    probe = (
        f'n=$(grep -c \'^{{.*"record": "{COLLECTIVE_RECORD}"\' "{results}" 2>/dev/null); '
        f'd=$(grep -c \'^{{.*"record": "{DONE_RECORD}"\' "{results}" 2>/dev/null); '
        f"{_process_alive(f'{BENCH_PATTERN}.*{layout.run_id}')}; "
        f'if [ "${{d:-0}}" -ge 1 ] && [ "${{n:-0}}" -ge {expected} ]; '
        "then c=1; else c=0; fi; "
        'echo "complete=$c progress=${n:-0} alive=$a"'
    )
    return Wait(
        "collective results",
        exp.head.ssh_host,
        probe,
        expected=expected,
        timeout_s=exp.runner.run_timeout_s,
    )


def _collect_steps(exp: CollectiveExperiment, layout: _Layout) -> list[Step]:
    steps = [_collect(exp.head.ssh_host, layout, RESULTS_NAME, RESULTS_NAME)]
    for rank in exp.ranks():
        log = rank_log_name(rank.rank)
        steps.append(_collect(rank.node.ssh_host, layout, log, log))
    return steps


def build_collective_plan(exp: CollectiveExperiment, *, date: str, runs_root: Path, remote_src: Path) -> Plan:
    run_id = run_id_for(exp, date)
    layout = _Layout(exp.lab.remote_root, run_id, runs_root / run_id)
    actions: list[Action] = [
        *_sync_steps(exp, layout, remote_src),
        *_kill_steps(exp, Phase.CLEAN, check=True),
        *_probe_steps(exp, layout),
        *_launch_steps(exp, layout),
        _wait(exp, layout),
        *_collect_steps(exp, layout),
    ]
    return Plan(
        run_id=run_id,
        remote_run_dir=layout.run_dir,
        local_run_dir=layout.local,
        actions=tuple(actions),
        teardown=tuple(_kill_steps(exp, Phase.TEARDOWN, check=False)),
    )
