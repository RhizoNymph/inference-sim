"""collective-bench specs, plan generation (golden), and runner control flow."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from labharness.collective_plan import build_collective_plan, rank_env
from labharness.commands import Phase, Plan, Step, Wait
from labharness.errors import RemoteCommandError, SpecError
from labharness.runner import Runner
from labharness.spec import (
    CollectiveExperiment,
    CollectiveOp,
    Experiment,
    LaunchKind,
    Mode,
    parse_experiment,
    parse_spec,
)
from tests.conftest import FIXTURES, MINIMAL_STATIC, SPECS, write_spec

RUNS_ROOT = Path("/runs")
REMOTE_SRC = Path("/src/tools/lab/remote")
NCCL_SPEC = SPECS / "rtx3090_nccl_curve.toml"

MINIMAL_COLLECTIVE = """
[experiment]
name = "t-nccl"
mode = "collective-bench"
lab = "@LAB@"
nodes = ["node0", "node1"]
launch = "venv"

[collective_bench]
"""


def _exp() -> CollectiveExperiment:
    exp = parse_spec(NCCL_SPEC)
    assert isinstance(exp, CollectiveExperiment)
    return exp


def _plan() -> Plan:
    return build_collective_plan(_exp(), date="2026-10-01", runs_root=RUNS_ROOT, remote_src=REMOTE_SRC)


# -- spec ---------------------------------------------------------------------


def test_checked_in_collective_spec() -> None:
    exp = _exp()
    assert exp.mode is Mode.COLLECTIVE_BENCH
    assert exp.launch is LaunchKind.VENV
    assert [n.name for n in exp.nodes] == ["node0", "node1"]
    assert exp.world_size == 2
    wl = exp.workload
    assert wl.ops == tuple(CollectiveOp)
    assert (wl.min_bytes, wl.max_bytes, wl.iters, wl.warmup, wl.master_port) == (
        1024,
        268435456,
        30,
        5,
        29500,
    )
    assert len(wl.message_sizes()) == 19
    assert wl.message_sizes()[0] == 1024 and wl.message_sizes()[-1] == 268435456
    # 4 collectives + 2 directed sends per size.
    assert wl.ops_per_size(2) == 6
    assert wl.expected_rows(2) == 114
    assert wl.ops_per_size(3) == 4 + 6
    assert [(r.rank, r.node.name, r.local_rank) for r in exp.ranks()] == [(0, "node0", 0), (1, "node1", 0)]


def test_minimal_collective_defaults(tmp_path: Path, lab_toml: Path) -> None:
    exp = parse_spec(write_spec(tmp_path, lab_toml, MINIMAL_COLLECTIVE))
    assert isinstance(exp, CollectiveExperiment)
    assert exp.workload.ops == tuple(CollectiveOp)
    assert exp.workload.min_bytes == 1024
    assert exp.workload.max_bytes == 256 * 1024 * 1024
    assert exp.runner.poll_interval_s == 15.0


def test_parse_spec_still_returns_vllm_experiments() -> None:
    assert isinstance(parse_spec(SPECS / "rtx3090_qwen7b_static_pp1.toml"), Experiment)


def test_parse_experiment_rejects_collective_specs() -> None:
    with pytest.raises(SpecError, match=r"only work with `lab\.py run`"):
        parse_experiment(NCCL_SPEC)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (("[collective_bench]", "[model]\nhf_id = 'x'\n[collective_bench]"), r"no \[model\] section"),
        (("[collective_bench]", "[parallelism]\ntp = 2\n[collective_bench]"), r"no \[parallelism\]"),
        (("[collective_bench]", "[engine]\n[collective_bench]"), r"no \[engine\] section"),
        (("[collective_bench]", "[serving]\n[collective_bench]"), r"no \[serving\] section"),
        (('launch = "venv"', 'launch = "venv"\nvllm_version = "0.29.0"'), "no vllm_version"),
        (('nodes = ["node0", "node1"]', 'nodes = ["node1"]'), "at least 2 GPUs"),
        (('nodes = ["node0", "node1"]\nlaunch = "venv"', 'nodes = ["node0"]\nlaunch = "docker"'), "venvs"),
        (("[collective_bench]", ""), r"requires a \[collective_bench\] section"),
        (("[collective_bench]", '[collective_bench]\nops = ["broadcast"]'), "values must be in"),
        (("[collective_bench]", '[collective_bench]\nops = ["all_reduce", "all_reduce"]'), "duplicate ops"),
        (("[collective_bench]", "[collective_bench]\nops = []"), "at least one op"),
        (("[collective_bench]", "[collective_bench]\nmin_bytes = 4096\nmax_bytes = 1024"), ">= min_bytes"),
        (("[collective_bench]", "[collective_bench]\nmin_bytes = 8"), ">= 16"),
    ],
)
def test_invalid_collective_specs(
    tmp_path: Path, lab_toml: Path, edit: tuple[str, str], message: str
) -> None:
    body = MINIMAL_COLLECTIVE.replace(*edit)
    with pytest.raises(SpecError, match=message):
        parse_spec(write_spec(tmp_path, lab_toml, body))


def test_vllm_modes_still_reject_missing_workload(tmp_path: Path, lab_toml: Path) -> None:
    body = MINIMAL_STATIC.replace('mode = "static-batch"', 'mode = "collective-bench"')
    with pytest.raises(SpecError, match=r"no \[model\] section"):
        parse_spec(write_spec(tmp_path, lab_toml, body))


# -- plan ---------------------------------------------------------------------


def test_dry_run_matches_golden() -> None:
    rendered = _plan().render()
    golden = FIXTURES / "golden_dry_run_nccl_curve.txt"
    if os.environ.get("LAB_UPDATE_GOLDEN") == "1":
        golden.write_text(rendered, encoding="utf-8")
    assert rendered == golden.read_text(encoding="utf-8")


def test_remote_scripts_never_on_the_command_line() -> None:
    for action in _plan().actions:
        step = action.as_step() if isinstance(action, Wait) else action
        if step.host is not None:
            assert step.argv() == ["ssh", "-o", "BatchMode=yes", step.host, "bash", "-l", "-s"]
            assert step.stdin() == step.script + "\n"


def test_process_patterns_are_bracketed() -> None:
    plan = _plan()
    patterns: list[str] = []
    for action in (*plan.actions, *plan.teardown):
        script = action.probe if isinstance(action, Wait) else action.script
        patterns += re.findall(r"p(?:kill|grep) -f '([^']+)'", script)
    assert patterns
    assert all(pattern.startswith("[") for pattern in patterns)


def test_phases_in_order_and_no_vllm_or_ray() -> None:
    plan = _plan()
    phases = [a.phase for a in plan.actions]
    order = [Phase.SYNC, Phase.CLEAN, Phase.PREFLIGHT, Phase.LAUNCH, Phase.WAIT, Phase.COLLECT]
    assert [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p] == order
    text = plan.render()
    assert "vllm" not in text and "ray " not in text and "docker" not in text
    sync = [a for a in plan.actions if isinstance(a, Step) and a.name.startswith("copy bench")]
    assert len(sync) == 2
    assert all("collective_bench.py" in s.script and "probe_env.py" in s.script for s in sync)
    assert all("bench_latency.py" not in s.script for s in sync)


def test_one_launch_per_rank_with_rendezvous_env() -> None:
    plan = _plan()
    launches = [a for a in plan.actions if isinstance(a, Step) and a.phase is Phase.LAUNCH]
    assert [(s.name, s.host) for s in launches] == [
        ("start rank 0 on node0", "node0"),
        ("start rank 1 on node1", "node1"),
    ]
    for rank, step in enumerate(launches):
        for fragment in (
            'PATH="$HOME/inference-sim-lab/.venv/bin:$PATH"',
            'MASTER_ADDR="10.1.1.69"',
            'MASTER_PORT="29500"',
            f'RANK="{rank}"',
            'WORLD_SIZE="2"',
            'LOCAL_RANK="0"',
            'NCCL_SOCKET_IFNAME="bond0"',
            'GLOO_SOCKET_IFNAME="bond0"',
            f"--sim-node-id {rank}",
            "nohup",
            f"collective_rank{rank}.log",
            "--ops all_reduce,all_gather,reduce_scatter,all_to_all,send_recv",
        ):
            assert fragment in step.script, (rank, fragment)


def test_rank_env_uses_head_address() -> None:
    exp = _exp()
    env = rank_env(exp, exp.ranks()[1])
    assert env["MASTER_ADDR"] == exp.head.address
    assert env["RANK"] == "1" and env["WORLD_SIZE"] == "2"


def test_wait_counts_tagged_rows_and_done() -> None:
    waits = [a for a in _plan().actions if isinstance(a, Wait)]
    assert len(waits) == 1
    wait = waits[0]
    assert wait.host == "node0" and wait.expected == 114
    assert 'grep -c \'^{.*"record": "lab.collective.v1"\'' in wait.probe
    assert "lab.done.v1" in wait.probe
    assert "-ge 114" in wait.probe
    assert "wc -l" not in wait.probe


def test_collect_and_teardown() -> None:
    plan = _plan()
    collected = [a.script for a in plan.actions if isinstance(a, Step) and a.phase is Phase.COLLECT]
    assert collected == [
        "scp -q node0:inference-sim-lab/runs/2026-10-01-nccl-curve/collective_curve.jsonl "
        "/runs/2026-10-01-nccl-curve/collective_curve.jsonl",
        "scp -q node0:inference-sim-lab/runs/2026-10-01-nccl-curve/collective_rank0.log "
        "/runs/2026-10-01-nccl-curve/collective_rank0.log",
        "scp -q node1:inference-sim-lab/runs/2026-10-01-nccl-curve/collective_rank1.log "
        "/runs/2026-10-01-nccl-curve/collective_rank1.log",
    ]
    assert [(s.host, s.check) for s in plan.teardown] == [("node0", False), ("node1", False)]
    assert all("pkill -f '[c]ollective_bench.py'" in s.script for s in plan.teardown)


def test_vllm_golden_plans_unaffected_by_layout_change() -> None:
    # The vLLM goldens are covered in test_commands; this guards the shared layout helper.
    plan = _plan()
    assert plan.remote_run_dir == "$HOME/inference-sim-lab/runs/2026-10-01-nccl-curve"
    assert plan.local_run_dir == RUNS_ROOT / "2026-10-01-nccl-curve"


# -- runner -------------------------------------------------------------------

TORCH_PROBE: dict[str, object] = {
    "record": "lab.probe.v1",
    "vllm_version": None,
    "torch_version": "2.13.0",
    "nccl_version": "2.28.3",
    "gpus": [],
}


class FakeExecutor:
    def __init__(self, probe: dict[str, object]) -> None:
        self.probe = probe
        self.calls: list[tuple[list[str], str | None]] = []
        self.polls = ["complete=0 progress=40 alive=1", "complete=1 progress=114 alive=1"]

    def __call__(
        self, argv: list[str], stdin: str | None, timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, stdin))
        text = stdin or " ".join(argv)
        out = ""
        if "probe_env.py" in text:
            out = json.dumps(self.probe) + "\n"
        elif "complete=$c" in text:
            out = self.polls.pop(0) + "\n"
        elif argv[:1] == ["git"]:
            out = "cafef00d\n" if "rev-parse" in argv else ""
        return subprocess.CompletedProcess(argv, 0, out, "")


def _runner(tmp_path: Path, executor: FakeExecutor) -> Runner:
    exp = _exp()
    plan = build_collective_plan(exp, date="2026-10-01", runs_root=tmp_path, remote_src=Path("/src"))
    return Runner(exp, plan, repo=tmp_path, executor=executor, sleep=lambda _: None)


def test_collective_run_writes_manifest_without_vllm_pin(tmp_path: Path) -> None:
    executor = FakeExecutor(TORCH_PROBE)
    _runner(tmp_path, executor).run()
    run_dir = tmp_path / "2026-10-01-nccl-curve"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["mode"] == "collective-bench"
    assert manifest["vllm_version_pin"] is None
    assert set(manifest["probes"]) == {"node0", "node1"}
    launches = [stdin for _, stdin in executor.calls if stdin and '.py" --min-bytes' in stdin]
    assert len(launches) == 2
    assert "pkill -f '[c]ollective_bench.py'" in (executor.calls[-3][1] or "")


@pytest.mark.parametrize("missing", ["torch_version", "nccl_version"])
def test_probe_without_torch_aborts_before_launch(tmp_path: Path, missing: str) -> None:
    executor = FakeExecutor({**TORCH_PROBE, missing: None})
    with pytest.raises(RemoteCommandError, match=f"no {missing}"):
        _runner(tmp_path, executor).run()
    assert not any(stdin and "--min-bytes" in stdin for _, stdin in executor.calls)
    manifest = json.loads((tmp_path / "2026-10-01-nccl-curve" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
