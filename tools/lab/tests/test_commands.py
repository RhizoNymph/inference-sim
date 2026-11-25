"""Command generation: golden dry-run output plus targeted invariants.

Set LAB_UPDATE_GOLDEN=1 to rewrite the golden files after an intended change,
then review the diff.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

import pytest

from labharness.commands import Phase, Plan, Step, Wait, bench_serve_args, build_plan
from labharness.spec import ServingWorkload, parse_experiment
from tests.conftest import FIXTURES, SPECS

RUNS_ROOT = Path("/runs")
REMOTE_SRC = Path("/src/tools/lab/remote")
CASES = {
    "static_pp1": ("rtx3090_qwen7b_static_pp1.toml", "2026-09-28"),
    "static_pp2": ("rtx3090_qwen7b_static_pp2.toml", "2026-09-28"),
    "serving_pp1": ("rtx3090_qwen7b_serving_pp1.toml", "2026-09-30"),
}


def _plan(case: str) -> Plan:
    spec, date = CASES[case]
    return build_plan(parse_experiment(SPECS / spec), date=date, runs_root=RUNS_ROOT, remote_src=REMOTE_SRC)


@pytest.mark.parametrize("case", sorted(CASES))
def test_dry_run_matches_golden(case: str) -> None:
    rendered = _plan(case).render()
    golden = FIXTURES / f"golden_dry_run_{case}.txt"
    if os.environ.get("LAB_UPDATE_GOLDEN") == "1":
        golden.write_text(rendered, encoding="utf-8")
    assert rendered == golden.read_text(encoding="utf-8")


@pytest.mark.parametrize("case", sorted(CASES))
def test_remote_scripts_never_on_the_command_line(case: str) -> None:
    for action in _plan(case).actions:
        step = action.as_step() if isinstance(action, Wait) else action
        if step.host is not None:
            assert step.argv() == ["ssh", "-o", "BatchMode=yes", step.host, "bash", "-l", "-s"]
            assert step.stdin() == step.script + "\n"


@pytest.mark.parametrize("case", sorted(CASES))
def test_process_patterns_are_bracketed(case: str) -> None:
    plan = _plan(case)
    for action in (*plan.actions, *plan.teardown):
        script = action.probe if isinstance(action, Wait) else action.script
        for pattern in re.findall(r"p(?:kill|grep) -f '([^']+)'", script):
            assert pattern.startswith("["), pattern


def test_hf_home_always_overridden() -> None:
    for case in CASES:
        for action in _plan(case).actions:
            if (
                isinstance(action, Step)
                and ("vllm" in action.script or "bench_latency" in action.script)
                and action.phase in {Phase.LAUNCH, Phase.PREFLIGHT}
            ):
                assert "HF_HOME=" in action.script, action.name


def test_multi_node_ray_environment() -> None:
    plan = _plan("static_pp2")
    cluster = [a for a in plan.actions if isinstance(a, Step) and a.phase is Phase.CLUSTER]
    assert [s.name for s in cluster] == ["ray head on node0", "ray worker on node1", "verify ray sees 2 GPUs"]
    for step in cluster:
        # The venv must be on PATH for the ray daemons (vLLM JIT needs its ninja).
        assert 'PATH="$HOME/inference-sim-lab/.venv/bin:$PATH"' in step.script
        assert 'NCCL_SOCKET_IFNAME="bond0"' in step.script
        assert 'GLOO_SOCKET_IFNAME="bond0"' in step.script
    assert "--address 10.1.1.69:6379 --node-ip-address 10.1.1.68" in cluster[1].script
    launch = next(a for a in plan.actions if isinstance(a, Step) and a.phase is Phase.LAUNCH)
    assert 'PATH="$HOME/inference-sim-lab/.venv/bin:$PATH"' in launch.script
    assert "--backend ray" in launch.script
    teardown = " ".join(s.script for s in plan.teardown)
    assert teardown.count('ray" stop --force') == 2


def test_static_completion_counts_json_records_only() -> None:
    plan = _plan("static_pp1")
    wait = next(a for a in plan.actions if isinstance(a, Wait))
    assert wait.expected == 5
    assert 'grep -c \'^{.*"record": "lab.static_batch.v1"\'' in wait.probe
    assert "lab.done.v1" in wait.probe
    assert "wc -l" not in wait.probe


# The command validated on node0 on 2026-09-30 (run_serving.sh).
GROUND_TRUTH_SERVER = (
    "docker run -d --name {name} --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host "
    '--user "$(id -u):$(id -g)"'
)
GROUND_TRUTH_SERVER_ARGS = (
    "vllm/vllm-openai:v0.29.0 --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-model-len 4096 "
    "--max-num-batched-tokens 2048 --max-num-seqs 64 --gpu-memory-utilization 0.85 "
    "--no-enable-prefix-caching --seed 0"
)
GROUND_TRUTH_CLIENT = (
    "--backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B-Instruct "
    "--dataset-name random --random-input-len 512 --random-output-len 128 --random-range-ratio 0 "
    "--ignore-eos --num-prompts 200 --request-rate {rate} --seed 0 --disable-tqdm "
    "--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 "
    "--save-result --result-dir DIR --result-filename rate_{rate}.json"
)


def test_serving_matches_validated_commands() -> None:
    plan = _plan("serving_pp1")
    server = next(a for a in plan.actions if isinstance(a, Step) and a.name == "start vllm serve")
    assert server.script.startswith(GROUND_TRUTH_SERVER.format(name="lab-2026-09-30-qwen7b-serving-pp1"))
    assert server.script.endswith(GROUND_TRUTH_SERVER_ARGS)
    for fragment in (
        '-e HOME="/tmp"',
        '-e HF_HOME="/hf"',
        '-e VLLM_CACHE_ROOT="/tmp/vllm-cache"',
        '-v "$HOME/.cache/huggingface:/hf"',
        "-p 127.0.0.1:8000:8000",
    ):
        assert fragment in server.script

    exp = parse_experiment(SPECS / "rtx3090_qwen7b_serving_pp1.toml")
    assert isinstance(exp.workload, ServingWorkload)
    for rate in exp.workload.request_rates:
        args = shlex.join(bench_serve_args(exp, exp.workload, rate, "DIR"))
        text = "inf" if rate == float("inf") else f"{rate:g}"
        assert args == GROUND_TRUTH_CLIENT.format(rate=text)

    clients = [a for a in plan.actions if isinstance(a, Step) and a.name.startswith("bench serve")]
    assert len(clients) == 7
    # The client runs from the node0 venv, not inside the server container.
    assert all("vllm bench serve" in c.script and "docker exec" not in c.script for c in clients)
    waits = [a for a in plan.actions if isinstance(a, Wait)]
    assert waits[0].name == "server ready" and "curl -sf http://127.0.0.1:8000/health" in waits[0].probe
    # User-mode containers need no chown pass.
    assert not any("chown" in s.script for s in plan.teardown)


def test_collect_steps_land_in_run_dir() -> None:
    plan = _plan("serving_pp1")
    assert plan.local_run_dir == RUNS_ROOT / "2026-09-30-qwen7b-serving-pp1"
    collected = [
        a.script.split()[-1] for a in plan.actions if isinstance(a, Step) and a.name.startswith("collect")
    ]
    assert "/runs/2026-09-30-qwen7b-serving-pp1/rate_inf.json" in collected
    assert "/runs/2026-09-30-qwen7b-serving-pp1/server.log" in collected
