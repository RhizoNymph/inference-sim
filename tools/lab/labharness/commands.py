"""Pure command generation: Experiment -> ordered Plan of shell steps and waits.

Nothing here touches the network or the filesystem; `runner.py` executes (or,
with --dry-run, prints) the plan. Keeping generation pure is what makes the
dry-run output golden-testable.

Conventions every generated script follows:

* Remote steps run as `ssh -o BatchMode=yes HOST bash -l -s` with the script
  on stdin (rendered as a heredoc in --dry-run output), so scripts are never
  re-quoted. The login shell puts docker/uv on PATH; HF_HOME is always set
  explicitly per process because the nodes' own defaults are wrong.
* Remote paths use `$HOME/...` inside double quotes so they expand remotely.
* Process matching uses bracketed patterns (`[b]ench_latency`) so `pkill -f` /
  `pgrep -f` never match the ssh command line that carries the pattern itself.
* Every benchmark runs detached (nohup or `docker run -d`) and is polled, so a
  dropped ssh session never kills a measurement.
* Probes print exactly one line `complete=<0|1> progress=<n> alive=<0|1>`.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from labharness.errors import SpecError
from labharness.spec import (
    CollectiveExperiment,
    ContainerUser,
    DockerLaunch,
    Experiment,
    Node,
    ServingWorkload,
    StaticBatchWorkload,
    VenvLaunch,
    rate_label,
    rate_text,
)

STATIC_RECORD: Final = "lab.static_batch.v1"
DONE_RECORD: Final = "lab.done.v1"
REMOTE_SCRIPTS: Final = ("bench_latency.py", "probe_env.py")

# ---------------------------------------------------------------------------
# Plan types
# ---------------------------------------------------------------------------


class Phase(StrEnum):
    SYNC = "sync"
    PREFLIGHT = "preflight"
    CLEAN = "clean"
    CLUSTER = "cluster"
    LAUNCH = "launch"
    WAIT = "wait"
    COLLECT = "collect"
    TEARDOWN = "teardown"


class CaptureKind(StrEnum):
    PROBE = "probe"  # one JSON line from probe_env.py, validated against the pin
    TEXT = "text"  # stored verbatim


@dataclass(frozen=True, slots=True)
class Capture:
    kind: CaptureKind
    filename: str
    node: str


@dataclass(frozen=True, slots=True)
class Step:
    phase: Phase
    name: str
    host: str | None  # ssh host; None runs locally
    script: str
    capture: Capture | None = None
    check: bool = True

    def argv(self) -> list[str]:
        if self.host is None:
            return ["bash", "-c", self.script]
        # The script travels on stdin, so it is never re-quoted and never
        # appears on the ssh command line (where pkill -f could match it).
        return ["ssh", "-o", "BatchMode=yes", self.host, "bash", "-l", "-s"]

    def stdin(self) -> str | None:
        return None if self.host is None else self.script + "\n"

    def render(self) -> str:
        if self.host is None:
            return self.script
        return f"ssh -o BatchMode=yes {self.host} bash -l -s <<'LAB_EOF'\n{self.script}\nLAB_EOF"


@dataclass(frozen=True, slots=True)
class Wait:
    """Poll `probe` on `host` until it reports complete=1."""

    name: str
    host: str
    probe: str
    expected: int
    timeout_s: float

    @property
    def phase(self) -> Phase:
        return Phase.WAIT

    def as_step(self) -> Step:
        return Step(Phase.WAIT, self.name, self.host, self.probe)

    def render(self) -> str:
        return self.as_step().render()


type Action = Step | Wait


@dataclass(frozen=True, slots=True)
class Plan:
    run_id: str
    remote_run_dir: str
    local_run_dir: Path
    actions: tuple[Action, ...]
    teardown: tuple[Step, ...]

    def render(self) -> str:
        lines = [f"# plan {self.run_id}", f"# local results -> {self.local_run_dir}"]
        for action in (*self.actions, *self.teardown):
            match action:
                case Wait():
                    lines.append(
                        f"# [{action.phase}] {action.name} "
                        f"(poll until complete=1, expected={action.expected}, "
                        f"timeout={action.timeout_s:g}s)"
                    )
                case Step():
                    suffix = f" -> {action.capture.filename}" if action.capture else ""
                    lines.append(f"# [{action.phase}] {action.name}{suffix}")
            lines.append(action.render())
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _remote_rel(path: str) -> str:
    """`$HOME/x/y` -> `x/y` (scp resolves relative paths against $HOME)."""
    prefix = "$HOME/"
    if not path.startswith(prefix):
        raise SpecError(f"remote paths must start with $HOME/: {path}", path="lab", field="remote_root")
    return path[len(prefix) :]


def _venv_env(exp: Experiment, node: Node, launch: VenvLaunch) -> dict[str, str]:
    """Environment for native processes: ray daemons, drivers, bench clients.

    The venv's bin/ goes first on PATH: Ray workers inherit the daemon's PATH
    and vLLM JIT-compiles kernels with the venv's `ninja` (node0 has no system
    ninja), so `ray start` and the driver must both see it.
    """
    env = {
        "PATH": f"{launch.venv}/bin:$PATH",
        "HF_HOME": exp.lab.hf_home,
        "NCCL_SOCKET_IFNAME": exp.lab.socket_ifname,
        "GLOO_SOCKET_IFNAME": exp.lab.socket_ifname,
        "VLLM_HOST_IP": node.address,
        "PYTHONUNBUFFERED": "1",
    }
    env.update(exp.lab.extra_env)
    return env


def _docker_env(exp: Experiment, launch: DockerLaunch) -> dict[str, str]:
    env = {"HF_HOME": launch.container_hf_home}
    if launch.container_user is ContainerUser.USER:
        env |= {"HOME": "/tmp", "VLLM_CACHE_ROOT": "/tmp/vllm-cache"}
    env |= {"PYTHONUNBUFFERED": "1"}
    env.update(exp.lab.extra_env)
    return env


def _env_prefix(env: dict[str, str]) -> str:
    # Values stay double-quoted so `$HOME`/`$PATH` expand on the remote side.
    return "env " + " ".join(f'{key}="{value}"' for key, value in env.items())


class _Layout:
    """Remote and local paths for one run."""

    def __init__(self, remote_root: str, run_id: str, local_run_dir: Path) -> None:
        self.root = remote_root
        self.bench_dir = f"{self.root}/bench"
        self.run_dir = f"{self.root}/runs/{run_id}"
        self.run_rel = _remote_rel(self.run_dir)
        self.local = local_run_dir
        self.run_id = run_id
        self.container = f"lab-{run_id}"
        # Inside the container the lab root is mounted at /lab.
        self.c_bench_dir = "/lab/bench"
        self.c_run_dir = f"/lab/runs/{run_id}"


def _docker_run(
    exp: Experiment,
    launch: DockerLaunch,
    *,
    entrypoint: str | None,
    args: Sequence[str],
    name: str | None,
    publish_port: int | None = None,
) -> str:
    """`docker run` for this lab's container quirks (runtime flags, user, mounts)."""
    parts = ["docker run", f"-d --name {name}" if name else "--rm"]
    parts += list(launch.runtime_args)
    parts.append("--ipc=host")
    if launch.container_user is ContainerUser.USER:
        parts.append('--user "$(id -u):$(id -g)"')
    parts += [f'-e {key}="{value}"' for key, value in _docker_env(exp, launch).items()]
    parts += [
        f'-v "{exp.lab.hf_home}:{launch.container_hf_home}"',
        f'-v "{exp.lab.remote_root}:/lab"',
    ]
    if publish_port is not None:
        parts.append(f"-p 127.0.0.1:{publish_port}:{publish_port}")
    if entrypoint is not None:
        parts.append(f"--entrypoint {entrypoint}")
    parts += [launch.image, shlex.join(list(args))]
    return " ".join(parts)


def _engine_args(exp: Experiment) -> list[str]:
    """Engine flags shared by `vllm serve`; parallel sizes only when > 1."""
    engine = exp.engine
    args = ["--dtype", exp.model.dtype, "--max-model-len", str(exp.model.max_model_len)]
    if exp.parallelism.tp > 1:
        args += ["--tensor-parallel-size", str(exp.parallelism.tp)]
    if exp.parallelism.pp > 1:
        args += ["--pipeline-parallel-size", str(exp.parallelism.pp)]
    args += [
        "--max-num-batched-tokens", str(engine.max_num_batched_tokens),
        "--max-num-seqs", str(engine.max_num_seqs),
        "--gpu-memory-utilization", f"{engine.gpu_memory_utilization:g}",
        "--enable-prefix-caching" if engine.enable_prefix_caching else "--no-enable-prefix-caching",
        "--seed", str(engine.seed),
    ]  # fmt: skip
    if engine.enforce_eager:
        args.append("--enforce-eager")
    if exp.multi_node:
        args += ["--distributed-executor-backend", "ray"]
    return args


# ---------------------------------------------------------------------------
# Phases common to both modes
# ---------------------------------------------------------------------------


def _sync_steps(exp: Experiment, layout: _Layout, remote_src: Path) -> list[Step]:
    steps: list[Step] = []
    sources = " ".join(shlex.quote(str(remote_src / name)) for name in REMOTE_SCRIPTS)
    for node in exp.nodes:
        steps.append(
            Step(
                Phase.SYNC,
                f"mkdir {node.name}",
                node.ssh_host,
                f'mkdir -p "{layout.bench_dir}" "{layout.run_dir}"',
            )
        )
        steps.append(
            Step(
                Phase.SYNC,
                f"copy bench scripts to {node.name}",
                None,
                f"scp -q {sources} {node.ssh_host}:{_remote_rel(layout.bench_dir)}/",
            )
        )
    return steps


def _probe_steps(exp: Experiment, layout: _Layout) -> list[Step]:
    steps: list[Step] = []
    for node in exp.nodes:
        match node.launch(exp.launch):
            case DockerLaunch() as launch:
                script = _docker_run(
                    exp,
                    launch,
                    entrypoint="python3",
                    args=[f"{layout.c_bench_dir}/probe_env.py"],
                    name=None,
                )
            case VenvLaunch() as launch:
                script = (
                    f'{_env_prefix(_venv_env(exp, node, launch))} python "{layout.bench_dir}/probe_env.py"'
                )
        steps.append(
            Step(
                Phase.PREFLIGHT,
                f"probe {node.name}",
                node.ssh_host,
                script,
                capture=Capture(CaptureKind.PROBE, f"probe-{node.name}.json", node.name),
            )
        )
    return steps


def _kill_steps(exp: Experiment, layout: _Layout, phase: Phase) -> list[Step]:
    steps: list[Step] = []
    for node in exp.nodes:
        commands = [
            "pkill -f '[b]ench_latency.py' || true",
            "pkill -f '[v]llm bench serve' || true",
            "pkill -f '[v]llm serve' || true",
        ]
        if _uses_docker(node, exp):
            commands.append(f"docker rm -f {layout.container} >/dev/null 2>&1 || true")
        for launch in node.launches.values():
            if isinstance(launch, VenvLaunch):
                commands.append(f'"{launch.bin("ray")}" stop --force >/dev/null 2>&1 || true')
        steps.append(Step(phase, f"stop stale processes on {node.name}", node.ssh_host, "; ".join(commands)))
    return steps


def _uses_docker(node: Node, exp: Experiment) -> bool:
    return isinstance(node.launch(exp.launch), DockerLaunch)


def _ray_steps(exp: Experiment) -> list[Step]:
    if not exp.multi_node:
        return []
    head = exp.head
    head_launch = head.launch(exp.launch)
    if not isinstance(head_launch, VenvLaunch):
        raise SpecError("multi-node requires venv launch", path=str(exp.path), field="launch")
    port = exp.lab.ray_port
    steps = [
        Step(
            Phase.CLUSTER,
            f"ray head on {head.name}",
            head.ssh_host,
            f"{_env_prefix(_venv_env(exp, head, head_launch))} ray start --head "
            f"--node-ip-address {head.address} --port {port} "
            f"--num-gpus {head.gpu_count} --disable-usage-stats",
        )
    ]
    for node in exp.nodes[1:]:
        launch = node.launch(exp.launch)
        if not isinstance(launch, VenvLaunch):
            raise SpecError("multi-node requires venv launch", path=str(exp.path), field="launch")
        steps.append(
            Step(
                Phase.CLUSTER,
                f"ray worker on {node.name}",
                node.ssh_host,
                f"{_env_prefix(_venv_env(exp, node, launch))} ray start "
                f"--address {head.address}:{port} --node-ip-address {node.address} "
                f"--num-gpus {node.gpu_count} --disable-usage-stats",
            )
        )
    gpus = exp.parallelism.world_size
    check = (
        "import ray; ray.init(address='auto'); "
        "n = int(ray.cluster_resources().get('GPU', 0)); print(f'ray_gpus={n}'); "
        f"raise SystemExit(0 if n >= {gpus} else 1)"
    )
    steps.append(
        Step(
            Phase.CLUSTER,
            f"verify ray sees {gpus} GPUs",
            head.ssh_host,
            f"{_env_prefix(_venv_env(exp, head, head_launch))} python -c {shlex.quote(check)}",
        )
    )
    return steps


def _teardown(exp: Experiment, layout: _Layout) -> list[Step]:
    steps = [
        Step(s.phase, s.name, s.host, s.script, check=False) for s in _kill_steps(exp, layout, Phase.TEARDOWN)
    ]
    for node in exp.nodes:
        launch = node.launch(exp.launch)
        if isinstance(launch, DockerLaunch) and launch.container_user is ContainerUser.ROOT:
            # Root containers leave root-owned files in the HF cache and the
            # run dir; hand them back to the ssh user.
            steps.append(
                Step(
                    Phase.TEARDOWN,
                    f"chown container outputs on {node.name}",
                    node.ssh_host,
                    f'docker run --rm -v "{exp.lab.hf_home}:/hf" -v "{exp.lab.remote_root}:/lab" '
                    f'--entrypoint chown {launch.image} -R "$(id -u):$(id -g)" /hf /lab',
                    check=False,
                )
            )
    return steps


def _collect(host: str, layout: _Layout, remote_name: str, local_name: str) -> Step:
    return Step(
        Phase.COLLECT,
        f"collect {remote_name}",
        None,
        f"scp -q {host}:{layout.run_rel}/{remote_name} {shlex.quote(str(layout.local / local_name))}",
    )


def _container_alive(layout: _Layout) -> str:
    return f'if [ -n "$(docker ps -q --filter name=^{layout.container}$)" ]; then a=1; else a=0; fi'


def _process_alive(pattern: str) -> str:
    return f"if pgrep -f '{pattern}' >/dev/null; then a=1; else a=0; fi"


def _nohup(env: dict[str, str], layout: _Layout, command: str, log: str) -> str:
    return (
        f'cd "{layout.run_dir}" && nohup {_env_prefix(env)} {command} '
        f'> "{layout.run_dir}/{log}" 2>&1 < /dev/null &'
    )


# ---------------------------------------------------------------------------
# Static batch
# ---------------------------------------------------------------------------


def _static_bench_args(exp: Experiment, wl: StaticBatchWorkload, output: str) -> list[str]:
    args = [
        "--model", exp.model.hf_id,
        "--shapes", ",".join(s.label for s in wl.shapes),
        "--decode", str(wl.decode_tokens),
        "--warmup", str(wl.warmup),
        "--iters", str(wl.iters),
        "--tp", str(exp.parallelism.tp),
        "--pp", str(exp.parallelism.pp),
        "--dtype", exp.model.dtype,
        "--gpu-memory-utilization", f"{exp.engine.gpu_memory_utilization:g}",
        "--max-model-len", str(exp.model.max_model_len),
        "--max-num-batched-tokens", str(exp.engine.max_num_batched_tokens),
        "--max-num-seqs", str(exp.engine.max_num_seqs),
        "--seed", str(exp.engine.seed),
        "--output", output,
    ]  # fmt: skip
    if exp.engine.enable_prefix_caching:
        args.append("--enable-prefix-caching")
    if exp.engine.enforce_eager:
        args.append("--enforce-eager")
    if exp.multi_node:
        args += ["--backend", "ray"]
    return args


def _static_actions(
    exp: Experiment, wl: StaticBatchWorkload, layout: _Layout
) -> tuple[list[Action], list[Step]]:
    head = exp.head
    collect: list[Step] = []
    match head.launch(exp.launch):
        case DockerLaunch() as launch:
            script = _docker_run(
                exp,
                launch,
                entrypoint="python3",
                args=[
                    f"{layout.c_bench_dir}/bench_latency.py",
                    *_static_bench_args(exp, wl, f"{layout.c_run_dir}/results.jsonl"),
                ],
                name=layout.container,
            )
            alive = _container_alive(layout)
            collect.append(
                Step(
                    Phase.COLLECT,
                    "dump container log",
                    head.ssh_host,
                    f'docker logs {layout.container} > "{layout.run_dir}/bench.log" 2>&1',
                    check=False,
                )
            )
        case VenvLaunch() as launch:
            args = shlex.join(_static_bench_args(exp, wl, "@OUTPUT@")).replace(
                "@OUTPUT@", f'"{layout.run_dir}/results.jsonl"'
            )
            script = _nohup(
                _venv_env(exp, head, launch),
                layout,
                f'python "{layout.bench_dir}/bench_latency.py" {args}',
                "bench.log",
            )
            alive = _process_alive(f"[b]ench_latency.py.*{layout.run_id}")

    results = f"{layout.run_dir}/results.jsonl"
    # Count JSON result records only: vLLM INFO logs share stdout, so raw
    # line counts are wrong. The done sentinel is written after the last shape.
    probe = (
        f'n=$(grep -c \'^{{.*"record": "{STATIC_RECORD}"\' "{results}" 2>/dev/null); '
        f'd=$(grep -c \'^{{.*"record": "{DONE_RECORD}"\' "{results}" 2>/dev/null); '
        f"{alive}; "
        f'if [ "${{d:-0}}" -ge 1 ] && [ "${{n:-0}}" -ge {len(wl.shapes)} ]; '
        "then c=1; else c=0; fi; "
        'echo "complete=$c progress=${n:-0} alive=$a"'
    )
    actions: list[Action] = [
        Step(Phase.LAUNCH, "start static-batch benchmark", head.ssh_host, script),
        Wait(
            "static-batch results",
            head.ssh_host,
            probe,
            expected=len(wl.shapes),
            timeout_s=exp.runner.run_timeout_s,
        ),
    ]
    collect += [
        _collect(head.ssh_host, layout, "results.jsonl", "measured.jsonl"),
        _collect(head.ssh_host, layout, "bench.log", "bench.log"),
    ]
    return actions, collect


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


def serve_result_name(rate: float) -> str:
    return f"{rate_label(rate)}.json"


def bench_serve_args(exp: Experiment, wl: ServingWorkload, rate: float, result_dir: str) -> list[str]:
    """Arguments after `vllm bench serve`: random dataset, fixed lengths, Poisson arrivals.

    Matches the command validated on node0 on 2026-09-30
    (lab-runs/2026-09-30-serving-baseline/run_serving.sh).
    """
    args = [
        "--backend", "vllm",
        "--base-url", f"http://127.0.0.1:{wl.port}",
        "--model", exp.model.hf_id,
        "--dataset-name", "random",
        "--random-input-len", str(wl.input_len),
        "--random-output-len", str(wl.output_len),
        "--random-range-ratio", "0",
        "--ignore-eos",
        "--num-prompts", str(wl.num_prompts),
        "--request-rate", rate_text(rate),
        "--seed", str(wl.seed),
        "--disable-tqdm",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", serve_result_name(rate),
    ]  # fmt: skip
    if wl.burstiness != 1.0:
        args += ["--burstiness", f"{wl.burstiness:g}"]
    return args


def _serve_args(exp: Experiment, wl: ServingWorkload) -> list[str]:
    args = ["--model", exp.model.hf_id, *_engine_args(exp)]
    if wl.port != 8000:
        args += ["--port", str(wl.port)]
    return args


def _server_step(exp: Experiment, wl: ServingWorkload, layout: _Layout) -> tuple[Step, str, Step | None]:
    """(launch step, alive probe expression, optional log-dump step)."""
    head = exp.head
    match head.launch(exp.launch):
        case DockerLaunch() as launch:
            # The image entrypoint is `vllm serve`, so arguments start at --model.
            script = _docker_run(
                exp, launch, entrypoint=None, args=_serve_args(exp, wl),
                name=layout.container, publish_port=wl.port,
            )  # fmt: skip
            dump = Step(
                Phase.COLLECT,
                "dump server log",
                head.ssh_host,
                f'docker logs {layout.container} > "{layout.run_dir}/server.log" 2>&1',
                check=False,
            )
            return (
                Step(Phase.LAUNCH, "start vllm serve", head.ssh_host, script),
                _container_alive(layout),
                dump,
            )
        case VenvLaunch() as launch:
            script = _nohup(
                _venv_env(exp, head, launch),
                layout,
                f"vllm serve {shlex.join(_serve_args(exp, wl))}",
                "server.log",
            )
            return (
                Step(Phase.LAUNCH, "start vllm serve", head.ssh_host, script),
                _process_alive("[v]llm serve"),
                None,
            )


def _client_step(exp: Experiment, wl: ServingWorkload, layout: _Layout, rate: float) -> Step:
    """Bench client from the head node's venv when it has one, else inside the server container."""
    head = exp.head
    log = f"bench_{rate_label(rate)}.log"
    venv = next((launch for launch in head.launches.values() if isinstance(launch, VenvLaunch)), None)
    if venv is not None:
        args = shlex.join(bench_serve_args(exp, wl, rate, "@DIR@")).replace("@DIR@", f'"{layout.run_dir}"')
        script = _nohup(_venv_env(exp, head, venv), layout, f"vllm bench serve {args}", log)
    else:
        inner = (
            "vllm bench serve "
            + shlex.join(bench_serve_args(exp, wl, rate, layout.c_run_dir))
            + f" > {layout.c_run_dir}/{log} 2>&1"
        )
        script = f"docker exec -d {layout.container} sh -c {shlex.quote(inner)}"
    return Step(Phase.LAUNCH, f"bench serve rate={rate_text(rate)}", head.ssh_host, script)


def _serving_actions(
    exp: Experiment, wl: ServingWorkload, layout: _Layout
) -> tuple[list[Action], list[Step]]:
    head = exp.head
    server, alive, dump = _server_step(exp, wl, layout)
    actions: list[Action] = [
        server,
        Wait(
            "server ready",
            head.ssh_host,
            f"if curl -sf http://127.0.0.1:{wl.port}/health >/dev/null; then c=1; else c=0; fi; "
            f'{alive}; echo "complete=$c progress=$c alive=$a"',
            expected=1,
            timeout_s=wl.ready_timeout_s,
        ),
    ]
    collect: list[Step] = [dump] if dump else []
    for rate in wl.request_rates:
        name = serve_result_name(rate)
        actions.append(_client_step(exp, wl, layout, rate))
        # vLLM writes the result JSON once, at the end of the run.
        actions.append(
            Wait(
                f"bench serve rate={rate_text(rate)}",
                head.ssh_host,
                f'if [ -s "{layout.run_dir}/{name}" ]; then c=1; else c=0; fi; '
                f"{_process_alive(f'[v]llm bench serve.*{name}')}; "
                'echo "complete=$c progress=$c alive=$a"',
                expected=1,
                timeout_s=exp.runner.run_timeout_s,
            )
        )
        collect.append(_collect(head.ssh_host, layout, name, name))
        collect.append(
            _collect(head.ssh_host, layout, f"bench_{rate_label(rate)}.log", f"bench_{rate_label(rate)}.log")
        )
    collect.append(_collect(head.ssh_host, layout, "server.log", "server.log"))
    return actions, collect


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_id_for(exp: Experiment | CollectiveExperiment, date: str) -> str:
    return f"{date}-{exp.name}"


def build_plan(exp: Experiment, *, date: str, runs_root: Path, remote_src: Path) -> Plan:
    run_id = run_id_for(exp, date)
    layout = _Layout(exp.lab.remote_root, run_id, runs_root / run_id)
    actions: list[Action] = []
    actions += _sync_steps(exp, layout, remote_src)
    actions += _kill_steps(exp, layout, Phase.CLEAN)
    actions += _probe_steps(exp, layout)
    actions += _ray_steps(exp)
    match exp.workload:
        case StaticBatchWorkload() as wl:
            mode_actions, collect = _static_actions(exp, wl, layout)
        case ServingWorkload() as wl:
            mode_actions, collect = _serving_actions(exp, wl, layout)
    actions += mode_actions
    actions += collect
    return Plan(
        run_id=run_id,
        remote_run_dir=layout.run_dir,
        local_run_dir=layout.local,
        actions=tuple(actions),
        teardown=tuple(_teardown(exp, layout)),
    )
