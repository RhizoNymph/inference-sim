"""Typed experiment and lab specs, parsed from TOML.

Two files describe an experiment:

* a **lab spec** (`tools/lab/labs/*.toml`) lists nodes and every site quirk:
  how to reach each node, how vLLM is launched there (docker image + runtime
  flags, or a native venv), the HF_HOME override, the NCCL/GLOO socket
  interface, the Ray port, and the simulator cluster TOML describing the same
  hardware;
* an **experiment spec** (`tools/lab/specs/*.toml`) names a lab, the nodes to
  use, the model (HF id plus the simulator's model shape), parallelism, engine
  settings, the vLLM version pin, and exactly one workload section
  (`[static_batch]` or `[serving]`) matching `experiment.mode`.

Parsing resolves the lab reference and node names, picks the launch method,
and validates cross-field invariants, so everything downstream receives a
fully-resolved, internally consistent `Experiment`.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from labharness.errors import SpecError

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class Mode(StrEnum):
    STATIC_BATCH = "static-batch"
    SERVING = "serving"
    COLLECTIVE_BENCH = "collective-bench"


class CollectiveOp(StrEnum):
    """Operations `remote/collective_bench.py` can time.

    `send_recv` expands to one directed `send_{src}to{dst}` op per ordered
    rank pair.
    """

    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    SEND_RECV = "send_recv"


class LaunchKind(StrEnum):
    DOCKER = "docker"
    VENV = "venv"


class ContainerUser(StrEnum):
    """Who processes inside the container run as.

    `user` passes `--user $(id -u):$(id -g)` with HOME and the vLLM cache in
    /tmp, so nothing root-owned lands in the host HF cache. `root` is the
    image default and requires a chown pass during teardown.
    """

    USER = "user"
    ROOT = "root"


@dataclass(frozen=True, slots=True)
class DockerLaunch:
    """Run vLLM inside a container on the node."""

    image: str
    runtime_args: tuple[str, ...]
    container_hf_home: str
    container_user: ContainerUser

    @property
    def kind(self) -> LaunchKind:
        return LaunchKind.DOCKER


@dataclass(frozen=True, slots=True)
class VenvLaunch:
    """Run vLLM from a native virtualenv on the node."""

    venv: str

    @property
    def kind(self) -> LaunchKind:
        return LaunchKind.VENV

    def bin(self, name: str) -> str:
        return f"{self.venv}/bin/{name}"


type Launch = DockerLaunch | VenvLaunch


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    ssh_host: str
    address: str
    sim_node_id: int
    gpu_count: int
    launches: Mapping[LaunchKind, Launch]
    default_launch: LaunchKind
    notes: str

    def launch(self, kind: LaunchKind) -> Launch:
        return self.launches[kind]


@dataclass(frozen=True, slots=True)
class Lab:
    name: str
    hardware: str
    remote_root: str
    hf_home: str
    socket_ifname: str
    ray_port: int
    cluster_toml: Path
    extra_env: Mapping[str, str]
    nodes: tuple[Node, ...]
    path: Path

    def node(self, name: str) -> Node:
        for node in self.nodes:
            if node.name == name:
                return node
        raise SpecError(
            f"lab {self.name!r} has no node named {name!r}; known: {[n.name for n in self.nodes]}",
            path=str(self.path),
            field="nodes",
        )


@dataclass(frozen=True, slots=True)
class SimModel:
    """The simulator's `[model]` section for this model."""

    layers: int
    hidden_size: int
    attention_heads: int
    kv_heads: int
    vocab_size: int
    parameters_gb: float
    dtype: str
    kv_dtype: str | None
    ffn_hidden_size: int | None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.attention_heads


@dataclass(frozen=True, slots=True)
class ModelSpec:
    hf_id: str
    name: str
    dtype: str
    max_model_len: int
    sim: SimModel


@dataclass(frozen=True, slots=True)
class Parallelism:
    tp: int
    pp: int

    @property
    def world_size(self) -> int:
        return self.tp * self.pp


@dataclass(frozen=True, slots=True)
class Engine:
    """vLLM engine arguments shared by both modes."""

    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    enable_prefix_caching: bool
    enforce_eager: bool
    seed: int


@dataclass(frozen=True, slots=True, order=True)
class Shape:
    batch: int
    prompt: int

    @property
    def label(self) -> str:
        return f"{self.batch}x{self.prompt}"

    @classmethod
    def parse(cls, text: str, *, path: str) -> Shape:
        match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", text)
        if match is None:
            raise SpecError(
                f"shape {text!r} must look like BATCHxPROMPT, e.g. 8x512",
                path=path,
                field="static_batch.shapes",
            )
        batch, prompt = int(match.group(1)), int(match.group(2))
        if batch <= 0 or prompt <= 0:
            raise SpecError(f"shape {text!r} must be positive", path=path, field="static_batch.shapes")
        return cls(batch=batch, prompt=prompt)


@dataclass(frozen=True, slots=True)
class StaticBatchWorkload:
    shapes: tuple[Shape, ...]
    decode_tokens: int
    warmup: int
    iters: int

    @property
    def mode(self) -> Mode:
        return Mode.STATIC_BATCH


@dataclass(frozen=True, slots=True)
class LittlesLawBatch:
    """Serving-sim reference batch = steady-state concurrency from Little's law."""


@dataclass(frozen=True, slots=True)
class FixedBatch:
    batch: int


type ReferenceBatch = LittlesLawBatch | FixedBatch


class SimAdmission(StrEnum):
    """Admission limits given to the serving simulation.

    `engine` mirrors vLLM: max_num_seqs concurrent sequences and the measured
    KV capacity. Like vLLM, the simulator's iteration engine queues requests
    past those limits (it reserves KV for a whole sequence at admission and
    does not preempt). `uncapped` lifts both limits to the request count, a
    what-if with no admission pressure.
    """

    ENGINE = "engine"
    UNCAPPED = "uncapped"


@dataclass(frozen=True, slots=True)
class ServingWorkload:
    input_len: int
    output_len: int
    request_rates: tuple[float, ...]
    num_prompts: int
    seed: int
    burstiness: float
    port: int
    ready_timeout_s: float
    sim_request_count: int | None
    sim_reference_batch: ReferenceBatch
    sim_admission: SimAdmission
    kv_cache_tokens: int | None

    @property
    def mode(self) -> Mode:
        return Mode.SERVING


def rate_text(rate: float) -> str:
    """`2.0` -> `2`, `inf` -> `inf`: the form vLLM's --request-rate accepts."""
    return "inf" if math.isinf(rate) else f"{rate:g}"


def rate_label(rate: float) -> str:
    return f"rate_{rate_text(rate)}"


type Workload = StaticBatchWorkload | ServingWorkload


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    poll_interval_s: float
    run_timeout_s: float


@dataclass(frozen=True, slots=True)
class CollectiveBenchWorkload:
    """NCCL message-size sweep: sizes double from min_bytes up to max_bytes."""

    ops: tuple[CollectiveOp, ...]
    min_bytes: int
    max_bytes: int
    iters: int
    warmup: int
    master_port: int

    @property
    def mode(self) -> Mode:
        return Mode.COLLECTIVE_BENCH

    def message_sizes(self) -> tuple[int, ...]:
        sizes: list[int] = []
        size = self.min_bytes
        while size <= self.max_bytes:
            sizes.append(size)
            size *= 2
        return tuple(sizes)

    def ops_per_size(self, world_size: int) -> int:
        """Result rows per message size: one per collective, one per ordered rank pair."""
        return sum(world_size * (world_size - 1) if op is CollectiveOp.SEND_RECV else 1 for op in self.ops)

    def expected_rows(self, world_size: int) -> int:
        return len(self.message_sizes()) * self.ops_per_size(world_size)


@dataclass(frozen=True, slots=True)
class Experiment:
    """A vLLM experiment (static-batch or serving)."""

    name: str
    description: str
    vllm_version: str
    lab: Lab
    nodes: tuple[Node, ...]
    launch: LaunchKind
    model: ModelSpec
    parallelism: Parallelism
    engine: Engine
    workload: Workload
    runner: RunnerSettings
    path: Path
    source_text: str

    @property
    def mode(self) -> Mode:
        return self.workload.mode

    @property
    def head(self) -> Node:
        return self.nodes[0]

    @property
    def multi_node(self) -> bool:
        return len(self.nodes) > 1


@dataclass(frozen=True, slots=True)
class GpuRank:
    """One torch.distributed process: global rank, its node, and its GPU on that node."""

    rank: int
    node: Node
    local_rank: int


@dataclass(frozen=True, slots=True)
class CollectiveExperiment:
    """A NCCL collective benchmark: no model, engine, or vLLM pin; one process per GPU."""

    name: str
    description: str
    lab: Lab
    nodes: tuple[Node, ...]
    launch: LaunchKind
    workload: CollectiveBenchWorkload
    runner: RunnerSettings
    path: Path
    source_text: str

    @property
    def mode(self) -> Mode:
        return self.workload.mode

    @property
    def head(self) -> Node:
        return self.nodes[0]

    @property
    def multi_node(self) -> bool:
        return len(self.nodes) > 1

    @property
    def world_size(self) -> int:
        return sum(node.gpu_count for node in self.nodes)

    def ranks(self) -> tuple[GpuRank, ...]:
        """Ranks in node order, then local GPU index (rank 0 is the head's first GPU)."""
        ranks: list[GpuRank] = []
        for node in self.nodes:
            for local in range(node.gpu_count):
                ranks.append(GpuRank(rank=len(ranks), node=node, local_rank=local))
        return tuple(ranks)


type Spec = Experiment | CollectiveExperiment


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------

_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class _Table:
    """Typed accessors over one TOML table that report the dotted field path."""

    def __init__(self, data: Mapping[str, object], *, path: str, prefix: str) -> None:
        self._data = data
        self._path = path
        self._prefix = prefix

    def _field(self, key: str) -> str:
        return f"{self._prefix}.{key}" if self._prefix else key

    def _error(self, key: str, message: str) -> SpecError:
        return SpecError(f"{self._field(key)}: {message}", path=self._path, field=self._field(key))

    def has(self, key: str) -> bool:
        return key in self._data

    def sub(self, key: str) -> _Table:
        value = self._data.get(key)
        if not isinstance(value, dict):
            raise self._error(key, "missing table")
        return _Table(value, path=self._path, prefix=self._field(key))

    def opt_sub(self, key: str) -> _Table | None:
        return self.sub(key) if key in self._data else None

    def tables(self, key: str) -> list[_Table]:
        value = self._data.get(key)
        if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
            raise self._error(key, "must be an array of tables")
        return [_Table(v, path=self._path, prefix=f"{self._field(key)}[{i}]") for i, v in enumerate(value)]

    def string(self, key: str, default: str | None = None) -> str:
        value = self._data.get(key, default)
        if not isinstance(value, str) or not value:
            raise self._error(key, "must be a non-empty string")
        return value

    def opt_string(self, key: str) -> str | None:
        return self.string(key) if key in self._data else None

    def integer(self, key: str, default: int | None = None, *, minimum: int = 1) -> int:
        value = self._data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._error(key, "must be an integer")
        if value < minimum:
            raise self._error(key, f"must be >= {minimum}")
        return value

    def opt_integer(self, key: str, *, minimum: int = 1) -> int | None:
        return self.integer(key, minimum=minimum) if key in self._data else None

    def number(self, key: str, default: float | None = None, *, positive: bool = True) -> float:
        value = self._data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._error(key, "must be a number")
        if positive and not value > 0:
            raise self._error(key, "must be > 0")
        return float(value)

    def boolean(self, key: str, default: bool | None = None) -> bool:
        value = self._data.get(key, default)
        if not isinstance(value, bool):
            raise self._error(key, "must be true or false")
        return value

    def strings(self, key: str, default: list[str] | None = None) -> tuple[str, ...]:
        value = self._data.get(key, default)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise self._error(key, "must be an array of strings")
        return tuple(value)

    def rates(self, key: str) -> tuple[float, ...]:
        """Positive numbers, or the string "inf" (send every request at t=0)."""
        value = self._data.get(key)
        if not isinstance(value, list) or not value:
            raise self._error(key, 'must be a non-empty array of numbers or "inf"')
        rates: list[float] = []
        for item in value:
            match item:
                case bool():
                    raise self._error(key, 'values must be numbers or "inf"')
                case int() | float() if item > 0 and math.isfinite(item):
                    rates.append(float(item))
                case "inf":
                    rates.append(math.inf)
                case _:
                    raise self._error(key, f'invalid rate {item!r}; use a positive number or "inf"')
        return tuple(rates)

    def string_map(self, key: str) -> dict[str, str]:
        value = self._data.get(key, {})
        if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
            raise self._error(key, "must be a table of strings")
        return dict(value)

    def enum[E: StrEnum](self, key: str, enum: type[E], default: str | None = None) -> E:
        text = self.string(key, default)
        try:
            return enum(text)
        except ValueError as error:
            raise self._error(key, f"must be one of {[e.value for e in enum]}") from error

    def name(self, key: str) -> str:
        text = self.string(key)
        if _NAME_RE.fullmatch(text) is None:
            raise self._error(key, "must match [A-Za-z0-9][A-Za-z0-9._-]*")
        return text


def _load_toml(path: Path) -> tuple[dict[str, object], str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SpecError(f"cannot read spec: {error}", path=str(path)) from error
    try:
        return tomllib.loads(text), text
    except tomllib.TOMLDecodeError as error:
        raise SpecError(f"invalid TOML: {error}", path=str(path)) from error


# ---------------------------------------------------------------------------
# Lab spec
# ---------------------------------------------------------------------------


def _parse_launches(node: _Table) -> dict[LaunchKind, Launch]:
    launches: dict[LaunchKind, Launch] = {}
    if (docker := node.opt_sub("docker")) is not None:
        launches[LaunchKind.DOCKER] = DockerLaunch(
            image=docker.string("image"),
            runtime_args=docker.strings("runtime_args", []),
            container_hf_home=docker.string("container_hf_home", "/hf"),
            container_user=docker.enum("container_user", ContainerUser, "user"),
        )
    if (venv := node.opt_sub("venv")) is not None:
        launches[LaunchKind.VENV] = VenvLaunch(venv=venv.string("path"))
    return launches


def parse_lab(path: Path) -> Lab:
    data, _ = _load_toml(path)
    root = _Table(data, path=str(path), prefix="")
    lab = root.sub("lab")
    nodes: list[Node] = []
    for table in root.tables("nodes"):
        launches = _parse_launches(table)
        if not launches:
            raise SpecError(
                "node needs a [nodes.docker] and/or [nodes.venv] launch table",
                path=str(path),
                field=f"{table._prefix}",
            )
        default = table.enum("default_launch", LaunchKind, next(iter(launches)).value)
        if default not in launches:
            raise SpecError(
                f"default_launch {default.value!r} has no matching launch table",
                path=str(path),
                field=f"{table._prefix}.default_launch",
            )
        nodes.append(
            Node(
                name=table.name("name"),
                ssh_host=table.string("ssh_host", table.string("name")),
                address=table.string("address"),
                sim_node_id=table.integer("sim_node_id", minimum=0),
                gpu_count=table.integer("gpu_count", 1),
                launches=launches,
                default_launch=default,
                notes=table.string("notes", "-"),
            )
        )
    names = [n.name for n in nodes]
    if len(set(names)) != len(names):
        raise SpecError(f"duplicate node names: {names}", path=str(path), field="nodes")
    cluster = (path.parent / lab.string("cluster_toml")).resolve()
    if not cluster.is_file():
        raise SpecError(f"cluster_toml {cluster} does not exist", path=str(path), field="lab.cluster_toml")
    return Lab(
        name=lab.name("name"),
        hardware=lab.string("hardware"),
        remote_root=lab.string("remote_root"),
        hf_home=lab.string("hf_home"),
        socket_ifname=lab.string("socket_ifname"),
        ray_port=lab.integer("ray_port", 6379),
        cluster_toml=cluster,
        extra_env=lab.string_map("env"),
        nodes=tuple(nodes),
        path=path.resolve(),
    )


# ---------------------------------------------------------------------------
# Experiment spec
# ---------------------------------------------------------------------------


def _parse_model(root: _Table) -> ModelSpec:
    model = root.sub("model")
    sim = model.sub("sim")
    sim_model = SimModel(
        layers=sim.integer("layers"),
        hidden_size=sim.integer("hidden_size"),
        attention_heads=sim.integer("attention_heads"),
        kv_heads=sim.integer("kv_heads"),
        vocab_size=sim.integer("vocab_size"),
        parameters_gb=sim.number("parameters_gb"),
        dtype=sim.string("dtype", "bf16"),
        kv_dtype=sim.opt_string("kv_dtype"),
        ffn_hidden_size=sim.opt_integer("ffn_hidden_size"),
    )
    if sim_model.hidden_size % sim_model.attention_heads != 0:
        raise SpecError(
            "hidden_size must be divisible by attention_heads",
            path=model._path,
            field="model.sim.hidden_size",
        )
    return ModelSpec(
        hf_id=model.string("hf_id"),
        name=model.name("name"),
        dtype=model.string("dtype", "bfloat16"),
        max_model_len=model.integer("max_model_len", 4096),
        sim=sim_model,
    )


def _parse_engine(root: _Table) -> Engine:
    engine = root.sub("engine")
    utilization = engine.number("gpu_memory_utilization", 0.85)
    if utilization > 1.0:
        raise SpecError(
            "gpu_memory_utilization must be <= 1.0",
            path=engine._path,
            field="engine.gpu_memory_utilization",
        )
    return Engine(
        gpu_memory_utilization=utilization,
        max_num_batched_tokens=engine.integer("max_num_batched_tokens", 16384),
        max_num_seqs=engine.integer("max_num_seqs", 64),
        enable_prefix_caching=engine.boolean("enable_prefix_caching", False),
        enforce_eager=engine.boolean("enforce_eager", False),
        seed=engine.integer("seed", 0, minimum=0),
    )


def _parse_static(table: _Table, model: ModelSpec, engine: Engine) -> StaticBatchWorkload:
    shapes = tuple(Shape.parse(s, path=table._path) for s in table.strings("shapes"))
    if not shapes:
        raise SpecError("at least one shape required", path=table._path, field="static_batch.shapes")
    if len(set(shapes)) != len(shapes):
        raise SpecError("duplicate shapes", path=table._path, field="static_batch.shapes")
    decode = table.integer("decode_tokens", 128)
    for shape in shapes:
        if shape.prompt + decode + 1 > model.max_model_len:
            raise SpecError(
                f"shape {shape.label} + {decode} decode tokens exceeds max_model_len {model.max_model_len}",
                path=table._path,
                field="static_batch.shapes",
            )
        if shape.batch > engine.max_num_seqs:
            raise SpecError(
                f"shape {shape.label} batch exceeds engine.max_num_seqs {engine.max_num_seqs}; "
                "vLLM would split it and the static batch would not be static",
                path=table._path,
                field="static_batch.shapes",
            )
        if shape.batch * shape.prompt > engine.max_num_batched_tokens:
            raise SpecError(
                f"shape {shape.label} has {shape.batch * shape.prompt} prompt tokens, more than "
                f"engine.max_num_batched_tokens {engine.max_num_batched_tokens}; the prefill "
                "would be chunked across steps",
                path=table._path,
                field="static_batch.shapes",
            )
    return StaticBatchWorkload(
        shapes=shapes,
        decode_tokens=decode,
        warmup=table.integer("warmup", 2, minimum=0),
        iters=table.integer("iters", 5),
    )


def _parse_serving(table: _Table, model: ModelSpec) -> ServingWorkload:
    input_len = table.integer("input_len")
    output_len = table.integer("output_len")
    if input_len + output_len > model.max_model_len:
        raise SpecError(
            "input_len + output_len exceeds model.max_model_len",
            path=table._path,
            field="serving.input_len",
        )
    rates = table.rates("request_rates")
    if len(set(rates)) != len(rates):
        raise SpecError("duplicate request rates", path=table._path, field="serving.request_rates")
    return ServingWorkload(
        input_len=input_len,
        output_len=output_len,
        request_rates=rates,
        num_prompts=table.integer("num_prompts", 200),
        seed=table.integer("seed", 0, minimum=0),
        burstiness=table.number("burstiness", 1.0),
        port=table.integer("port", 8000),
        ready_timeout_s=table.number("ready_timeout_s", 900.0),
        sim_request_count=table.opt_integer("sim_request_count"),
        sim_reference_batch=_parse_reference_batch(table),
        sim_admission=table.enum("sim_admission", SimAdmission, "engine"),
        kv_cache_tokens=table.opt_integer("kv_cache_tokens"),
    )


def _parse_reference_batch(table: _Table) -> ReferenceBatch:
    match table._data.get("sim_reference_batch", "littles_law"):
        case "littles_law":
            return LittlesLawBatch()
        case int() as batch if not isinstance(batch, bool) and batch >= 1:
            return FixedBatch(batch)
        case other:
            raise SpecError(
                f'sim_reference_batch must be "littles_law" or a positive integer, got {other!r}',
                path=table._path,
                field="serving.sim_reference_batch",
            )


def _resolve_launch(requested: str, nodes: tuple[Node, ...], path: str) -> LaunchKind:
    kinds: set[LaunchKind] = set()
    for node in nodes:
        kind = node.default_launch if requested == "auto" else LaunchKind(requested)
        if kind not in node.launches:
            raise SpecError(
                f"node {node.name} has no {kind.value} launch configured",
                path=path,
                field="experiment.launch",
            )
        kinds.add(kind)
    if len(kinds) != 1:
        raise SpecError(
            f"nodes resolve to mixed launch methods {sorted(k.value for k in kinds)}; "
            "set experiment.launch explicitly",
            path=path,
            field="experiment.launch",
        )
    kind = kinds.pop()
    if len(nodes) > 1 and kind is LaunchKind.DOCKER:
        raise SpecError(
            "multi-node experiments run Ray from native venvs; set experiment.launch = 'venv'",
            path=path,
            field="experiment.launch",
        )
    return kind


_VLLM_ONLY_SECTIONS: Final = ("model", "parallelism", "engine", "static_batch", "serving")
_COLLECTIVE_MIN_BYTES: Final = 16


def _parse_collective_workload(table: _Table) -> CollectiveBenchWorkload:
    ops_text = table.strings("ops", [op.value for op in CollectiveOp])
    if not ops_text:
        raise SpecError("at least one op required", path=table._path, field="collective_bench.ops")
    try:
        ops = tuple(CollectiveOp(text) for text in ops_text)
    except ValueError as error:
        raise SpecError(
            f"collective_bench.ops: values must be in {[op.value for op in CollectiveOp]}",
            path=table._path,
            field="collective_bench.ops",
        ) from error
    if len(set(ops)) != len(ops):
        raise SpecError("duplicate ops", path=table._path, field="collective_bench.ops")
    min_bytes = table.integer("min_bytes", 1024, minimum=_COLLECTIVE_MIN_BYTES)
    max_bytes = table.integer("max_bytes", 256 * 1024 * 1024, minimum=_COLLECTIVE_MIN_BYTES)
    if max_bytes < min_bytes:
        raise SpecError(
            "collective_bench.max_bytes must be >= min_bytes",
            path=table._path,
            field="collective_bench.max_bytes",
        )
    return CollectiveBenchWorkload(
        ops=ops,
        min_bytes=min_bytes,
        max_bytes=max_bytes,
        iters=table.integer("iters", 30),
        warmup=table.integer("warmup", 5, minimum=0),
        master_port=table.integer("master_port", 29500),
    )


def _parse_collective(
    root: _Table, experiment: _Table, header: _Header, runner: RunnerSettings
) -> CollectiveExperiment:
    spath = header.path_text
    for section in _VLLM_ONLY_SECTIONS:
        if root.has(section):
            raise SpecError(
                f"collective-bench specs take no [{section}] section",
                path=spath,
                field=section,
            )
    if experiment.has("vllm_version"):
        raise SpecError(
            "collective-bench specs take no vllm_version (the probe checks for torch instead)",
            path=spath,
            field="experiment.vllm_version",
        )
    if header.launch is not LaunchKind.VENV:
        raise SpecError(
            "collective-bench runs torch.distributed from native venvs; set experiment.launch = 'venv'",
            path=spath,
            field="experiment.launch",
        )
    if not root.has("collective_bench"):
        raise SpecError(
            "mode 'collective-bench' requires a [collective_bench] section",
            path=spath,
            field="experiment.mode",
        )
    workload = _parse_collective_workload(root.sub("collective_bench"))
    exp = CollectiveExperiment(
        name=header.name,
        description=header.description,
        lab=header.lab,
        nodes=header.nodes,
        launch=header.launch,
        workload=workload,
        runner=runner,
        path=header.path.resolve(),
        source_text=header.text,
    )
    if exp.world_size < 2:
        raise SpecError(
            f"collective-bench needs at least 2 GPUs; the listed nodes have {exp.world_size}",
            path=spath,
            field="experiment.nodes",
        )
    return exp


def _parse_vllm(root: _Table, experiment: _Table, header: _Header, runner: RunnerSettings) -> Experiment:
    spath = header.path_text
    mode = header.mode
    nodes = header.nodes
    model = _parse_model(root)
    par = root.sub("parallelism")
    parallelism = Parallelism(tp=par.integer("tp", 1), pp=par.integer("pp", 1))
    gpus = sum(node.gpu_count for node in nodes)
    if parallelism.world_size != gpus:
        raise SpecError(
            f"tp*pp = {parallelism.world_size} but the listed nodes have {gpus} GPUs",
            path=spath,
            field="parallelism",
        )
    engine = _parse_engine(root)

    has_static, has_serving = root.has("static_batch"), root.has("serving")
    match mode:
        case Mode.STATIC_BATCH if has_static and not has_serving:
            workload: Workload = _parse_static(root.sub("static_batch"), model, engine)
        case Mode.SERVING if has_serving and not has_static:
            workload = _parse_serving(root.sub("serving"), model)
        case _:
            raise SpecError(
                f"mode {mode.value!r} requires exactly one matching workload section "
                "([static_batch] or [serving])",
                path=spath,
                field="experiment.mode",
            )

    return Experiment(
        name=header.name,
        description=header.description,
        vllm_version=experiment.string("vllm_version"),
        lab=header.lab,
        nodes=nodes,
        launch=header.launch,
        model=model,
        parallelism=parallelism,
        engine=engine,
        workload=workload,
        runner=runner,
        path=header.path.resolve(),
        source_text=header.text,
    )


@dataclass(frozen=True, slots=True)
class _Header:
    """`[experiment]` fields shared by every mode, already resolved against the lab."""

    path: Path
    text: str
    mode: Mode
    name: str
    description: str
    lab: Lab
    nodes: tuple[Node, ...]
    launch: LaunchKind

    @property
    def path_text(self) -> str:
        return str(self.path)


def parse_spec(path: Path) -> Spec:
    """Parse any experiment spec: a vLLM `Experiment` or a `CollectiveExperiment`."""
    data, text = _load_toml(path)
    spath = str(path)
    root = _Table(data, path=spath, prefix="")
    experiment = root.sub("experiment")
    mode = experiment.enum("mode", Mode)
    lab = parse_lab((path.parent / experiment.string("lab")).resolve())
    node_names = experiment.strings("nodes")
    if not node_names:
        raise SpecError("at least one node required", path=spath, field="experiment.nodes")
    if len(set(node_names)) != len(node_names):
        raise SpecError("duplicate nodes", path=spath, field="experiment.nodes")
    nodes = tuple(lab.node(name) for name in node_names)
    launch_text = experiment.string("launch", "auto")
    if launch_text != "auto" and launch_text not in {k.value for k in LaunchKind}:
        raise SpecError("launch must be auto, docker, or venv", path=spath, field="experiment.launch")
    launch = _resolve_launch(launch_text, nodes, spath)
    header = _Header(
        path=path,
        text=text,
        mode=mode,
        name=experiment.name("name"),
        description=experiment.string("description", "-"),
        lab=lab,
        nodes=nodes,
        launch=launch,
    )
    runner_table = root.opt_sub("runner") or _Table({}, path=spath, prefix="runner")
    runner = RunnerSettings(
        poll_interval_s=runner_table.number("poll_interval_s", 15.0),
        run_timeout_s=runner_table.number("run_timeout_s", 3600.0),
    )
    match mode:
        case Mode.COLLECTIVE_BENCH:
            return _parse_collective(root, experiment, header, runner)
        case Mode.STATIC_BATCH | Mode.SERVING:
            return _parse_vllm(root, experiment, header, runner)


def parse_experiment(path: Path) -> Experiment:
    """Parse a vLLM experiment spec; `sim`, `calibrate`, and `validate` only accept these."""
    match parse_spec(path):
        case Experiment() as exp:
            return exp
        case CollectiveExperiment():
            raise SpecError(
                "collective-bench specs only work with `lab.py run`; turn their results into "
                "simulator curves with `lab.py curves`",
                path=str(path),
                field="experiment.mode",
            )
