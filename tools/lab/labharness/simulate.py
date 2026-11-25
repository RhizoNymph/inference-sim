"""Run the inference-sim release binary over an experiment spec.

Static batch: one workload per (shape, phase) with phase in prefill / decode /
end_to_end, searched at exactly the experiment's (tp, pp).

Serving: one colocated serving workload per request rate that mirrors the
`vllm bench serve` run: Poisson arrivals at the rate, fixed prompt/output
lengths, one sequence per request, continuous batching with a prefill chunk
budget equal to vLLM's `max_num_batched_tokens`, `max_num_seqs` concurrent
decode sequences, and a KV budget derived the way vLLM sizes its cache.

Simulator runs are independent subprocesses, so they run concurrently under an
asyncio semaphore.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from labharness.errors import SimulatorError
from labharness.results import LatencyStats, number_or_none
from labharness.spec import (
    Experiment,
    FixedBatch,
    LittlesLawBatch,
    ModelSpec,
    Parallelism,
    ServingWorkload,
    Shape,
    SimAdmission,
    rate_label,
    rate_text,
)
from labharness.toml_emit import TomlValue, table

DEFAULT_CONCURRENCY: Final = 8
KV_BLOCK_TOKENS: Final = 16
_DTYPE_BYTES: Final = {"bf16": 2, "fp16": 2, "f16": 2, "bfloat16": 2, "float16": 2, "fp8": 1, "f8": 1}

# ---------------------------------------------------------------------------
# Calibration choice
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DefaultCalibration:
    """No [calibration] section: the simulator's built-in defaults."""

    @property
    def label(self) -> str:
        return "default"


@dataclass(frozen=True, slots=True)
class ScalarCalibration:
    compute_efficiency: float
    decode_memory_bandwidth_scale: float

    @property
    def label(self) -> str:
        return f"ce{self.compute_efficiency:.4f}-dmbs{self.decode_memory_bandwidth_scale:.4f}"


@dataclass(frozen=True, slots=True)
class ProfileCalibration:
    path: Path

    @property
    def label(self) -> str:
        return f"profile-{self.path.stem}"


type Calibration = DefaultCalibration | ScalarCalibration | ProfileCalibration


def _calibration_lines(calibration: Calibration) -> list[str]:
    match calibration:
        case DefaultCalibration():
            return []
        case ScalarCalibration():
            return [
                "",
                *table(
                    "calibration",
                    [
                        ("compute_efficiency", calibration.compute_efficiency),
                        ("decode_memory_bandwidth_scale", calibration.decode_memory_bandwidth_scale),
                    ],
                ),
            ]
        case ProfileCalibration():
            return ["", *table("calibration_profile", [("path", str(calibration.path.resolve()))])]


# ---------------------------------------------------------------------------
# Workload rendering
# ---------------------------------------------------------------------------


class SimPhase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
    END_TO_END = "end_to_end"


def _model_lines(model: ModelSpec) -> list[str]:
    sim = model.sim
    return table(
        "model",
        [
            ("id", model.name),
            ("layers", sim.layers),
            ("hidden_size", sim.hidden_size),
            ("attention_heads", sim.attention_heads),
            ("kv_heads", sim.kv_heads),
            ("vocab_size", sim.vocab_size),
            ("ffn_hidden_size", sim.ffn_hidden_size),
            ("parameters_gb", sim.parameters_gb),
            ("dtype", sim.dtype),
            ("kv_dtype", sim.kv_dtype),
        ],
    )


def _search_lines(name: str, par: Parallelism) -> list[str]:
    return table(
        name,
        [
            ("tensor_ranks", [par.tp]),
            ("pipeline_ranks", [par.pp]),
            ("expert_ranks", [1]),
            ("data_ranks", [1]),
        ],
    )


def static_workload_toml(
    model: ModelSpec,
    par: Parallelism,
    shape: Shape,
    decode_tokens: int,
    phase: SimPhase,
    calibration: Calibration,
) -> str:
    lines = ["schema_version = 1", "", *_model_lines(model), ""]
    lines += table(
        "request",
        [
            ("batch_size", shape.batch),
            ("prompt_tokens", shape.prompt),
            ("decode_tokens", decode_tokens),
            # The real sequence length (not max_model_len) keeps the request
            # inside a profile's valid_shape; latency does not depend on it.
            ("max_sequence_tokens", shape.prompt + decode_tokens),
            ("phase", phase.value),
        ],
    )
    lines += _calibration_lines(calibration)
    lines += ["", *_search_lines("search", par)]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class KvBudget:
    tokens: int
    blocks: int


def kv_budget(exp: Experiment, hbm_gb_per_gpu: float, measured_tokens: int | None = None) -> KvBudget:
    """KV capacity for the serving workload.

    Prefer `measured_tokens` (vLLM's logged "GPU KV cache size: N tokens",
    via the spec's `kv_cache_tokens` or the run's server.log). Otherwise
    estimate it the way vLLM sizes it: utilization * HBM minus weights, which
    ignores activation/CUDA-graph workspace and so over-estimates by ~10%.
    """
    if measured_tokens is not None:
        blocks = measured_tokens // KV_BLOCK_TOKENS
        return KvBudget(tokens=blocks * KV_BLOCK_TOKENS, blocks=blocks)
    sim = exp.model.sim
    kv_bytes = _DTYPE_BYTES.get((sim.kv_dtype or sim.dtype).lower(), 2)
    bytes_per_token = 2 * sim.layers * sim.kv_heads * sim.head_dim * kv_bytes
    world = exp.parallelism.world_size
    free_gb = exp.engine.gpu_memory_utilization * hbm_gb_per_gpu * world - sim.parameters_gb
    tokens = max(0, math.floor(free_gb * 1e9 / bytes_per_token))
    blocks = tokens // KV_BLOCK_TOKENS
    return KvBudget(tokens=blocks * KV_BLOCK_TOKENS, blocks=blocks)


def serving_workload_toml(
    exp: Experiment,
    wl: ServingWorkload,
    *,
    rate: float,
    reference_batch: int,
    request_count: int,
    kv: KvBudget,
    calibration: Calibration,
) -> str:
    nodes = [node.sim_node_id for node in exp.nodes]
    engine = exp.engine
    match wl.sim_admission:
        case SimAdmission.ENGINE:
            decode_cap = engine.max_num_seqs
        case SimAdmission.UNCAPPED:
            decode_cap = max(engine.max_num_seqs, request_count)
            per_request = wl.input_len + wl.output_len
            blocks = max(kv.blocks, math.ceil(request_count * per_request / KV_BLOCK_TOKENS))
            kv = KvBudget(tokens=blocks * KV_BLOCK_TOKENS, blocks=blocks)
    lines = ["schema_version = 1", "", *_model_lines(exp.model), ""]
    # [request] is the serving model's reference shape. This colocated,
    # continuously batched workload runs on the simulator's iteration engine,
    # which prices each step from its actual composition, so batch_size only
    # sizes the static feasibility and memory scores.
    lines += table(
        "request",
        [
            ("batch_size", reference_batch),
            ("prompt_tokens", wl.input_len),
            ("decode_tokens", wl.output_len),
            ("max_sequence_tokens", wl.input_len + wl.output_len),
            ("phase", "end_to_end"),
        ],
    )
    lines += _calibration_lines(calibration)
    lines += ["", *_search_lines("search", exp.parallelism), ""]
    lines += table(
        "serving",
        [
            ("mode", "colocated"),
            ("serving_stack", "vllm"),
            ("objective", "e2el"),
            ("prefill_nodes", nodes),
            ("decode_nodes", nodes),
        ],
    )
    lines += [""]
    # rate = inf is vLLM's "send everything at t=0": a zero fixed gap.
    arrival: list[tuple[str, TomlValue]] = (
        [("arrival", "fixed"), ("arrival_gap_ms", 0.0)]
        if math.isinf(rate)
        else [("arrival", "poisson"), ("arrival_rate_per_s", rate), ("arrival_seed", wl.seed)]
    )
    lines += table(
        "serving.traffic",
        [
            ("request_count", request_count),
            *arrival,
            ("shape_seed", wl.seed),
            ("prefill_batching", "continuous"),
            ("max_prefill_batch_tokens", engine.max_num_batched_tokens),
            ("max_prefill_chunk_tokens", engine.max_num_batched_tokens),
            ("decode_batching", "continuous"),
            ("max_decode_batch_tokens", decode_cap),
            ("max_decode_sequences", decode_cap),
            ("max_resident_tokens", kv.tokens),
            ("kv_block_tokens", KV_BLOCK_TOKENS),
            ("max_kv_blocks", kv.blocks),
            ("prefix_cache_hit_rate", 0.0),
            ("batch_sizes", [1]),
            ("prompt_tokens", [wl.input_len]),
            ("decode_tokens", [wl.output_len]),
        ],
    )
    lines += ["", *_search_lines("serving.prefill_search", exp.parallelism)]
    lines += ["", *_search_lines("serving.decode_search", exp.parallelism)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Binary invocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimRunner:
    binary: Path
    cluster: Path
    work_dir: Path
    concurrency: int = DEFAULT_CONCURRENCY

    async def run(self, name: str, workload_text: str, semaphore: asyncio.Semaphore) -> dict[str, object]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        workload = self.work_dir / f"{name}.toml"
        # Concurrent tasks may render the same workload while a simulator
        # process is reading it: never rewrite identical content, and replace
        # atomically so a reader never sees a truncated file.
        if not workload.is_file() or workload.read_text(encoding="utf-8") != workload_text:
            staging = workload.with_suffix(f".{os.getpid()}.{id(workload_text)}.tmp")
            staging.write_text(workload_text, encoding="utf-8")
            os.replace(staging, workload)
        async with semaphore:
            process = await asyncio.create_subprocess_exec(
                str(self.binary),
                "--cluster",
                str(self.cluster),
                "--workload",
                str(workload),
                "--json",
                "--top-k",
                "1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise SimulatorError(
                f"inference-sim exited with {process.returncode}",
                workload=str(workload),
                stderr_tail=stderr.decode(errors="replace")[-600:],
            )
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise SimulatorError(f"non-JSON output: {error}", workload=str(workload)) from error
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or not results:
            raise SimulatorError("simulator returned no candidates", workload=str(workload))
        return payload


def _latency_ms(first: dict[str, object], shape: Shape) -> float:
    value = number_or_none(first, "estimated_latency_ms")
    if value is None:
        raise SimulatorError("candidate has no estimated_latency_ms", workload=shape.label)
    return value


def _first(payload: Mapping[str, object]) -> dict[str, object]:
    results = payload["results"]
    assert isinstance(results, list)
    first = results[0]
    assert isinstance(first, dict)
    return first


# ---------------------------------------------------------------------------
# Static sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StaticSimRow:
    shape: Shape
    decode_tokens: int
    prefill_ms: float
    decode_ms: float
    end_to_end_ms: float
    feasible: bool

    @property
    def decode_ms_per_step(self) -> float:
        return self.decode_ms / self.decode_tokens

    def to_json(self) -> dict[str, object]:
        return {
            "record": "lab.sim_static.v1",
            "batch": self.shape.batch,
            "prompt": self.shape.prompt,
            "decode": self.decode_tokens,
            "prefill_ms": round(self.prefill_ms, 4),
            "decode_ms": round(self.decode_ms, 4),
            "decode_ms_per_step": round(self.decode_ms_per_step, 5),
            "end_to_end_ms": round(self.end_to_end_ms, 4),
            "feasible": self.feasible,
        }


async def sweep_static(
    runner: SimRunner,
    model: ModelSpec,
    par: Parallelism,
    shapes: Sequence[Shape],
    decode_tokens: int,
    calibration: Calibration,
) -> list[StaticSimRow]:
    semaphore = asyncio.Semaphore(runner.concurrency)
    tasks: dict[tuple[Shape, SimPhase], asyncio.Task[dict[str, object]]] = {}
    async with asyncio.TaskGroup() as group:
        for shape in shapes:
            for phase in SimPhase:
                name = f"{calibration.label}-tp{par.tp}-pp{par.pp}-{shape.label}-{phase.value}"
                text = static_workload_toml(model, par, shape, decode_tokens, phase, calibration)
                tasks[(shape, phase)] = group.create_task(runner.run(name, text, semaphore))
    rows: list[StaticSimRow] = []
    for shape in shapes:
        firsts = {phase: _first(tasks[(shape, phase)].result()) for phase in SimPhase}
        rows.append(
            StaticSimRow(
                shape=shape,
                decode_tokens=decode_tokens,
                prefill_ms=_latency_ms(firsts[SimPhase.PREFILL], shape),
                decode_ms=_latency_ms(firsts[SimPhase.DECODE], shape),
                end_to_end_ms=_latency_ms(firsts[SimPhase.END_TO_END], shape),
                feasible=all(bool(f["feasible"]) for f in firsts.values()),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Serving sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServeSimResult:
    request_rate: float
    reference_batch: int
    request_count: int
    feasible: bool
    status: str
    rejected_reason: str | None
    ttft: LatencyStats | None
    tpot: LatencyStats | None
    itl: LatencyStats | None
    e2el: LatencyStats | None
    output_throughput: float | None
    request_throughput: float | None

    def to_json(self) -> dict[str, object]:
        def stats(value: LatencyStats | None) -> dict[str, float] | None:
            if value is None:
                return None
            return {"mean_ms": value.mean_ms, "median_ms": value.median_ms, "p99_ms": value.p99_ms}

        return {
            "record": "lab.sim_serving.v1",
            "request_rate": rate_text(self.request_rate),
            "reference_batch": self.reference_batch,
            "request_count": self.request_count,
            "feasible": self.feasible,
            "status": self.status,
            "rejected_reason": self.rejected_reason,
            "ttft": stats(self.ttft),
            "tpot": stats(self.tpot),
            "itl": stats(self.itl),
            "e2el": stats(self.e2el),
            "output_throughput": self.output_throughput,
            "request_throughput": self.request_throughput,
        }


def _sim_stats(metrics: dict[str, object], metric: str) -> LatencyStats | None:
    mean = number_or_none(metrics, f"{metric}_ms")
    median = number_or_none(metrics, f"{metric}_p50_ms")
    p99 = number_or_none(metrics, f"{metric}_p99_ms")
    if mean is None or median is None or p99 is None:
        return None
    return LatencyStats(mean_ms=mean, median_ms=median, p99_ms=p99)


def parse_serving_payload(
    payload: Mapping[str, object], *, rate: float, reference_batch: int, request_count: int, output_len: int
) -> ServeSimResult:
    first = _first(payload)
    metrics = first.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    output_throughput = number_or_none(metrics, "throughput_tokens_per_s")
    return ServeSimResult(
        request_rate=rate,
        reference_batch=reference_batch,
        request_count=request_count,
        feasible=bool(first.get("feasible")),
        status=str(first.get("status")),
        rejected_reason=str(first["rejected_reason"]) if first.get("rejected_reason") else None,
        ttft=_sim_stats(metrics, "ttft"),
        tpot=_sim_stats(metrics, "tpot"),
        itl=_sim_stats(metrics, "itl"),
        e2el=_sim_stats(metrics, "e2el"),
        output_throughput=output_throughput,
        request_throughput=output_throughput / output_len if output_throughput else None,
    )


def cluster_hbm_gb(cluster: Path, node_id: int) -> float:
    data = tomllib.loads(cluster.read_text(encoding="utf-8"))
    for node in data.get("nodes", []):
        if node.get("id") == node_id and "hbm_gb" in node:
            return float(node["hbm_gb"])
    raise SimulatorError(f"cluster has no node {node_id} with an explicit hbm_gb", workload=str(cluster))


async def littles_law_batch(
    runner: SimRunner,
    exp: Experiment,
    wl: ServingWorkload,
    rate: float,
    request_count: int,
    calibration: Calibration,
) -> int:
    """Steady-state concurrency L = rate * E2E(L) from static simulations.

    Only meaningful for serving candidates the simulator schedules phase by
    phase (it scales decode iterations from the reference batch); the
    iteration engine used for this harness's colocated workloads ignores it.

    E2E(L) = prefill(batch 1) + output_len * step(L), with step(L) the static
    decode step at batch L. Iterated from L = 1 until it stops changing (at
    most 8 rounds), clamped to [1, min(max_num_seqs, request_count)]. For
    rate = inf every request is in flight at once, so L is the clamp itself.
    """
    ceiling = min(exp.engine.max_num_seqs, request_count)
    if math.isinf(rate):
        return ceiling
    rows = await sweep_static(
        runner, exp.model, exp.parallelism, [Shape(1, wl.input_len)], wl.output_len, calibration
    )
    prefill_ms = rows[0].prefill_ms
    batch = 1
    for _ in range(8):
        step = await sweep_static(
            runner, exp.model, exp.parallelism, [Shape(batch, wl.input_len)], wl.output_len, calibration
        )
        e2e_s = (prefill_ms + wl.output_len * step[0].decode_ms_per_step) / 1000.0
        target = min(ceiling, max(1, math.ceil(rate * e2e_s)))
        if target == batch:
            break
        batch = target
    return batch


async def _serving_point(
    runner: SimRunner,
    exp: Experiment,
    wl: ServingWorkload,
    rate: float,
    request_count: int,
    kv: KvBudget,
    calibration: Calibration,
    semaphore: asyncio.Semaphore,
) -> ServeSimResult:
    match wl.sim_reference_batch:
        case FixedBatch(batch=fixed):
            batch = fixed
        case LittlesLawBatch():
            batch = await littles_law_batch(runner, exp, wl, rate, request_count, calibration)
    text = serving_workload_toml(
        exp, wl, rate=rate, reference_batch=batch, request_count=request_count, kv=kv, calibration=calibration
    )
    name = f"{calibration.label}-serving-ref{batch}-{rate_label(rate)}"
    payload = await runner.run(name, text, semaphore)
    return parse_serving_payload(
        payload, rate=rate, reference_batch=batch, request_count=request_count, output_len=wl.output_len
    )


async def sweep_serving(
    runner: SimRunner,
    exp: Experiment,
    wl: ServingWorkload,
    calibration: Calibration,
    *,
    measured_kv_tokens: int | None = None,
) -> list[ServeSimResult]:
    """One serving simulation per request rate, run concurrently."""
    hbm = cluster_hbm_gb(runner.cluster, exp.head.sim_node_id)
    kv = kv_budget(exp, hbm, wl.kv_cache_tokens or measured_kv_tokens)
    request_count = wl.sim_request_count or wl.num_prompts
    semaphore = asyncio.Semaphore(runner.concurrency)
    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(
                _serving_point(runner, exp, wl, rate, request_count, kv, calibration, semaphore)
            )
            for rate in wl.request_rates
        ]
    return [task.result() for task in tasks]
