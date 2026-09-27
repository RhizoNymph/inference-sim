"""Locating and hashing the measured input tables and system spec."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from converter.errors import TableLoadError
from converter.spec import GpuSpec

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def parse_simple_yaml_scalars(text: str) -> dict[str, dict[str, str]]:
    """Two-level `section: / key: value` reader for AISimulate system YAMLs."""
    sections: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key, _, value = line.partition(":")
            current = key.strip()
            sections.setdefault(current, {})
            if value.strip():
                sections[current]["__value__"] = value.strip()
            continue
        if current is None or line.startswith("    "):
            continue
        key, _, value = line.strip().partition(":")
        if value.strip():
            sections[current][key.strip()] = value.strip().strip("'\"")
    return sections


def load_gpu_spec(path: Path) -> GpuSpec:
    try:
        sections = parse_simple_yaml_scalars(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise TableLoadError(
            f"cannot read system spec: {error}", source=str(path)
        ) from error
    gpu = sections.get("gpu", {})
    node = sections.get("node", {})
    misc = sections.get("misc", {})
    try:
        return GpuSpec(
            mem_bw=float(gpu["mem_bw"]),
            mem_bw_empirical_scaling_factor=float(
                gpu["mem_bw_empirical_scaling_factor"]
            ),
            mem_empirical_constant_latency=float(gpu["mem_empirical_constant_latency"]),
            tc_flops={
                key.removesuffix("_tc_flops"): float(value)
                for key, value in gpu.items()
                if key.endswith("_tc_flops")
            },
            nccl_version=misc.get("nccl_version", "unknown"),
            num_gpus_per_node=int(node.get("num_gpus_per_node", 8)),
        )
    except (KeyError, ValueError) as error:
        raise TableLoadError(
            f"system spec is missing a required gpu field: {error}", source=str(path)
        ) from error


def load_gpu_spec_json(path: Path) -> GpuSpec:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TableLoadError(
            f"cannot read system spec: {error}", source=str(path)
        ) from error
    try:
        return GpuSpec(
            mem_bw=float(payload["mem_bw"]),
            mem_bw_empirical_scaling_factor=float(
                payload["mem_bw_empirical_scaling_factor"]
            ),
            mem_empirical_constant_latency=float(
                payload["mem_empirical_constant_latency"]
            ),
            tc_flops={k: float(v) for k, v in payload["tc_flops"].items()},
            nccl_version=str(payload["nccl_version"]),
            num_gpus_per_node=int(payload["num_gpus_per_node"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TableLoadError(
            f"system spec json is missing a required field: {error}", source=str(path)
        ) from error


@dataclasses.dataclass(frozen=True, slots=True)
class TableSources:
    gemm: Path
    context_attention: Path
    generation_attention: Path
    allreduce: Path

    def all(self) -> tuple[Path, ...]:
        return (
            self.gemm,
            self.context_attention,
            self.generation_attention,
            self.allreduce,
        )

    def require(self) -> None:
        for path in self.all():
            if not path.is_file():
                raise TableLoadError(
                    f"required table is missing: {path}", source=str(path)
                )


def parquet_sources(
    root: Path, system: str, backend: str, version: str
) -> TableSources:
    data = root / "python/aisimulate/src/aisimulate_core/systems/data" / system
    if not data.is_dir():
        legacy = (
            root / "python/aisimulate/src/aiconfigurator_core/systems/data" / system
        )
        if legacy.is_dir():
            data = legacy
        else:
            raise TableLoadError(
                f"no measured data directory for system '{system}' under {root}",
                source=str(data),
            )
    return TableSources(
        gemm=data / "gemm" / backend / version / "gemm_perf.parquet",
        context_attention=data
        / "attention"
        / backend
        / version
        / "context_attention_perf.parquet",
        generation_attention=data
        / "attention"
        / backend
        / version
        / "generation_attention_perf.parquet",
        allreduce=data / "comm" / backend / version / "custom_allreduce_perf.parquet",
    )


def csv_sources(root: Path) -> TableSources:
    return TableSources(
        gemm=root / "gemm_perf.csv",
        context_attention=root / "context_attention_perf.csv",
        generation_attention=root / "generation_attention_perf.csv",
        allreduce=root / "custom_allreduce_perf.csv",
    )


def environment_hash(sources: TableSources) -> str:
    digest = hashlib.sha256()
    for path in sorted(sources.all(), key=lambda p: p.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return f"sha256:{digest.hexdigest()}"


def read_git_commit(root: Path) -> str:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return "unknown"
    text = head.read_text(encoding="utf-8").strip()
    if not text.startswith("ref:"):
        return text
    ref = (root / ".git" / text.split(" ", 1)[1].strip()).resolve()
    if ref.is_file():
        return ref.read_text(encoding="utf-8").strip()
    packed = root / ".git" / "packed-refs"
    if packed.is_file():
        needle = text.split(" ", 1)[1].strip()
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(f" {needle}"):
                return line.split(" ", 1)[0]
    return "unknown"
