"""Measured collective curves: collective-benchmark JSON lines -> `[[collective_curves]]` TOML.

Input is the output of `remote/collective_bench.py` (`lab.collective_meta.v1`,
`lab.collective.v1` rows, `lab.done.v1`) or the legacy two-rank format of
`lab-runs/2026-09-30-nccl-curve/collective_curve.jsonl` (rows without a
`record` tag, printed and timed by rank 0 only).

Collective rows are kept as measured. Point-to-point rows are screened:

* Timing side. New rows say who timed them (`timed_on`). In legacy files only
  rank 0 timed, so `send_0to{d}` rows are sender-timed and every other send is
  receiver-timed.
* A sender-timed row is unreliable when its implied bandwidth (bytes /
  median) exceeds `tolerance` x the bandwidth of that op's largest-message row:
  NCCL send returns once the message is buffered, before delivery. Every
  sender-timed row at or below the largest unreliable size is dropped too (a
  send that returns early at size X also does so for smaller sizes).
* If a directed curve then lacks its low end (its smallest kept size is above
  the smallest size measured for any op) and the reverse direction was
  measured, the missing sizes are derived from the reverse curve:
  `t_fwd(b) = t_rev(b) + b * (1/bw_fwd - 1/bw_rev)`, where `bw` is the marginal
  bandwidth of each direction's two largest kept rows. Otherwise the curve is
  emitted truncated and the simulator extrapolates its floor.

`curves_toml.py` renders the result as TOML.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from labharness.errors import CurveError, ResultParseError
from labharness.logging_setup import get_logger

META_RECORD: Final = "lab.collective_meta.v1"
ROW_RECORD: Final = "lab.collective.v1"
DONE_RECORD: Final = "lab.done.v1"
LEGACY_WORLD_SIZE: Final = 2
LEGACY_TIMING_RANK: Final = 0
DEFAULT_SENDER_TIMED_TOLERANCE: Final = 1.25
_SEND_RE: Final = re.compile(r"send_(\d+)to(\d+)")

# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


class TimedOn(StrEnum):
    """Whose clock a row's samples come from."""

    MAX_RANK = "max_rank"  # per-iteration max across ranks (collectives, new format)
    RECEIVER = "receiver"  # the receiving rank (point-to-point, new format)
    SENDER = "sender"  # the sending rank: under-reports buffered sends
    RANK0 = "rank0"  # legacy collectives: rank 0's clock only


class CollectiveKind(StrEnum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"


@dataclass(frozen=True, slots=True, order=True)
class CollectiveOpKey:
    kind: CollectiveKind

    @property
    def label(self) -> str:
        return self.kind.value


@dataclass(frozen=True, slots=True, order=True)
class SendOpKey:
    src_rank: int
    dst_rank: int

    @property
    def label(self) -> str:
        return f"send_{self.src_rank}to{self.dst_rank}"

    def reverse(self) -> SendOpKey:
        return SendOpKey(self.dst_rank, self.src_rank)


type OpKey = CollectiveOpKey | SendOpKey


@dataclass(frozen=True, slots=True)
class Row:
    op: OpKey
    message_bytes: int
    median_us: float
    timed_on: TimedOn
    world_size: int

    @property
    def bandwidth(self) -> float:
        """Implied algorithm bandwidth in bytes/us (= MB/s)."""
        return self.message_bytes / self.median_us


@dataclass(frozen=True, slots=True)
class RankInfo:
    rank: int
    hostname: str | None
    sim_node_id: int | None


@dataclass(frozen=True, slots=True)
class BenchMeta:
    world_size: int
    dtype: str | None
    ranks: tuple[RankInfo, ...]


@dataclass(frozen=True, slots=True)
class BenchData:
    meta: BenchMeta | None
    rows: tuple[Row, ...]
    legacy: bool
    done: bool

    @property
    def world_size(self) -> int:
        if self.meta is not None:
            return self.meta.world_size
        return max(row.world_size for row in self.rows)

    @property
    def smallest_bytes(self) -> int:
        return min(row.message_bytes for row in self.rows)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


class ScopeKind(StrEnum):
    """`--scope` for collective curves."""

    NODE_GROUP = "node_group"
    INTER_NODE = "inter_node"


@dataclass(frozen=True, slots=True)
class NodeGroupScope:
    nodes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class InterNodeScope:
    pass


type CurveScope = NodeGroupScope | InterNodeScope


@dataclass(frozen=True, slots=True)
class CurvePoint:
    message_bytes: int
    latency_us: float


@dataclass(frozen=True, slots=True)
class MeasuredLowEnd:
    """Every point is measured; the curve starts at the smallest benchmarked size."""


@dataclass(frozen=True, slots=True)
class DerivedLowEnd:
    """Points below `below_bytes` come from the reverse direction's curve."""

    below_bytes: int
    reverse: SendOpKey
    forward_bandwidth: float  # bytes/us
    reverse_bandwidth: float  # bytes/us


@dataclass(frozen=True, slots=True)
class TruncatedLowEnd:
    """No reverse direction to derive from: the curve starts above the benchmark floor."""

    smallest_bytes: int
    floor_bytes: int


type LowEnd = MeasuredLowEnd | DerivedLowEnd | TruncatedLowEnd


class DropReason(StrEnum):
    IMPLAUSIBLE_BANDWIDTH = "implausible_bandwidth"
    BELOW_UNRELIABLE_SIZE = "at_or_below_unreliable_size"


@dataclass(frozen=True, slots=True)
class DroppedRow:
    op: SendOpKey
    message_bytes: int
    median_us: float
    bandwidth: float  # bytes/us
    limit_bandwidth: float  # tolerance x the largest row's bandwidth
    reason: DropReason


@dataclass(frozen=True, slots=True)
class CollectiveCurve:
    kind: CollectiveKind
    scope: CurveScope
    ranks: int
    timed_on: TimedOn
    points: tuple[CurvePoint, ...]


@dataclass(frozen=True, slots=True)
class SendRecvCurve:
    op: SendOpKey
    src_node: int
    dst_node: int
    timed_on: frozenset[TimedOn]
    measured_rows: int
    points: tuple[CurvePoint, ...]
    low_end: LowEnd

    @property
    def derived_below_bytes(self) -> int | None:
        match self.low_end:
            case DerivedLowEnd(below_bytes=below):
                return below
            case MeasuredLowEnd() | TruncatedLowEnd():
                return None


@dataclass(frozen=True, slots=True)
class CurveOptions:
    rank_nodes: tuple[int, ...]
    scope: ScopeKind = ScopeKind.NODE_GROUP
    sender_timed_tolerance: float = DEFAULT_SENDER_TIMED_TOLERANCE
    source: str | None = None


@dataclass(frozen=True, slots=True)
class CurveSet:
    options: CurveOptions
    legacy: bool
    collectives: tuple[CollectiveCurve, ...]
    send_recv: tuple[SendRecvCurve, ...]
    dropped: tuple[DroppedRow, ...]

    def dropped_for(self, op: SendOpKey) -> tuple[DroppedRow, ...]:
        return tuple(row for row in self.dropped if row.op == op)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_op(text: str, *, source: str) -> OpKey:
    if (match := _SEND_RE.fullmatch(text)) is not None:
        src, dst = int(match.group(1)), int(match.group(2))
        if src == dst:
            raise CurveError(f"send op {text!r} has the same source and destination", op=text, source=source)
        return SendOpKey(src, dst)
    try:
        return CollectiveOpKey(CollectiveKind(text))
    except ValueError as error:
        raise CurveError(f"unknown op {text!r}", op=text, source=source) from error


def _int_field(record: Mapping[str, object], key: str, source: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResultParseError(f"{key} must be an integer in {record!r}", source=source)
    return value


def _float_field(record: Mapping[str, object], key: str, source: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ResultParseError(f"{key} must be a number in {record!r}", source=source)
    return float(value)


def _legacy_timed_on(op: OpKey) -> TimedOn:
    match op:
        case SendOpKey(src_rank=src) if src == LEGACY_TIMING_RANK:
            return TimedOn.SENDER
        case SendOpKey():
            return TimedOn.RECEIVER
        case CollectiveOpKey():
            return TimedOn.RANK0


def _parse_row(record: Mapping[str, object], *, legacy: bool, source: str) -> Row:
    op_text = record.get("op")
    if not isinstance(op_text, str):
        raise ResultParseError(f"row without an op: {record!r}", source=source)
    op = parse_op(op_text, source=source)
    message_bytes = _int_field(record, "bytes", source)
    median_us = _float_field(record, "median_us", source)
    if message_bytes <= 0:
        raise CurveError(f"{op.label}: non-positive message size {message_bytes}", op=op.label, source=source)
    if not (math.isfinite(median_us) and median_us > 0):
        raise CurveError(
            f"{op.label}: non-positive median {median_us} us at {message_bytes} bytes",
            op=op.label,
            source=source,
        )
    if legacy:
        return Row(op, message_bytes, median_us, _legacy_timed_on(op), LEGACY_WORLD_SIZE)
    timed_text = record.get("timed_on")
    try:
        timed_on = TimedOn(str(timed_text))
    except ValueError as error:
        raise ResultParseError(f"unknown timed_on {timed_text!r}", source=source) from error
    return Row(op, message_bytes, median_us, timed_on, _int_field(record, "world_size", source))


def _parse_meta(record: Mapping[str, object], source: str) -> BenchMeta:
    ranks_value = record.get("ranks")
    if not isinstance(ranks_value, list):
        raise ResultParseError("meta record needs a ranks list", source=source)
    ranks: list[RankInfo] = []
    for item in ranks_value:
        if not isinstance(item, dict):
            raise ResultParseError(f"meta rank entry must be an object: {item!r}", source=source)
        node = item.get("sim_node_id")
        hostname = item.get("hostname")
        ranks.append(
            RankInfo(
                rank=_int_field(item, "rank", source),
                hostname=hostname if isinstance(hostname, str) else None,
                sim_node_id=node if isinstance(node, int) and not isinstance(node, bool) else None,
            )
        )
    dtype = record.get("dtype")
    return BenchMeta(
        world_size=_int_field(record, "world_size", source),
        dtype=dtype if isinstance(dtype, str) else None,
        ranks=tuple(sorted(ranks, key=lambda info: info.rank)),
    )


def parse_bench_lines(lines: Iterable[str], *, source: str) -> BenchData:
    """Strict: every non-blank line must be a JSON object this module understands."""
    meta: BenchMeta | None = None
    rows: list[Row] = []
    kinds: set[bool] = set()
    done = False
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as error:
            raise ResultParseError(f"line {number} is not JSON: {error}", source=source) from error
        if not isinstance(record, dict):
            raise ResultParseError(f"line {number} is not a JSON object", source=source)
        match record.get("record"):
            case None if "op" in record:
                kinds.add(True)
                rows.append(_parse_row(record, legacy=True, source=source))
            case "lab.collective.v1":
                kinds.add(False)
                rows.append(_parse_row(record, legacy=False, source=source))
            case "lab.collective_meta.v1":
                meta = _parse_meta(record, source)
            case "lab.done.v1":
                done = True
            case other:
                raise ResultParseError(f"line {number}: unexpected record {other!r}", source=source)
    if not rows:
        raise CurveError("no collective rows found", source=source)
    if len(kinds) > 1:
        raise CurveError("file mixes legacy and lab.collective.v1 rows", source=source)
    return BenchData(meta=meta, rows=tuple(rows), legacy=kinds == {True}, done=done)


def load_bench(path: Path) -> BenchData:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ResultParseError(f"cannot read collective results: {error}", source=str(path)) from error
    return parse_bench_lines(text.splitlines(), source=str(path))


def parse_rank_nodes(text: str) -> tuple[int, ...]:
    """`0,1` -> (0, 1): simulator node id of rank 0, rank 1, ..."""
    parts = [part.strip() for part in text.split(",")]
    if not parts or not all(part.isdigit() for part in parts):
        raise CurveError(f"--rank-nodes must be comma-separated node ids, got {text!r}")
    return tuple(int(part) for part in parts)


def rank_nodes_from_meta(meta: BenchMeta | None) -> tuple[int, ...] | None:
    """Rank -> sim node id from the meta line, or None when any rank lacks one."""
    if meta is None:
        return None
    nodes = [info.sim_node_id for info in meta.ranks]
    if [info.rank for info in meta.ranks] != list(range(meta.world_size)):
        return None
    if any(node is None for node in nodes):
        return None
    return tuple(node for node in nodes if node is not None)


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------


def _group(rows: Sequence[Row], source: str) -> dict[OpKey, list[Row]]:
    groups: dict[OpKey, list[Row]] = {}
    for row in rows:
        groups.setdefault(row.op, []).append(row)
    for op, group in groups.items():
        group.sort(key=lambda row: row.message_bytes)
        sizes = [row.message_bytes for row in group]
        if len(set(sizes)) != len(sizes):
            duplicates = sorted({size for size in sizes if sizes.count(size) > 1})
            raise CurveError(f"{op.label}: duplicate message sizes {duplicates}", op=op.label, source=source)
        worlds = {row.world_size for row in group}
        if len(worlds) != 1:
            raise CurveError(f"{op.label}: mixed world sizes {sorted(worlds)}", op=op.label, source=source)
    return groups


def screen_sender_timed(
    op: SendOpKey, rows: Sequence[Row], tolerance: float
) -> tuple[list[Row], list[DroppedRow]]:
    """(kept rows, dropped rows) for one directed op; `rows` sorted by size."""
    limit = tolerance * rows[-1].bandwidth
    unreliable = [
        row.message_bytes for row in rows if row.timed_on is TimedOn.SENDER and row.bandwidth > limit
    ]
    if not unreliable:
        return list(rows), []
    cutoff = max(unreliable)
    kept: list[Row] = []
    dropped: list[DroppedRow] = []
    for row in rows:
        if row.timed_on is TimedOn.SENDER and row.message_bytes <= cutoff:
            reason = (
                DropReason.IMPLAUSIBLE_BANDWIDTH
                if row.bandwidth > limit
                else DropReason.BELOW_UNRELIABLE_SIZE
            )
            dropped.append(DroppedRow(op, row.message_bytes, row.median_us, row.bandwidth, limit, reason))
        else:
            kept.append(row)
    return kept, dropped


def marginal_bandwidth(rows: Sequence[Row]) -> float:
    """(b2 - b1) / (t2 - t1) over the two largest rows; b/t of the largest when that is not positive."""
    largest = rows[-1]
    if len(rows) >= 2:
        previous = rows[-2]
        dt = largest.median_us - previous.median_us
        if dt > 0:
            return (largest.message_bytes - previous.message_bytes) / dt
    return largest.bandwidth


def _node_of(rank: int, rank_nodes: tuple[int, ...]) -> int:
    return rank_nodes[rank]


def _points(rows: Iterable[Row]) -> tuple[CurvePoint, ...]:
    return tuple(CurvePoint(row.message_bytes, row.median_us) for row in rows)


def _send_curve(
    op: SendOpKey,
    kept: Mapping[SendOpKey, list[Row]],
    measured_rows: int,
    floor_bytes: int,
    options: CurveOptions,
    source: str,
) -> SendRecvCurve:
    rows = kept[op]
    src_node = _node_of(op.src_rank, options.rank_nodes)
    dst_node = _node_of(op.dst_rank, options.rank_nodes)
    if src_node == dst_node:
        raise CurveError(
            f"{op.label}: ranks {op.src_rank} and {op.dst_rank} map to the same node {src_node}; "
            "node_pair curves need distinct nodes",
            op=op.label,
            source=source,
        )
    timed_on = frozenset(row.timed_on for row in rows)
    log = get_logger()
    smallest = rows[0].message_bytes
    reverse = kept.get(op.reverse())
    if smallest <= floor_bytes:
        low_end: LowEnd = MeasuredLowEnd()
        points = _points(rows)
    elif reverse is None or not any(row.message_bytes < smallest for row in reverse):
        low_end = TruncatedLowEnd(smallest_bytes=smallest, floor_bytes=floor_bytes)
        points = _points(rows)
        log.warning(
            "send curve truncated; no reverse direction to derive its low end",
            extra={"op": op.label, "smallest_bytes": smallest, "floor_bytes": floor_bytes},
        )
    else:
        forward_bw = marginal_bandwidth(rows)
        reverse_bw = marginal_bandwidth(reverse)
        slope = 1.0 / forward_bw - 1.0 / reverse_bw
        derived: list[CurvePoint] = []
        for row in reverse:
            if row.message_bytes >= smallest:
                break
            latency = row.median_us + row.message_bytes * slope
            if not latency > 0:
                raise CurveError(
                    f"{op.label}: derived latency {latency:.3f} us at {row.message_bytes} bytes "
                    "is not positive",
                    op=op.label,
                    source=source,
                )
            derived.append(CurvePoint(row.message_bytes, latency))
        low_end = DerivedLowEnd(
            below_bytes=smallest,
            reverse=op.reverse(),
            forward_bandwidth=forward_bw,
            reverse_bandwidth=reverse_bw,
        )
        points = (*derived, *_points(rows))
        log.info(
            "derived send low end from reverse direction",
            extra={
                "op": op.label,
                "reverse_op": op.reverse().label,
                "derived_points": len(derived),
                "derived_below_bytes": smallest,
                "forward_bytes_per_us": round(forward_bw, 2),
                "reverse_bytes_per_us": round(reverse_bw, 2),
            },
        )
    if len(points) < 2:
        raise CurveError(f"{op.label}: fewer than 2 usable points", op=op.label, source=source)
    return SendRecvCurve(
        op=op,
        src_node=src_node,
        dst_node=dst_node,
        timed_on=timed_on,
        measured_rows=measured_rows,
        points=points,
        low_end=low_end,
    )


def _collective_scope(world_size: int, options: CurveOptions, op: str, source: str) -> CurveScope:
    match options.scope:
        case ScopeKind.INTER_NODE:
            return InterNodeScope()
        case ScopeKind.NODE_GROUP:
            nodes = tuple(sorted(set(options.rank_nodes[:world_size])))
            if len(nodes) < 2:
                raise CurveError(
                    f"{op}: node_group needs ranks on at least 2 distinct nodes, got {list(nodes)}",
                    op=op,
                    source=source,
                )
            return NodeGroupScope(nodes)


def build_curves(data: BenchData, options: CurveOptions, *, source: str) -> CurveSet:
    if not (math.isfinite(options.sender_timed_tolerance) and options.sender_timed_tolerance >= 1.0):
        raise CurveError(
            f"--sender-timed-tolerance must be >= 1.0, got {options.sender_timed_tolerance}", source=source
        )
    world_size = data.world_size
    if len(options.rank_nodes) != world_size:
        raise CurveError(
            f"--rank-nodes lists {len(options.rank_nodes)} ranks but the benchmark has {world_size}",
            source=source,
        )
    groups = _group(data.rows, source)
    log = get_logger()

    collectives: list[CollectiveCurve] = []
    kept: dict[SendOpKey, list[Row]] = {}
    measured_counts: dict[SendOpKey, int] = {}
    dropped: list[DroppedRow] = []
    for op in sorted(groups, key=_op_sort_key):
        rows = groups[op]
        match op:
            case CollectiveOpKey(kind=kind):
                if len(rows) < 2:
                    raise CurveError(f"{op.label}: fewer than 2 points", op=op.label, source=source)
                ranks = rows[0].world_size
                collectives.append(
                    CollectiveCurve(
                        kind=kind,
                        scope=_collective_scope(ranks, options, op.label, source),
                        ranks=ranks,
                        timed_on=rows[0].timed_on,
                        points=_points(rows),
                    )
                )
            case SendOpKey():
                if max(op.src_rank, op.dst_rank) >= world_size:
                    raise CurveError(
                        f"{op.label}: rank out of range for world size {world_size}",
                        op=op.label,
                        source=source,
                    )
                op_kept, op_dropped = screen_sender_timed(op, rows, options.sender_timed_tolerance)
                for row in op_dropped:
                    log.warning(
                        "dropped sender-timed row",
                        extra={
                            "op": op.label,
                            "message_bytes": row.message_bytes,
                            "median_us": row.median_us,
                            "bytes_per_us": round(row.bandwidth, 2),
                            "limit_bytes_per_us": round(row.limit_bandwidth, 2),
                            "reason": row.reason.value,
                        },
                    )
                kept[op] = op_kept
                measured_counts[op] = len(rows)
                dropped += op_dropped

    floor = data.smallest_bytes
    send_curves = tuple(
        _send_curve(op, kept, measured_counts[op], floor, options, source) for op in sorted(kept)
    )
    return CurveSet(
        options=options,
        legacy=data.legacy,
        collectives=tuple(collectives),
        send_recv=send_curves,
        dropped=tuple(dropped),
    )


def _op_sort_key(op: OpKey) -> tuple[int, str, int, int]:
    match op:
        case CollectiveOpKey(kind=kind):
            return (0, kind.value, 0, 0)
        case SendOpKey(src_rank=src, dst_rank=dst):
            return (1, "", src, dst)
