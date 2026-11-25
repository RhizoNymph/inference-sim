"""Render a `CurveSet` as the simulator's `[[collective_curves]]` TOML.

The emitted keys match the simulator's schema exactly (it rejects unknown
keys); every explanation (timing side, dropped rows, derived or truncated low
ends) goes into `#` comments. Latencies are microseconds with two decimals,
message sizes integers. Every rendered text is parsed back with tomllib before
it is returned.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from typing import Final

from labharness.curves import (
    CollectiveCurve,
    CurveOptions,
    CurvePoint,
    CurveSet,
    DerivedLowEnd,
    DroppedRow,
    InterNodeScope,
    MeasuredLowEnd,
    NodeGroupScope,
    SendRecvCurve,
    TimedOn,
    TruncatedLowEnd,
)
from labharness.errors import CurveError
from labharness.toml_emit import toml_string

_DROPPED_SIZES_PER_LINE: Final = 8


def _fmt_latency(value: float) -> str:
    return f"{value:.2f}"


def _points_lines(points: Sequence[CurvePoint]) -> list[str]:
    return [
        "points = [",
        *(f"  [{point.message_bytes}, {_fmt_latency(point.latency_us)}]," for point in points),
        "]",
    ]


def _gb_s(bytes_per_us: float) -> str:
    return f"{bytes_per_us / 1000:.3f} GB/s"


def _wrap_sizes(sizes: Sequence[int]) -> list[str]:
    return [
        "#   " + ", ".join(str(size) for size in sizes[i : i + _DROPPED_SIZES_PER_LINE])
        for i in range(0, len(sizes), _DROPPED_SIZES_PER_LINE)
    ]


_TIMED_ON_TEXT: Final = {
    TimedOn.MAX_RANK: "per-iteration max across ranks",
    TimedOn.RECEIVER: "receiver-timed",
    TimedOn.SENDER: "sender-timed",
    TimedOn.RANK0: "rank 0's clock only (legacy two-rank benchmark)",
}


def _collective_block(curve: CollectiveCurve, source: str | None) -> list[str]:
    match curve.scope:
        case NodeGroupScope(nodes=nodes):
            where = f"nodes {list(nodes)}"
            scope_pairs = [("scope", toml_string("node_group")), ("nodes", f"[{', '.join(map(str, nodes))}]")]
        case InterNodeScope():
            where = "any inter-node group"
            scope_pairs = [("scope", toml_string("inter_node"))]
    lines = [
        f"# {curve.kind.value}: {curve.ranks} ranks over {where}; {len(curve.points)} measured points, "
        f"{_TIMED_ON_TEXT[curve.timed_on]}.",
        "[[collective_curves]]",
        f"op = {toml_string(curve.kind.value)}",
        *(f"{key} = {value}" for key, value in scope_pairs),
        f"ranks = {curve.ranks}",
    ]
    if source is not None:
        lines.append(f"source = {toml_string(source)}")
    return lines + _points_lines(curve.points)


def _send_comments(curve: SendRecvCurve, dropped: Sequence[DroppedRow], tolerance: float) -> list[str]:
    timing = ", ".join(sorted(_TIMED_ON_TEXT[t] for t in curve.timed_on))
    lines = [
        f"# {curve.op.label}: rank {curve.op.src_rank} (node {curve.src_node}) -> "
        f"rank {curve.op.dst_rank} (node {curve.dst_node}); {timing}; "
        f"{curve.measured_rows - len(dropped)} of {curve.measured_rows} measured rows kept.",
    ]
    if dropped:
        cutoff = max(row.message_bytes for row in dropped)
        largest_bw = _gb_s(dropped[0].limit_bandwidth / tolerance)
        lines += [
            f"# Dropped {len(dropped)} sender-timed rows: the send returned before delivery (implied",
            f"# bandwidth > {tolerance:g}x the largest message's {largest_bw} up to {cutoff} bytes; every",
            "# smaller sender-timed row goes too). Dropped message bytes:",
        ]
        lines += _wrap_sizes([row.message_bytes for row in dropped])
    match curve.low_end:
        case MeasuredLowEnd():
            pass
        case DerivedLowEnd() as derived:
            lines.append(
                f"# Points below {derived.below_bytes} bytes are derived from {derived.reverse.label}: "
                "t(b) = t_rev(b) + b * (1/bw_fwd - 1/bw_rev),"
            )
            lines.append(
                f"# bw_fwd = {_gb_s(derived.forward_bandwidth)}, bw_rev = {_gb_s(derived.reverse_bandwidth)} "
                "(marginal bandwidth of each direction's two largest kept rows)."
            )
        case TruncatedLowEnd() as truncated:
            lines.append(
                f"# No reverse direction measured: the curve starts at {truncated.smallest_bytes} bytes "
                f"(benchmark floor {truncated.floor_bytes}); the simulator extrapolates its floor below that."
            )
    return lines


def _send_block(curve: SendRecvCurve, dropped: Sequence[DroppedRow], options: CurveOptions) -> list[str]:
    lines = [
        *_send_comments(curve, dropped, options.sender_timed_tolerance),
        "[[collective_curves]]",
        'op = "send_recv"',
        'scope = "node_pair"',
        f"src_node = {curve.src_node}",
        f"dst_node = {curve.dst_node}",
    ]
    if options.source is not None:
        lines.append(f"source = {toml_string(options.source)}")
    if (below := curve.derived_below_bytes) is not None:
        lines.append(f"derived_below_bytes = {below}")
    return lines + _points_lines(curve.points)


def render_curves(curves: CurveSet) -> str:
    """The `[[collective_curves]]` section, comments included."""
    options = curves.options
    mapping = ", ".join(f"rank {rank} -> node {node}" for rank, node in enumerate(options.rank_nodes))
    lines = [
        "# Measured collective curves, generated by `tools/lab/lab.py curves`.",
        *([f"# Source: {options.source}"] if options.source else []),
        f"# {mapping}. Points are [message_bytes_per_rank, latency_us].",
    ]
    if curves.legacy:
        lines.append(
            "# Legacy two-rank benchmark: only rank 0 timed, so send_0to* rows are sender-timed "
            "and were screened."
        )
    blocks = [_collective_block(curve, options.source) for curve in curves.collectives]
    blocks += [_send_block(curve, curves.dropped_for(curve.op), options) for curve in curves.send_recv]
    text = "\n".join(lines) + "\n"
    for block in blocks:
        text += "\n" + "\n".join(block) + "\n"
    return _checked(text, options.source)


def _checked(text: str, source: str | None) -> str:
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise CurveError(f"generated TOML does not parse: {error}", source=source) from error
    return text


def _base_node_ids(base: Mapping[str, object]) -> set[int] | None:
    nodes = base.get("nodes")
    if not isinstance(nodes, list):
        return None
    ids = {node.get("id") for node in nodes if isinstance(node, dict)}
    return {node_id for node_id in ids if isinstance(node_id, int)}


def _referenced_nodes(curves: CurveSet) -> set[int]:
    nodes: set[int] = set()
    for curve in curves.collectives:
        match curve.scope:
            case NodeGroupScope(nodes=group):
                nodes.update(group)
            case InterNodeScope():
                pass
    for send in curves.send_recv:
        nodes.update((send.src_node, send.dst_node))
    return nodes


def check_base_cluster(base_text: str, curves: CurveSet, *, base_path: str) -> None:
    """The base cluster must parse, carry no curves yet, and define every referenced node."""
    try:
        base = tomllib.loads(base_text)
    except tomllib.TOMLDecodeError as error:
        raise CurveError(f"base cluster is not valid TOML: {error}", source=base_path) from error
    if "collective_curves" in base:
        raise CurveError("base cluster already has [[collective_curves]]", source=base_path)
    known = _base_node_ids(base)
    if known is not None and (missing := sorted(_referenced_nodes(curves) - known)):
        raise CurveError(f"curves reference nodes {missing} missing from the base cluster", source=base_path)


def render_cluster(base_text: str | None, curves: CurveSet, *, base_path: str | None) -> str:
    """Generated header + base cluster text (if any) + the curve section; checked to parse."""
    header = ["# Generated by `tools/lab/lab.py curves`; regenerate rather than editing the curves below."]
    if base_path is not None:
        header.append(f"# Base cluster: {base_path}")
    parts = ["\n".join(header) + "\n"]
    if base_text is not None:
        parts.append(base_text.rstrip("\n") + "\n")
    parts.append(render_curves(curves))
    return _checked("\n".join(parts), base_path)
