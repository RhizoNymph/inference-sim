"""Collective-benchmark rows -> [[collective_curves]] TOML (curves.py, curves_toml.py, `lab.py curves`)."""

from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path

import pytest

import lab
from labharness.curves import (
    BenchData,
    CollectiveKind,
    CurveOptions,
    CurveSet,
    DerivedLowEnd,
    DropReason,
    InterNodeScope,
    MeasuredLowEnd,
    NodeGroupScope,
    ScopeKind,
    SendOpKey,
    SendRecvCurve,
    TimedOn,
    TruncatedLowEnd,
    build_curves,
    load_bench,
    parse_bench_lines,
    parse_rank_nodes,
    rank_nodes_from_meta,
)
from labharness.curves_toml import check_base_cluster, render_cluster, render_curves
from labharness.errors import CurveError, ResultParseError
from tests.conftest import FIXTURES, REPO_ROOT

LEGACY = FIXTURES / "nccl_curve_legacy.jsonl"
BASE_CLUSTER = REPO_ROOT / "lab-runs" / "2026-09-30-tp2" / "rtx3090_lab_cluster.toml"
SIZES = [1024 * 2**i for i in range(19)]
MIB = 1024 * 1024

COLLECTIVE_KEYS = {"op", "scope", "nodes", "ranks", "source", "points"}
SEND_KEYS = {"op", "scope", "src_node", "dst_node", "source", "derived_below_bytes", "points"}


def _legacy_curves(
    *, scope: ScopeKind = ScopeKind.NODE_GROUP, sender_timed_tolerance: float = 1.25
) -> CurveSet:
    opts = CurveOptions(
        rank_nodes=(0, 1),
        scope=scope,
        sender_timed_tolerance=sender_timed_tolerance,
        source="legacy.jsonl",
    )
    return build_curves(load_bench(LEGACY), opts, source=str(LEGACY))


def _send(curves: CurveSet, src: int, dst: int) -> SendRecvCurve:
    return next(c for c in curves.send_recv if c.op == SendOpKey(src, dst))


def _legacy_rows() -> dict[tuple[str, int], float]:
    rows = [json.loads(line) for line in LEGACY.read_text(encoding="utf-8").splitlines()]
    return {(r["op"], r["bytes"]): r["median_us"] for r in rows}


# -- legacy fixture -------------------------------------------------------------


def test_legacy_send_0to1_drops_sender_timed_rows_up_to_32_mib() -> None:
    curves = _legacy_curves()
    dropped = curves.dropped_for(SendOpKey(0, 1))
    assert [d.message_bytes for d in dropped] == SIZES[:16]
    assert dropped[-1].message_bytes == 32 * MIB
    assert all(d.op == SendOpKey(0, 1) for d in curves.dropped)
    # 32 KiB and up exceed the limit themselves; 1..16 KiB are dropped because a larger row was.
    assert {d.message_bytes for d in dropped if d.reason is DropReason.IMPLAUSIBLE_BANDWIDTH} == set(
        SIZES[5:16]
    )
    assert {d.message_bytes for d in dropped if d.reason is DropReason.BELOW_UNRELIABLE_SIZE} == set(
        SIZES[:5]
    )
    largest_bw = 268435456 / 752233.1
    assert dropped[0].limit_bandwidth == pytest.approx(1.25 * largest_bw)


def test_legacy_send_0to1_derives_its_low_end_from_send_1to0() -> None:
    curves = _legacy_curves()
    curve = _send(curves, 0, 1)
    assert (curve.src_node, curve.dst_node) == (0, 1)
    assert curve.timed_on == frozenset({TimedOn.SENDER})
    assert curve.derived_below_bytes == 64 * MIB
    low_end = curve.low_end
    assert isinstance(low_end, DerivedLowEnd)
    assert low_end.reverse == SendOpKey(1, 0)
    assert [p.message_bytes for p in curve.points] == SIZES
    measured = _legacy_rows()
    # Kept measured rows are untouched.
    for point in curve.points[16:]:
        assert point.latency_us == measured[("send_0to1", point.message_bytes)]
    # Derived: t_fwd(b) = t_rev(b) + b * (1/bw_fwd - 1/bw_rev), marginal bandwidths of the top two rows.
    bw_fwd = (268435456 - 134217728) / (752233.1 - 383856.5)
    bw_rev = (268435456 - 134217728) / (228434.65 - 114373.72)
    assert low_end.forward_bandwidth == pytest.approx(bw_fwd)
    assert low_end.reverse_bandwidth == pytest.approx(bw_rev)
    for point in curve.points[:16]:
        b = point.message_bytes
        expected = measured[("send_1to0", b)] + b * (1 / bw_fwd - 1 / bw_rev)
        assert point.latency_us == pytest.approx(expected)
    assert curve.points[0].latency_us == pytest.approx(151.81, abs=0.01)


def test_legacy_send_1to0_is_receiver_timed_and_complete() -> None:
    curve = _send(_legacy_curves(), 1, 0)
    assert curve.timed_on == frozenset({TimedOn.RECEIVER})
    assert curve.low_end == MeasuredLowEnd()
    assert curve.derived_below_bytes is None
    assert len(curve.points) == 19
    assert (curve.points[0].message_bytes, curve.points[0].latency_us) == (1024, 149.87)
    assert (curve.points[-1].message_bytes, curve.points[-1].latency_us) == (268435456, 228434.65)


def test_legacy_all_reduce_keeps_every_row() -> None:
    curves = _legacy_curves()
    assert len(curves.collectives) == 1
    curve = curves.collectives[0]
    assert curve.kind is CollectiveKind.ALL_REDUCE
    assert curve.scope == NodeGroupScope((0, 1))
    assert curve.ranks == 2
    assert curve.timed_on is TimedOn.RANK0
    assert len(curve.points) == 19
    assert (curve.points[0].message_bytes, curve.points[0].latency_us) == (1024, 132.66)


def test_high_tolerance_keeps_every_row() -> None:
    curves = _legacy_curves(sender_timed_tolerance=100.0)
    assert curves.dropped == ()
    assert _send(curves, 0, 1).low_end == MeasuredLowEnd()


def test_tolerance_below_one_is_rejected() -> None:
    with pytest.raises(CurveError, match="tolerance"):
        _legacy_curves(sender_timed_tolerance=0.9)


def test_inter_node_scope_has_no_nodes() -> None:
    curves = _legacy_curves(scope=ScopeKind.INTER_NODE)
    assert curves.collectives[0].scope == InterNodeScope()
    parsed = tomllib.loads(render_curves(curves))
    all_reduce = parsed["collective_curves"][0]
    assert all_reduce["scope"] == "inter_node" and "nodes" not in all_reduce
    # Directed send curves stay node_pair.
    assert {c["scope"] for c in parsed["collective_curves"][1:]} == {"node_pair"}


# -- rendering --------------------------------------------------------------------


def test_rendered_toml_matches_the_simulator_schema() -> None:
    text = render_curves(_legacy_curves())
    curves = tomllib.loads(text)["collective_curves"]
    assert [(c["op"], c["scope"]) for c in curves] == [
        ("all_reduce", "node_group"),
        ("send_recv", "node_pair"),
        ("send_recv", "node_pair"),
    ]
    all_reduce, fwd, rev = curves
    assert set(all_reduce) == COLLECTIVE_KEYS
    assert all_reduce["nodes"] == [0, 1] and all_reduce["ranks"] == 2
    assert all_reduce["source"] == "legacy.jsonl"
    assert set(fwd) == SEND_KEYS
    assert set(rev) == SEND_KEYS - {"derived_below_bytes"}
    assert (fwd["src_node"], fwd["dst_node"], fwd["derived_below_bytes"]) == (0, 1, 67108864)
    assert (rev["src_node"], rev["dst_node"]) == (1, 0)
    for curve in curves:
        sizes = [p[0] for p in curve["points"]]
        assert sizes == sorted(set(sizes)) and len(sizes) >= 2
        assert all(isinstance(p[0], int) and isinstance(p[1], float) for p in curve["points"])
    assert fwd["points"][0] == [1024, 151.81]
    assert all_reduce["points"][4] == [16384, 195.4]
    assert "[16384, 195.40]" in text  # two decimals


def test_rendered_comments_list_dropped_sizes_and_derivation() -> None:
    text = render_curves(_legacy_curves())
    assert "# Dropped 16 sender-timed rows" in text
    assert "#   1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072" in text
    assert "#   262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432" in text
    assert "derived from send_1to0" in text
    assert "only rank 0 timed" in text
    for line in text.splitlines():
        assert (
            line.startswith(
                (
                    "#",
                    "[[",
                    "op",
                    "scope",
                    "nodes",
                    "ranks",
                    "src_node",
                    "dst_node",
                    "source",
                    "derived_below_bytes",
                    "points",
                    "  [",
                    "]",
                )
            )
            or not line
        )


def test_render_cluster_appends_to_base_and_parses() -> None:
    curves = _legacy_curves()
    base_text = BASE_CLUSTER.read_text(encoding="utf-8")
    check_base_cluster(base_text, curves, base_path="base.toml")
    text = render_cluster(base_text, curves, base_path="base.toml")
    assert text.startswith("# Generated by `tools/lab/lab.py curves`")
    assert base_text.rstrip("\n") in text
    parsed = tomllib.loads(text)
    assert parsed["schema_version"] == 1
    assert len(parsed["nodes"]) == 3
    assert len(parsed["collective_curves"]) == 3


def test_base_cluster_checks() -> None:
    curves = _legacy_curves()
    with pytest.raises(CurveError, match="already has"):
        check_base_cluster("[[collective_curves]]\nop = 'x'\n", curves, base_path="b")
    with pytest.raises(CurveError, match=r"nodes \[1\] missing"):
        check_base_cluster("[[nodes]]\nid = 0\n", curves, base_path="b")
    with pytest.raises(CurveError, match="not valid TOML"):
        check_base_cluster("[[nodes]\n", curves, base_path="b")


# -- new format ---------------------------------------------------------------------


def _meta(world: int, nodes: list[int | None]) -> str:
    ranks = [{"rank": r, "hostname": f"node{r}", "sim_node_id": n} for r, n in enumerate(nodes)]
    return json.dumps({"record": "lab.collective_meta.v1", "world_size": world, "dtype": "bfloat16",
                       "ranks": ranks})  # fmt: skip


def _row(op: str, size: int, median: float, world: int, timed_on: str) -> str:
    return json.dumps({"record": "lab.collective.v1", "op": op, "bytes": size, "dtype": "bfloat16",
                       "world_size": world, "timed_on": timed_on, "median_us": median,
                       "p10_us": median, "p90_us": median, "iters": 30})  # fmt: skip


def _latency(size: int, bandwidth: float = 1000.0, floor: float = 50.0) -> float:
    return floor + size / bandwidth


def _new_format(world: int = 3, nodes: list[int | None] | None = None) -> list[str]:
    sizes = SIZES[:6]
    lines = [_meta(world, nodes if nodes is not None else list(range(world)))]
    for size in sizes:
        lines.append(_row("all_gather", size, _latency(size), world, "max_rank"))
        for src in range(world):
            for dst in range(world):
                if src != dst:
                    lines.append(_row(f"send_{src}to{dst}", size, _latency(size), world, "receiver"))
    lines.append(json.dumps({"record": "lab.done.v1", "rows": len(lines) - 1}))
    return lines


def test_new_format_uses_meta_mapping_and_world_size() -> None:
    data = parse_bench_lines(_new_format(nodes=[4, 5, 6]), source="new")
    assert not data.legacy and data.done
    rank_nodes = rank_nodes_from_meta(data.meta)
    assert rank_nodes == (4, 5, 6)
    curves = build_curves(data, CurveOptions(rank_nodes=rank_nodes), source="new")
    assert curves.dropped == ()
    (gather,) = curves.collectives
    assert gather.kind is CollectiveKind.ALL_GATHER and gather.ranks == 3
    assert gather.scope == NodeGroupScope((4, 5, 6))
    assert gather.timed_on is TimedOn.MAX_RANK
    assert len(curves.send_recv) == 6
    assert {(c.src_node, c.dst_node) for c in curves.send_recv} == {
        (a, b) for a in (4, 5, 6) for b in (4, 5, 6) if a != b
    }
    assert all(c.low_end == MeasuredLowEnd() for c in curves.send_recv)
    parsed = tomllib.loads(render_curves(curves))
    assert "source" not in parsed["collective_curves"][0]


def test_receiver_timed_rows_are_never_dropped() -> None:
    # Absurdly fast small sends, but timed on the receiver: kept.
    lines = [_meta(2, [0, 1])]
    for size in SIZES[:6]:
        lines.append(_row("send_0to1", size, 1.0 if size < 8192 else _latency(size), 2, "receiver"))
    data = parse_bench_lines(lines, source="r")
    curves = build_curves(data, CurveOptions(rank_nodes=(0, 1)), source="r")
    assert curves.dropped == ()
    assert len(curves.send_recv[0].points) == 6


def test_sender_timed_without_reverse_is_truncated(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = [_meta(2, [0, 1])]
    for size in SIZES[:8]:
        fast = size <= 8192
        lines.append(_row("send_0to1", size, 5.0 if fast else _latency(size), 2, "sender"))
    lines.append(_row("all_reduce", 1024, 100.0, 2, "max_rank"))
    lines.append(_row("all_reduce", 2048, 110.0, 2, "max_rank"))
    data = parse_bench_lines(lines, source="t")
    monkeypatch.setattr(logging.getLogger("lab"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="lab"):
        curves = build_curves(data, CurveOptions(rank_nodes=(0, 1)), source="t")
    curve = curves.send_recv[0]
    assert [d.message_bytes for d in curves.dropped] == SIZES[:4]
    assert curve.low_end == TruncatedLowEnd(smallest_bytes=16384, floor_bytes=1024)
    assert curve.derived_below_bytes is None
    assert [p.message_bytes for p in curve.points] == SIZES[4:8]
    assert any("truncated" in record.getMessage() for record in caplog.records)
    assert any(getattr(record, "message_bytes", None) == 1024 for record in caplog.records)
    text = render_curves(curves)
    assert "simulator extrapolates its floor" in text
    assert "derived_below_bytes" not in text


def test_reverse_direction_without_smaller_sizes_truncates() -> None:
    lines = [_meta(2, [0, 1])]
    for size in SIZES[:6]:
        lines.append(_row("send_0to1", size, 5.0 if size <= 4096 else _latency(size), 2, "sender"))
    for size in SIZES[4:6]:
        lines.append(_row("send_1to0", size, _latency(size), 2, "receiver"))
    curves = build_curves(parse_bench_lines(lines, source="x"), CurveOptions(rank_nodes=(0, 1)), source="x")
    assert isinstance(_send(curves, 0, 1).low_end, TruncatedLowEnd)


def test_marginal_bandwidth_falls_back_to_largest_row() -> None:
    # Forward kept rows have a non-positive time delta: fall back to b/t of the largest row.
    lines = [_meta(2, [0, 1])]
    fwd = {1024: 1.0, 2048: 1.0, 4096: 1.0, 8192: 400.0, 16384: 390.0}
    for size, median in fwd.items():
        lines.append(_row("send_0to1", size, median, 2, "sender"))
    for size in fwd:
        lines.append(_row("send_1to0", size, _latency(size), 2, "receiver"))
    curves = build_curves(parse_bench_lines(lines, source="x"), CurveOptions(rank_nodes=(0, 1)), source="x")
    low_end = _send(curves, 0, 1).low_end
    assert isinstance(low_end, DerivedLowEnd)
    assert low_end.below_bytes == 8192
    assert low_end.forward_bandwidth == pytest.approx(16384 / 390.0)


# -- errors -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([_row("all_reduce", 1024, 0.0, 2, "max_rank")], "non-positive median"),
        ([_row("all_reduce", 1024, -3.0, 2, "max_rank")], "non-positive median"),
        ([_row("all_reduce", 0, 3.0, 2, "max_rank")], "non-positive message size"),
        ([_row("broadcastx", 1024, 3.0, 2, "max_rank")], "unknown op"),
        ([_row("send_1to1", 1024, 3.0, 2, "receiver")], "same source and destination"),
        ([_meta(2, [0, 1]), json.dumps({"record": "lab.done.v1"})], "no collective rows"),
    ],
)
def test_bad_rows_raise_curve_error(lines: list[str], message: str) -> None:
    with pytest.raises(CurveError, match=message):
        parse_bench_lines(lines, source="bad")


def test_duplicate_sizes_raise_curve_error() -> None:
    lines = [_row("all_reduce", 1024, 3.0, 2, "max_rank"), _row("all_reduce", 1024, 4.0, 2, "max_rank")]
    with pytest.raises(CurveError, match="duplicate message sizes"):
        build_curves(parse_bench_lines(lines, source="d"), CurveOptions(rank_nodes=(0, 1)), source="d")


def test_unparseable_lines_raise_result_parse_error() -> None:
    with pytest.raises(ResultParseError, match="not JSON"):
        parse_bench_lines(["{oops"], source="x")
    with pytest.raises(ResultParseError, match="unexpected record"):
        parse_bench_lines([json.dumps({"record": "lab.static_batch.v1"})], source="x")


def test_mixed_legacy_and_new_rows_rejected() -> None:
    legacy = LEGACY.read_text(encoding="utf-8").splitlines()[0]
    with pytest.raises(CurveError, match="mixes"):
        parse_bench_lines([legacy, _row("all_reduce", 2048, 3.0, 2, "max_rank")], source="m")


def test_rank_node_mapping_errors() -> None:
    data = load_bench(LEGACY)
    with pytest.raises(CurveError, match="lists 3 ranks"):
        build_curves(data, CurveOptions(rank_nodes=(0, 1, 2)), source="x")
    with pytest.raises(CurveError, match="same node"):
        build_curves(data, CurveOptions(rank_nodes=(0, 0), scope=ScopeKind.INTER_NODE), source="x")
    with pytest.raises(CurveError, match="at least 2 distinct nodes"):
        build_curves(data, CurveOptions(rank_nodes=(0, 0)), source="x")


def test_parse_rank_nodes() -> None:
    assert parse_rank_nodes("0,1") == (0, 1)
    assert parse_rank_nodes(" 2 , 0 ") == (2, 0)
    for bad in ("", "0,,1", "a,b", "-1,0"):
        with pytest.raises(CurveError):
            parse_rank_nodes(bad)


def test_rank_nodes_from_meta_requires_every_rank() -> None:
    assert rank_nodes_from_meta(None) is None
    data: BenchData = parse_bench_lines(_new_format(world=2, nodes=[0, None]), source="x")
    assert rank_nodes_from_meta(data.meta) is None


# -- CLI ------------------------------------------------------------------------------


def test_cli_prints_section(capsys: pytest.CaptureFixture[str]) -> None:
    assert lab.main(["curves", str(LEGACY), "--rank-nodes", "0,1", "--source", "S"]) == 0
    out = capsys.readouterr().out
    parsed = tomllib.loads(out)
    assert len(parsed["collective_curves"]) == 3
    assert {c["source"] for c in parsed["collective_curves"]} == {"S"}


def test_cli_writes_base_plus_curves(tmp_path: Path) -> None:
    out = tmp_path / "cluster.toml"
    code = lab.main(
        ["curves", str(LEGACY), "--rank-nodes", "0,1", "--base-cluster", str(BASE_CLUSTER), "--out", str(out)]
    )
    assert code == 0
    parsed = tomllib.loads(out.read_text(encoding="utf-8"))
    assert len(parsed["nodes"]) == 3 and len(parsed["collective_curves"]) == 3
    assert parsed["collective_curves"][0]["source"] == "tools/lab/tests/fixtures/nccl_curve_legacy.jsonl"


def test_cli_rank_nodes_from_meta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "new.jsonl"
    path.write_text("\n".join(_new_format(world=2, nodes=[1, 0])) + "\n", encoding="utf-8")
    assert lab.main(["curves", str(path)]) == 0
    parsed = tomllib.loads(capsys.readouterr().out)
    sends = [c for c in parsed["collective_curves"] if c["op"] == "send_recv"]
    assert [(c["src_node"], c["dst_node"]) for c in sends] == [(1, 0), (0, 1)]


def test_cli_missing_mapping_exits_with_curve_error_code(tmp_path: Path) -> None:
    assert CurveError.exit_code == 9
    assert lab.main(["curves", str(LEGACY)]) == CurveError.exit_code
    path = tmp_path / "x.jsonl"
    path.write_text(_row("all_reduce", 1024, 0.0, 2, "max_rank") + "\n", encoding="utf-8")
    assert lab.main(["curves", str(path), "--rank-nodes", "0,1"]) == CurveError.exit_code
