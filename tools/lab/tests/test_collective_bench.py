"""Helpers of remote/collective_bench.py (importable without torch)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from types import ModuleType

import pytest

from tests.conftest import TOOL_DIR


def _load() -> ModuleType:
    path = TOOL_DIR / "remote" / "collective_bench.py"
    spec = importlib.util.spec_from_file_location("collective_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load()


def test_module_imports_without_torch() -> None:
    # torch is imported inside main(), so loading the module must not pull it in.
    assert "torch" not in sys.modules
    assert bench.COLLECTIVE_RECORD == "lab.collective.v1"
    assert bench.META_RECORD == "lab.collective_meta.v1"
    assert bench.DONE_RECORD == "lab.done.v1"


def test_message_sizes_double_inclusive() -> None:
    assert bench.message_sizes(1024, 8192) == [1024, 2048, 4096, 8192]
    assert bench.message_sizes(1024, 5000) == [1024, 2048, 4096]
    assert len(bench.message_sizes(1024, 256 * 1024 * 1024)) == 19
    with pytest.raises(ValueError):
        bench.message_sizes(4096, 1024)


def test_send_pairs_cover_every_ordered_pair() -> None:
    assert bench.send_pairs(2) == [(0, 1), (1, 0)]
    assert len(bench.send_pairs(4)) == 12


def test_expand_ops() -> None:
    assert bench.expand_ops(["all_reduce", "send_recv"], 2) == ["all_reduce", "send_0to1", "send_1to0"]
    assert bench.expand_ops(["send_recv"], 3) == [
        "send_0to1",
        "send_0to2",
        "send_1to0",
        "send_1to2",
        "send_2to0",
        "send_2to1",
    ]
    with pytest.raises(ValueError):
        bench.expand_ops(["broadcast"], 2)


def test_parse_ops() -> None:
    assert bench.parse_ops("all_reduce, all_gather") == ["all_reduce", "all_gather"]
    for bad in ("", "all_reduce,all_reduce", "nope"):
        with pytest.raises(ValueError):
            bench.parse_ops(bad)


def test_elements_for() -> None:
    assert bench.elements_for("all_reduce", 1024, 2, 2) == 512
    assert bench.elements_for("all_reduce", 1, 2, 2) == 1
    # all_to_all splits its buffer evenly across ranks.
    assert bench.elements_for("all_to_all", 1024, 2, 3) == 510
    assert bench.elements_for("all_to_all", 2, 2, 4) == 4


def test_iters_for_large_messages() -> None:
    assert bench.iters_for(16 * 1024 * 1024, 30) == 30
    assert bench.iters_for(32 * 1024 * 1024, 30) == 10
    assert bench.iters_for(32 * 1024 * 1024, 12) == 8


def test_summarize_matches_original_indexing() -> None:
    samples = [float(v) for v in range(30, 0, -1)]
    assert bench.summarize(samples) == {"median_us": 15.5, "p10_us": 4.0, "p90_us": 28.0}
    with pytest.raises(ValueError):
        bench.summarize([])


def test_per_iteration_max() -> None:
    assert bench.per_iteration_max([[1.0, 5.0, 2.0], [3.0, 1.0, 2.5]]) == [3.0, 5.0, 2.5]
    with pytest.raises(ValueError):
        bench.per_iteration_max([[1.0], [1.0, 2.0]])


def test_collective_row_uses_slowest_rank_per_iteration() -> None:
    row = bench.result_row("all_reduce", 1024, "bfloat16", 2, [[10.0, 30.0], [20.0, 10.0]], 2)
    assert row == {
        "record": "lab.collective.v1",
        "op": "all_reduce",
        "bytes": 1024,
        "dtype": "bfloat16",
        "world_size": 2,
        "timed_on": "max_rank",
        "median_us": 25.0,
        "p10_us": 20.0,
        "p90_us": 30.0,
        "iters": 2,
    }


def test_send_row_uses_receiver_samples() -> None:
    sender, receiver = [1.0, 1.0, 1.0], [100.0, 120.0, 110.0]
    row = bench.result_row("send_0to1", 4096, "bfloat16", 2, [sender, receiver], 3)
    assert row["timed_on"] == "receiver"
    assert (row["src_rank"], row["dst_rank"]) == (0, 1)
    assert row["median_us"] == 110.0
    reverse = bench.result_row("send_1to0", 4096, "bfloat16", 2, [receiver, sender], 3)
    assert reverse["median_us"] == 110.0


def test_meta_record_sorted_by_rank() -> None:
    record = bench.meta_record(2, "bfloat16", [{"rank": 1, "sim_node_id": 1}, {"rank": 0, "sim_node_id": 0}])
    assert record["record"] == "lab.collective_meta.v1"
    assert [r["rank"] for r in record["ranks"]] == [0, 1]


def _args(**overrides: object) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(bench.build_parser().parse_args([])), **overrides})


def test_rank_and_world_from_args_or_env() -> None:
    assert bench.resolve_rank_world(_args(), {"RANK": "1", "WORLD_SIZE": "2"}) == (1, 2)
    assert bench.resolve_rank_world(_args(rank=0, world_size=3), {"RANK": "1", "WORLD_SIZE": "2"}) == (0, 3)
    for env in ({}, {"RANK": "2", "WORLD_SIZE": "2"}, {"RANK": "0", "WORLD_SIZE": "1"}):
        with pytest.raises(ValueError):
            bench.resolve_rank_world(_args(), env)


def test_sim_node_id_from_args_or_env() -> None:
    assert bench.resolve_sim_node_id(_args(), {}) is None
    assert bench.resolve_sim_node_id(_args(), {"SIM_NODE_ID": "3"}) == 3
    assert bench.resolve_sim_node_id(_args(sim_node_id=1), {"SIM_NODE_ID": "3"}) == 1


def test_parser_defaults_match_original_sweep() -> None:
    args = bench.build_parser().parse_args([])
    assert (args.min_bytes, args.max_bytes, args.iters, args.warmup) == (1024, 256 * 1024 * 1024, 30, 5)
    assert args.dtype == "bfloat16"
    assert args.ops == "all_reduce,all_gather,reduce_scatter,all_to_all,send_recv"
