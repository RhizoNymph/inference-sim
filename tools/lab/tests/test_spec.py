from __future__ import annotations

import math
from pathlib import Path

import pytest

from labharness.errors import SpecError
from labharness.spec import (
    ContainerUser,
    DockerLaunch,
    FixedBatch,
    LaunchKind,
    LittlesLawBatch,
    Mode,
    ServingWorkload,
    Shape,
    StaticBatchWorkload,
    VenvLaunch,
    parse_experiment,
    parse_lab,
    rate_label,
    rate_text,
)
from tests.conftest import MINIMAL_STATIC, SPECS, write_spec


def test_lab_parses_node_quirks(lab_toml: Path) -> None:
    lab = parse_lab(lab_toml)
    assert [n.name for n in lab.nodes] == ["node0", "node1", "node2"]
    assert lab.hf_home == "$HOME/.cache/huggingface"
    assert lab.socket_ifname == "bond0"
    node0 = lab.node("node0")
    assert node0.default_launch is LaunchKind.DOCKER
    docker = node0.launch(LaunchKind.DOCKER)
    assert isinstance(docker, DockerLaunch)
    assert docker.image == "vllm/vllm-openai:v0.29.0"
    assert docker.runtime_args == ("--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all")
    assert docker.container_user is ContainerUser.USER
    assert isinstance(node0.launch(LaunchKind.VENV), VenvLaunch)
    assert lab.node("node1").default_launch is LaunchKind.VENV
    assert LaunchKind.DOCKER not in lab.node("node1").launches
    assert lab.cluster_toml.is_file()


def test_unknown_node_is_rejected(lab_toml: Path) -> None:
    with pytest.raises(SpecError, match="no node named"):
        parse_lab(lab_toml).node("node9")


def test_checked_in_static_specs() -> None:
    pp1 = parse_experiment(SPECS / "rtx3090_qwen7b_static_pp1.toml")
    assert pp1.mode is Mode.STATIC_BATCH
    assert pp1.launch is LaunchKind.DOCKER
    assert not pp1.multi_node
    assert isinstance(pp1.workload, StaticBatchWorkload)
    assert pp1.workload.shapes == (
        Shape(1, 512),
        Shape(1, 2048),
        Shape(8, 512),
        Shape(8, 2048),
        Shape(32, 512),
    )
    assert pp1.model.sim.ffn_hidden_size == 18944
    assert pp1.model.sim.head_dim == 128

    pp2 = parse_experiment(SPECS / "rtx3090_qwen7b_static_pp2.toml")
    assert pp2.parallelism.pp == 2
    assert pp2.launch is LaunchKind.VENV
    assert [n.name for n in pp2.nodes] == ["node0", "node1"]
    assert pp2.head.name == "node0"


def test_checked_in_serving_spec() -> None:
    exp = parse_experiment(SPECS / "rtx3090_qwen7b_serving_pp1.toml")
    assert exp.mode is Mode.SERVING
    wl = exp.workload
    assert isinstance(wl, ServingWorkload)
    assert wl.request_rates[:6] == (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)
    assert math.isinf(wl.request_rates[-1])
    assert wl.num_prompts == 200
    assert wl.sim_reference_batch == FixedBatch(1)
    assert exp.engine.max_num_batched_tokens == 2048


def test_rate_text_and_label() -> None:
    assert rate_text(2.0) == "2"
    assert rate_text(0.5) == "0.5"
    assert rate_text(math.inf) == "inf"
    assert rate_label(10.0) == "rate_10"
    assert rate_label(math.inf) == "rate_inf"


def test_shape_parse() -> None:
    assert Shape.parse("8x512", path="x") == Shape(8, 512)
    assert Shape.parse(" 32 x 512 ", path="x").label == "32x512"
    for bad in ("8*512", "0x512", "x512", "8x"):
        with pytest.raises(SpecError):
            Shape.parse(bad, path="x")


def test_minimal_spec_defaults(tmp_path: Path, lab_toml: Path) -> None:
    exp = parse_experiment(write_spec(tmp_path, lab_toml, MINIMAL_STATIC))
    assert exp.engine.gpu_memory_utilization == 0.85
    assert exp.engine.enable_prefix_caching is False
    assert exp.runner.poll_interval_s == 15.0
    assert isinstance(exp.workload, StaticBatchWorkload)
    assert exp.workload.iters == 5
    assert exp.source_text.startswith("\n[experiment]")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (('mode = "static-batch"', 'mode = "serving"'), "requires exactly one matching workload"),
        (('mode = "static-batch"', 'mode = "offline"'), "must be one of"),
        (("pp = 1", "pp = 2"), "tp\\*pp = 2 but the listed nodes have 1 GPUs"),
        (('"8x512"', '"128x512"'), "exceeds engine.max_num_seqs"),
        (('"8x512"', '"8x4000"'), "exceeds max_model_len"),
        (('"8x512"', '"1x512"'), "duplicate shapes"),
        (('nodes = ["node0"]', 'nodes = ["node7"]'), "no node named"),
        (('nodes = ["node0"]', 'nodes = ["node0", "node0"]'), "duplicate nodes"),
        (("hidden_size = 3584", "hidden_size = 3585"), "divisible"),
        (("[engine]", "[engine]\ngpu_memory_utilization = 1.5"), "<= 1.0"),
        (('vllm_version = "0.29.0"', ""), "vllm_version"),
    ],
)
def test_invalid_specs(tmp_path: Path, lab_toml: Path, edit: tuple[str, str], message: str) -> None:
    body = MINIMAL_STATIC.replace(*edit)
    with pytest.raises(SpecError, match=message):
        parse_experiment(write_spec(tmp_path, lab_toml, body))


def test_multi_node_requires_venv(tmp_path: Path, lab_toml: Path) -> None:
    body = MINIMAL_STATIC.replace(
        'nodes = ["node0"]', 'nodes = ["node0", "node1"]\nlaunch = "docker"'
    ).replace("pp = 1", "pp = 2")
    with pytest.raises(SpecError, match="node1 has no docker launch"):
        parse_experiment(write_spec(tmp_path, lab_toml, body))
    auto = MINIMAL_STATIC.replace('nodes = ["node0"]', 'nodes = ["node0", "node1"]').replace(
        "pp = 1", "pp = 2"
    )
    with pytest.raises(SpecError, match="mixed launch methods"):
        parse_experiment(write_spec(tmp_path, lab_toml, auto))


def test_multi_node_docker_rejected_even_when_available(tmp_path: Path) -> None:
    lab = tmp_path / "lab.toml"
    cluster = tmp_path / "cluster.toml"
    cluster.write_text("schema_version = 1\n", encoding="utf-8")
    lab.write_text(
        """
[lab]
name = "two-docker"
hardware = "x"
remote_root = "$HOME/lab"
hf_home = "$HOME/.cache/huggingface"
socket_ifname = "eth0"
cluster_toml = "cluster.toml"
[[nodes]]
name = "a"
address = "10.0.0.1"
sim_node_id = 0
[nodes.docker]
image = "img"
[[nodes]]
name = "b"
address = "10.0.0.2"
sim_node_id = 1
[nodes.docker]
image = "img"
""",
        encoding="utf-8",
    )
    body = MINIMAL_STATIC.replace('nodes = ["node0"]', 'nodes = ["a", "b"]').replace("pp = 1", "pp = 2")
    with pytest.raises(SpecError, match="multi-node experiments run Ray from native venvs"):
        parse_experiment(write_spec(tmp_path, lab, body))


SERVING = (
    MINIMAL_STATIC.replace('mode = "static-batch"', 'mode = "serving"').split("[static_batch]")[0]
    + """
[serving]
input_len = 512
output_len = 128
request_rates = [2, "inf"]
"""
)


def test_serving_spec_defaults_and_overrides(tmp_path: Path, lab_toml: Path) -> None:
    exp = parse_experiment(write_spec(tmp_path, lab_toml, SERVING))
    wl = exp.workload
    assert isinstance(wl, ServingWorkload)
    assert wl.request_rates == (2.0, math.inf)
    assert wl.num_prompts == 200 and wl.port == 8000 and wl.burstiness == 1.0
    assert wl.sim_reference_batch == LittlesLawBatch()
    assert wl.kv_cache_tokens is None

    fixed = SERVING + "sim_reference_batch = 16\nkv_cache_tokens = 82864\n"
    wl2 = parse_experiment(write_spec(tmp_path, lab_toml, fixed, "b.toml")).workload
    assert isinstance(wl2, ServingWorkload)
    assert wl2.sim_reference_batch == FixedBatch(16)
    assert wl2.kv_cache_tokens == 82864


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("request_rates = [0]", "invalid rate"),
        ('request_rates = ["fast"]', "invalid rate"),
        ("request_rates = []", "non-empty"),
        ("request_rates = [2, 2]", "duplicate request rates"),
        ('sim_reference_batch = "auto"', "sim_reference_batch"),
    ],
)
def test_invalid_serving(tmp_path: Path, lab_toml: Path, line: str, message: str) -> None:
    body = SERVING.replace(
        'request_rates = [2, "inf"]',
        line if line.startswith("request_rates") else "request_rates = [2]\n" + line,
    )
    with pytest.raises(SpecError, match=message):
        parse_experiment(write_spec(tmp_path, lab_toml, body))
