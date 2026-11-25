"""Shared fixtures: repo paths and the checked-in lab/spec files."""

from __future__ import annotations

from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
TOOL_DIR = TESTS_DIR.parent
REPO_ROOT = TOOL_DIR.parent.parent
FIXTURES = TESTS_DIR / "fixtures"
SPECS = TOOL_DIR / "specs"
BINARY = REPO_ROOT / "target" / "release" / "inference-sim"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def lab_toml() -> Path:
    return TOOL_DIR / "labs" / "rtx3090.toml"


def write_spec(tmp_path: Path, lab: Path, body: str, name: str = "spec.toml") -> Path:
    """Write an experiment spec whose `lab` points at `lab` (absolute)."""
    path = tmp_path / name
    path.write_text(body.replace("@LAB@", str(lab)), encoding="utf-8")
    return path


MINIMAL_STATIC = """
[experiment]
name = "t-static"
mode = "static-batch"
lab = "@LAB@"
nodes = ["node0"]
vllm_version = "0.29.0"

[model]
hf_id = "Qwen/Qwen2.5-7B-Instruct"
name = "qwen"
max_model_len = 4096

[model.sim]
layers = 28
hidden_size = 3584
attention_heads = 28
kv_heads = 4
vocab_size = 152064
parameters_gb = 15.23

[parallelism]
tp = 1
pp = 1

[engine]

[static_batch]
shapes = ["1x512", "8x512"]
decode_tokens = 128
"""
