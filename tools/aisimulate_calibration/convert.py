#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb==1.5.5",
#     "numpy==2.5.3",
# ]
# ///
"""Compose AISimulate measured per-op GPU tables into an inference-sim calibration profile.

AISimulate (Apache-2.0, https://github.com/ai-dynamo/AISimulate) ships measured
per-operation GPU performance tables as Parquet.  This tool walks a dense
Llama-style decoder exactly the way AISimulate's own estimator does, composes
per-phase prefill/decode latencies from those measured op tables, fits the
phase-level linear basis that inference-sim consumes, and emits a calibration
profile TOML.

See docs/features/aisimulate_calibration.md for the full op-walk derivation,
the interpolation rules, and the documented deviations.

Run it with `uv run tools/aisimulate_calibration/convert.py`; the supporting
modules live in the sibling `converter/` package.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))

from converter.errors import (  # noqa: E402
    ConfigurationError,
    ConverterError,
    SelfTestError,
)
from converter.logging_setup import configure_logging  # noqa: E402
from converter.pipeline import RunPlan, run  # noqa: E402
from converter.sources import (  # noqa: E402
    csv_sources,
    environment_hash,
    load_gpu_spec,
    load_gpu_spec_json,
    parquet_sources,
    read_git_commit,
)
from converter.spec import (  # noqa: E402
    DTYPE_BYTES,
    MODEL_PRESETS,
    GridSpec,
    ModelSpec,
    Provenance,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def int_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated int list: {text}"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


DEFAULT_BATCH: Final = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_PROMPT: Final = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
DEFAULT_DECODE: Final = (32, 128, 512, 1024)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convert.py",
        description=(
            "Compose AISimulate measured per-op GPU tables into an inference-sim "
            "calibration profile."
        ),
    )
    parser.add_argument(
        "--aisimulate-dir", type=Path, help="root of an AISimulate clone"
    )
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--version", default="0.24.0")
    parser.add_argument(
        "--model-preset", choices=sorted(MODEL_PRESETS), default="llama-3.1-70b"
    )
    parser.add_argument(
        "--model-name", help="profile `model` string (defaults to the preset)"
    )
    parser.add_argument("--layers", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--attention-heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--parameters-gb", type=float)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-dtype", default="bfloat16")
    parser.add_argument("--tp", type=int_list, default=(2, 4, 8))
    parser.add_argument("--batch-sizes", type=int_list, default=DEFAULT_BATCH)
    parser.add_argument("--prompt-tokens", type=int_list, default=DEFAULT_PROMPT)
    parser.add_argument("--decode-tokens", type=int_list, default=DEFAULT_DECODE)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--benchmark-limit", type=int, default=24)
    parser.add_argument(
        "--output", type=Path, help="write the profile here (default: stdout)"
    )
    parser.add_argument(
        "--from-csv-dir", type=Path, help="read CSV tables + system.json instead"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="run the checked-in fixture"
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="with --self-test, rewrite fixtures/golden_profile.toml instead of diffing",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def model_from_args(args: argparse.Namespace) -> ModelSpec:
    preset = dict(MODEL_PRESETS[args.model_preset])
    overrides = {
        "layers": args.layers,
        "hidden_size": args.hidden_size,
        "attention_heads": args.attention_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "intermediate_size": args.intermediate_size,
        "vocab_size": args.vocab_size,
        "parameters_gb": args.parameters_gb,
    }
    preset.update({k: v for k, v in overrides.items() if v is not None})
    if args.dtype not in DTYPE_BYTES:
        raise ConfigurationError(
            f"unsupported --dtype {args.dtype!r}; known: {sorted(DTYPE_BYTES)}"
        )
    return ModelSpec(
        name=args.model_name or args.model_preset,
        layers=int(preset["layers"]),
        hidden_size=int(preset["hidden_size"]),
        attention_heads=int(preset["attention_heads"]),
        kv_heads=int(preset["kv_heads"]),
        head_dim=int(preset["head_dim"]),
        intermediate_size=int(preset["intermediate_size"]),
        vocab_size=int(preset["vocab_size"]),
        dtype=args.dtype,
        kv_dtype=args.kv_dtype,
        parameters_gb=float(preset["parameters_gb"]),
    )


def plan_from_args(args: argparse.Namespace) -> RunPlan:
    if args.from_csv_dir is not None:
        root = args.from_csv_dir
        sources = csv_sources(root)
        sources.require()
        gpu = load_gpu_spec_json(root / "system.json")
        commit = "csv-fixture"
    else:
        if args.aisimulate_dir is None:
            raise ConfigurationError(
                "--aisimulate-dir is required (or use --from-csv-dir)"
            )
        root = args.aisimulate_dir
        sources = parquet_sources(root, args.system, args.backend, args.version)
        sources.require()
        gpu = load_gpu_spec(
            root
            / "python/aisimulate/src/aisimulate_core/systems"
            / f"{args.system}.yaml"
        )
        commit = read_git_commit(root)
    return RunPlan(
        sources=sources,
        gpu=gpu,
        model=model_from_args(args),
        grid=GridSpec(
            batch_sizes=tuple(sorted(args.batch_sizes)),
            prompt_tokens=tuple(sorted(args.prompt_tokens)),
            decode_tokens=tuple(sorted(args.decode_tokens)),
            tensor_ranks=tuple(sorted(args.tp)),
        ),
        provenance=Provenance(
            system=args.system,
            backend=args.backend,
            version=args.version,
            commit=commit,
            date=dt.date.today().isoformat(),
            environment_hash=environment_hash(sources),
        ),
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        benchmark_limit=args.benchmark_limit,
    )


FIXTURE_DIR: Final = Path(__file__).resolve().parent / "fixtures"


def self_test_plan() -> tuple[RunPlan, Path]:
    fixture_path = FIXTURE_DIR / "fixture.json"
    if not fixture_path.is_file():
        raise ConfigurationError(f"missing self-test fixture: {fixture_path}")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    sources = csv_sources(FIXTURE_DIR)
    sources.require()
    model_payload = payload["model"]
    model = ModelSpec(
        name=model_payload["name"],
        layers=int(model_payload["layers"]),
        hidden_size=int(model_payload["hidden_size"]),
        attention_heads=int(model_payload["attention_heads"]),
        kv_heads=int(model_payload["kv_heads"]),
        head_dim=int(model_payload["head_dim"]),
        intermediate_size=int(model_payload["intermediate_size"]),
        vocab_size=int(model_payload["vocab_size"]),
        dtype=model_payload["dtype"],
        kv_dtype=model_payload["kv_dtype"],
        parameters_gb=float(model_payload["parameters_gb"]),
    )
    plan = RunPlan(
        sources=sources,
        gpu=load_gpu_spec_json(FIXTURE_DIR / "system.json"),
        model=model,
        grid=GridSpec(
            batch_sizes=tuple(payload["batch_sizes"]),
            prompt_tokens=tuple(payload["prompt_tokens"]),
            decode_tokens=tuple(payload["decode_tokens"]),
            tensor_ranks=tuple(payload["tensor_ranks"]),
        ),
        provenance=Provenance(
            system=payload["system"],
            backend=payload["backend"],
            version=payload["version"],
            commit=payload["commit"],
            date=payload["date"],
            environment_hash=environment_hash(sources),
        ),
        holdout_fraction=float(payload["holdout_fraction"]),
        seed=int(payload["seed"]),
        benchmark_limit=int(payload["benchmark_limit"]),
    )
    return plan, FIXTURE_DIR / "golden_profile.toml"


def run_self_test(logger: logging.Logger, write_golden: bool = False) -> None:
    plan, golden_path = self_test_plan()
    produced = run(plan, logger)
    if write_golden:
        golden_path.write_text(produced, encoding="utf-8")
        logger.info("wrote golden profile", extra={"path": str(golden_path)})
        return
    if not golden_path.is_file():
        raise SelfTestError(f"missing golden profile: {golden_path}")
    expected = golden_path.read_text(encoding="utf-8")
    if produced == expected:
        logger.info("self-test passed", extra={"golden": str(golden_path)})
        return
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            produced.splitlines(keepends=True),
            fromfile="golden_profile.toml",
            tofile="generated",
        )
    )
    sys.stderr.write(diff)
    raise SelfTestError("self-test output differs from the golden profile")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(args.verbose)
    try:
        if args.self_test:
            run_self_test(logger, write_golden=args.update_golden)
            return 0
        plan = plan_from_args(args)
        logger.info(
            "starting conversion",
            extra={
                "system": plan.provenance.system,
                "backend": plan.provenance.backend,
                "version": plan.provenance.version,
                "model": plan.model.name,
                "tensor_ranks": ",".join(str(t) for t in plan.grid.tensor_ranks),
            },
        )
        profile = run(plan, logger)
        if args.output is None:
            sys.stdout.write(profile)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(profile, encoding="utf-8")
            logger.info("wrote calibration profile", extra={"path": str(args.output)})
        return 0
    except ConverterError as error:
        logger.error(
            str(error), extra={"error": type(error).__name__, **error.details()}
        )
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
