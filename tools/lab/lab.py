#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Lab measurement-to-calibration pipeline for inference-sim.

Subcommands (see docs/features/lab_harness.md):

  run        execute an experiment spec on the lab nodes (or --dry-run to print
             every command) and collect results into lab-runs/<date>-<name>/
  sim        run the simulator over a spec's shapes / request rates
  calibrate  static-batch: default-calibration sweep, two-scalar fit,
             leave-one-shape-out validation, profile emission + load check
  validate   compare a measured run against the simulator under a calibration
             (default, explicit scalars, or a profile)
  report     assemble the markdown fragments in a run dir into report.md

Only the standard library is needed locally; the remote/ scripts run inside
the vLLM environment on the nodes.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))

from labharness.commands import build_plan
from labharness.errors import LabError, ResultParseError, RunDirError, SpecError
from labharness.evaluate import (
    calibrate_static,
    evaluate_serving,
    evaluate_static,
    verify_profile,
)
from labharness.logging_setup import configure_logging, get_logger
from labharness.profile import Provenance
from labharness.report import (
    fit_json,
    fit_lines,
    profile_check_lines,
    serving_json,
    serving_table,
    static_json,
    static_table,
)
from labharness.results import (
    load_bench_serve,
    load_static,
    parse_kv_cache_tokens,
)
from labharness.runner import Runner, dry_run
from labharness.simulate import (
    Calibration,
    DefaultCalibration,
    ProfileCalibration,
    ScalarCalibration,
    SimRunner,
    sweep_serving,
    sweep_static,
)
from labharness.spec import (
    Experiment,
    FixedBatch,
    LittlesLawBatch,
    ReferenceBatch,
    ServingWorkload,
    SimAdmission,
    StaticBatchWorkload,
    parse_experiment,
    rate_label,
)

TOOL_DIR: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = TOOL_DIR.parent.parent
DEFAULT_BINARY: Final = REPO_ROOT / "target" / "release" / "inference-sim"
DEFAULT_RUNS_ROOT: Final = REPO_ROOT / "lab-runs"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_calibration_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--default-calibration", action="store_true", help="simulator defaults")
    group.add_argument("--profile", type=Path, help="calibration profile TOML")
    group.add_argument(
        "--scalars",
        metavar="CE,DMBS",
        help="compute_efficiency,decode_memory_bandwidth_scale",
    )


def _add_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True, help="lab-runs/<run> directory")
    parser.add_argument(
        "--measured", help="static-batch results file inside --run-dir (default measured.jsonl)"
    )
    parser.add_argument("--tag", help="output name tag (default: experiment name)")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--reference-batch",
        help="serving only: override serving.sim_reference_batch (an integer or littles_law)",
    )
    parser.add_argument(
        "--admission",
        choices=[a.value for a in SimAdmission],
        help="serving only: override serving.sim_admission",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab.py", description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment on the lab (or print it with --dry-run)")
    run.add_argument("spec", type=Path)
    run.add_argument("--dry-run", action="store_true", help="print every command; touch nothing")
    run.add_argument("--date", default=dt.date.today().isoformat(), help="run date (YYYY-MM-DD)")
    run.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)

    sim = sub.add_parser("sim", help="simulate a spec's shapes or rates")
    _add_sim_args(sim)
    _add_calibration_args(sim, required=True)

    calibrate = sub.add_parser("calibrate", help="fit scalars from a static-batch run and emit a profile")
    _add_sim_args(calibrate)

    validate = sub.add_parser("validate", help="compare a measured run with the simulator")
    _add_sim_args(validate)
    _add_calibration_args(validate, required=True)

    report = sub.add_parser("report", help="assemble report.md from a run dir's fragments")
    report.add_argument("--run-dir", type=Path, required=True)
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_spec(args: argparse.Namespace) -> Experiment:
    """Parse the spec, applying a --reference-batch override for serving specs."""
    exp = parse_experiment(args.spec)
    reference_text = getattr(args, "reference_batch", None)
    admission_text = getattr(args, "admission", None)
    if reference_text is None and admission_text is None:
        return exp
    wl = exp.workload
    if not isinstance(wl, ServingWorkload):
        raise SpecError("--reference-batch/--admission only apply to serving specs", path=str(args.spec))
    match reference_text:
        case None:
            reference: ReferenceBatch = wl.sim_reference_batch
        case "littles_law":
            reference = LittlesLawBatch()
        case text if text.isdigit() and int(text) >= 1:
            reference = FixedBatch(int(text))
        case _:
            raise SpecError("--reference-batch must be a positive integer or littles_law", path="cli")
    admission = SimAdmission(admission_text) if admission_text else wl.sim_admission
    return dataclasses.replace(
        exp, workload=dataclasses.replace(wl, sim_reference_batch=reference, sim_admission=admission)
    )


def _calibration(args: argparse.Namespace) -> Calibration:
    if args.profile is not None:
        if not args.profile.is_file():
            raise SpecError("profile not found", path=str(args.profile))
        return ProfileCalibration(args.profile.resolve())
    if args.scalars is not None:
        try:
            ce, dmbs = (float(part) for part in args.scalars.split(","))
        except ValueError as error:
            raise SpecError("--scalars must be CE,DMBS", path="cli", field="scalars") from error
        return ScalarCalibration(ce, dmbs)
    return DefaultCalibration()


def _sim_runner(args: argparse.Namespace, exp: Experiment, tag: str) -> SimRunner:
    if not args.binary.is_file():
        raise SpecError("simulator binary not found; run `cargo build --release`", path=str(args.binary))
    return SimRunner(
        binary=args.binary.resolve(),
        cluster=exp.lab.cluster_toml,
        work_dir=args.run_dir.resolve() / "sim-work" / tag,
        concurrency=args.concurrency,
    )


def _run_dir(args: argparse.Namespace) -> Path:
    if not args.run_dir.is_dir():
        raise RunDirError("run directory does not exist", path=str(args.run_dir))
    return args.run_dir


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _measured_path(args: argparse.Namespace) -> Path:
    return _run_dir(args) / (args.measured or "measured.jsonl")


def _write(path: Path, text: str, what: str) -> None:
    path.write_text(text, encoding="utf-8")
    get_logger().info("wrote", extra={"what": what.replace(" ", "_"), "path": _repo_relative(path)})


def _json(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


def _provenance(exp: Experiment, run_dir: Path, measured: Path) -> Provenance:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", run_dir.resolve().name)
    date = match.group(1) if match else dt.date.today().isoformat()
    probe_path = run_dir / f"probe-{exp.head.name}.json"
    probe: dict[str, object] = {}
    if probe_path.is_file():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(measured.read_bytes()).hexdigest()

    def text(key: str) -> str | None:
        value = probe.get(key)
        return str(value) if value else None

    return Provenance(
        run_dir=_repo_relative(run_dir),
        date=date,
        backend_version=text("vllm_version") or exp.vllm_version,
        driver_version=text("driver_version"),
        cuda_version=text("cuda_version"),
        nccl_version=text("nccl_version"),
        environment_hash=f"sha256:{digest}",
    )


def _serving_measurements(exp: Experiment, wl: ServingWorkload, run_dir: Path):
    measured = [load_bench_serve(run_dir / f"{rate_label(rate)}.json") for rate in wl.request_rates]
    server_log = run_dir / "server.log"
    kv_tokens = (
        parse_kv_cache_tokens(server_log.read_text(encoding="utf-8")) if server_log.is_file() else None
    )
    return measured, kv_tokens


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        raise SpecError("--date must be YYYY-MM-DD", path="cli", field="date")
    exp = parse_experiment(args.spec)
    plan = build_plan(exp, date=args.date, runs_root=args.runs_root.resolve(), remote_src=TOOL_DIR / "remote")
    if args.dry_run:
        dry_run(plan, sys.stdout)
        return
    Runner(exp, plan, repo=REPO_ROOT).run()


async def cmd_sim(args: argparse.Namespace) -> None:
    exp = _load_spec(args)
    calibration = _calibration(args)
    tag = args.tag or exp.name
    runner = _sim_runner(args, exp, tag)
    run_dir = _run_dir(args)
    match exp.workload:
        case StaticBatchWorkload() as wl:
            rows = await sweep_static(
                runner, exp.model, exp.parallelism, wl.shapes, wl.decode_tokens, calibration
            )
            out = run_dir / f"sim-{tag}-{calibration.label}.jsonl"
            _write(out, "".join(json.dumps(r.to_json()) + "\n" for r in rows), "static sweep")
        case ServingWorkload() as wl:
            server_log = run_dir / "server.log"
            kv = (
                parse_kv_cache_tokens(server_log.read_text(encoding="utf-8"))
                if server_log.is_file()
                else None
            )
            results = await sweep_serving(runner, exp, wl, calibration, measured_kv_tokens=kv)
            out = run_dir / f"sim-{tag}-{calibration.label}.json"
            _write(out, _json([r.to_json() for r in results]), "serving sweep")


async def cmd_calibrate(args: argparse.Namespace) -> None:
    exp = parse_experiment(args.spec)
    if not isinstance(exp.workload, StaticBatchWorkload):
        raise SpecError("calibrate needs a static-batch spec", path=str(args.spec), field="experiment.mode")
    tag = args.tag or exp.name
    run_dir = _run_dir(args)
    measured_path = _measured_path(args)
    measured = load_static(measured_path)
    runner = _sim_runner(args, exp, tag)
    outcome = await calibrate_static(runner, exp, measured, _provenance(exp, run_dir, measured_path), tag)

    profile_path = run_dir / f"calibration_profile-{tag}.toml"
    _write(profile_path, outcome.profile_text, "calibration profile")
    check = await verify_profile(runner, exp, profile_path, outcome.fitted)

    payload = {
        "record": "lab.calibration.v1",
        "tag": tag,
        "spec": _repo_relative(exp.path),
        "measured": _repo_relative(measured_path),
        "fit": fit_json(outcome.fit),
        "folds": {label: fit_json(fold) for label, fold in outcome.folds},
        "evaluations": [static_json(e) for e in (outcome.default, outcome.fitted, outcome.loo)],
        "profile": _repo_relative(profile_path),
        "profile_check": {
            "profile_name": check.profile_name,
            "compute_efficiency": check.compute_efficiency,
            "decode_memory_bandwidth_scale": check.decode_memory_bandwidth_scale,
            "applicability_status": check.applicability_status,
            "max_prediction_drift_pct": check.max_prediction_drift_pct,
        },
    }
    _write(run_dir / f"calibration-{tag}.json", _json(payload), "calibration results")
    fragment = [
        f"## Calibration: {tag}",
        "",
        f"Spec `{_repo_relative(exp.path)}`, measurements `{_repo_relative(measured_path)}`.",
        "",
        *fit_lines(outcome.fit, outcome.folds),
        *profile_check_lines(check, _repo_relative(profile_path)),
        "",
    ]
    for evaluation in (outcome.default, outcome.fitted, outcome.loo):
        fragment += [*static_table(evaluation), ""]
    _write(run_dir / f"report-calibration-{tag}.md", "\n".join(fragment), "report fragment")


async def cmd_validate(args: argparse.Namespace) -> None:
    exp = _load_spec(args)
    calibration = _calibration(args)
    tag = args.tag or f"{exp.name}-{calibration.label}"
    run_dir = _run_dir(args)
    runner = _sim_runner(args, exp, tag)
    match exp.workload:
        case StaticBatchWorkload():
            measured_path = _measured_path(args)
            evaluation = await evaluate_static(
                runner,
                exp,
                load_static(measured_path),
                calibration,
                f"{tag}: {exp.name} under {calibration.label}",
            )
            payload = {
                "record": "lab.validation.v1",
                "tag": tag,
                "mode": "static-batch",
                **static_json(evaluation),
            }
            fragment = [f"## Validation: {tag}", "", f"Measurements `{_repo_relative(measured_path)}`.", ""]
            fragment += static_table(evaluation)
        case ServingWorkload() as wl:
            measured, kv_tokens = _serving_measurements(exp, wl, run_dir)
            evaluation = await evaluate_serving(
                runner, exp, measured, calibration, f"{tag}: {exp.name} under {calibration.label}",
                measured_kv_tokens=kv_tokens,
            )  # fmt: skip
            payload = {
                "record": "lab.validation.v1",
                "tag": tag,
                "mode": "serving",
                "kv_cache_tokens": wl.kv_cache_tokens or kv_tokens,
                **serving_json(evaluation),
            }
            fragment = [
                f"## Validation: {tag}",
                "",
                f"Measurements `{_repo_relative(run_dir)}/rate_*.json`.",
                "",
            ]
            fragment += serving_table(evaluation)
    payload["spec"] = _repo_relative(exp.path)
    _write(run_dir / f"validation-{tag}.json", _json(payload), "validation results")
    _write(run_dir / f"report-validation-{tag}.md", "\n".join(fragment) + "\n", "report fragment")


def cmd_report(args: argparse.Namespace) -> None:
    run_dir = _run_dir(args)
    fragments = sorted(run_dir.glob("report-calibration-*.md")) + sorted(
        run_dir.glob("report-validation-*.md")
    )
    if not fragments:
        raise ResultParseError("no report fragments; run calibrate/validate first", source=str(run_dir))
    header = [
        f"# Lab report: {run_dir.resolve().name}",
        "",
        "Generated by `tools/lab/lab.py report`. Errors are `(sim - measured) / measured`; "
        "positive means the simulator predicts slower than measured.",
        "",
    ]
    body = [fragment.read_text(encoding="utf-8").rstrip() + "\n" for fragment in fragments]
    _write(run_dir / "report.md", "\n".join(header) + "\n".join(body), "report")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = configure_logging(args.verbose)
    try:
        match args.command:
            case "run":
                cmd_run(args)
            case "sim":
                asyncio.run(cmd_sim(args))
            case "calibrate":
                asyncio.run(cmd_calibrate(args))
            case "validate":
                asyncio.run(cmd_validate(args))
            case "report":
                cmd_report(args)
        return 0
    except (LabError, ExceptionGroup) as raised:
        errors = _lab_errors(raised)
        if not errors:
            raise
        for error in errors:
            logger.error(str(error), extra={"error": type(error).__name__, **error.details()})
        return errors[0].exit_code


def _lab_errors(raised: BaseException) -> list[LabError]:
    """Flatten TaskGroup exception groups; non-LabErrors are re-raised by the caller."""
    match raised:
        case LabError():
            return [raised]
        case BaseExceptionGroup():
            nested = [_lab_errors(inner) for inner in raised.exceptions]
            if any(not found for found in nested):
                return []
            return [error for found in nested for error in found]
        case _:
            return []


if __name__ == "__main__":
    raise SystemExit(main())
