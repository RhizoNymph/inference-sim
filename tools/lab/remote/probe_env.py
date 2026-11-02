"""Print one JSON line describing the vLLM environment and GPU on this node.

Runs inside the same environment the benchmark will use (container or venv).
The runner stores the line in the run manifest and refuses to start when
`vllm_version` differs from the experiment's pin. Missing optional packages
are reported as null rather than failing.

`gpu_state` records what changes GPU speed without changing software: the
power limit, the current and maximum SM clock, the clock event reasons
bitmask (non-zero when something such as a clock lock or power cap holds the
clock down), and every other process on the GPU. Lab runs taken a month apart
on unchanged software differed 1.7x in prefill speed with none of this
recorded (docs/validation_ledger.md, entry 26).
"""

import importlib
import json
import platform
import shutil
import socket
import subprocess
import sys


def version_of(module_name):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def gpu_facts():
    facts = {"gpus": [], "driver_version": None}
    if shutil.which("nvidia-smi") is None:
        return facts
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,pcie.link.gen.max",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in query.stdout.strip().splitlines():
        name, memory_mib, driver, pcie_gen = (part.strip() for part in line.split(","))
        facts["gpus"].append({"name": name, "memory_mib": int(memory_mib), "pcie_gen_max": pcie_gen})
        facts["driver_version"] = driver
    return facts


GPU_STATE_FIELDS = (
    "index",
    "power.limit",
    "power.default_limit",
    "clocks.max.sm",
    "clocks.sm",
    "clocks.mem",
    "clocks_event_reasons.active",
)


def _number(text, kind):
    """nvidia-smi value, or None for [N/A] / [Not Supported] / unparsable."""
    try:
        return kind(float(text)) if kind is int else kind(text)
    except ValueError:
        return None


def parse_gpu_state(stdout):
    states = []
    for line in stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(GPU_STATE_FIELDS):
            continue
        index, power, default_power, max_sm, sm, mem, reasons = parts
        states.append(
            {
                "index": _number(index, int),
                "power_limit_w": _number(power, float),
                "default_power_limit_w": _number(default_power, float),
                "max_sm_clock_mhz": _number(max_sm, int),
                "sm_clock_mhz": _number(sm, int),
                "mem_clock_mhz": _number(mem, int),
                "clock_event_reasons": reasons if reasons.startswith("0x") else None,
            }
        )
    return states


def parse_compute_apps(stdout):
    apps = []
    for line in stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        pid, name, memory = parts
        apps.append({"pid": _number(pid, int), "process_name": name, "used_memory_mib": _number(memory, int)})
    return apps


def _nvidia_smi(*args):
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=False).stdout


def gpu_state():
    if shutil.which("nvidia-smi") is None:
        return {"gpu_state": [], "gpu_compute_apps": []}
    state = _nvidia_smi(f"--query-gpu={','.join(GPU_STATE_FIELDS)}", "--format=csv,noheader,nounits")
    apps = _nvidia_smi("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
    return {"gpu_state": parse_gpu_state(state), "gpu_compute_apps": parse_compute_apps(apps)}


def main() -> int:
    torch_cuda = None
    nccl = None
    try:
        import torch

        torch_cuda = torch.version.cuda
        nccl = ".".join(str(part) for part in torch.cuda.nccl.version())
    except (ImportError, AttributeError, RuntimeError) as error:
        # Report, never fail the probe: the manifest records what is missing.
        torch_cuda = f"unavailable: {type(error).__name__}"
    record = {
        "record": "lab.probe.v1",
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "vllm_version": version_of("vllm"),
        "torch_version": version_of("torch"),
        "ray_version": version_of("ray"),
        "transformers_version": version_of("transformers"),
        "cuda_version": torch_cuda,
        "nccl_version": nccl,
        "kernel": platform.release(),
        **gpu_facts(),
        **gpu_state(),
    }
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
