"""Print one JSON line describing the vLLM environment and GPU on this node.

Runs inside the same environment the benchmark will use (container or venv).
The runner stores the line in the run manifest and refuses to start when
`vllm_version` differs from the experiment's pin. Missing optional packages
are reported as null rather than failing.
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
    }
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
