"""Capture the current reproduction machine and per-environment CUDA versions."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from activebench.runtime import conda_python

ENVIRONMENTS = ["habitat", "habitat-gs", "bencheval", "r3con", "magician", "fisherrf", "gavis", "gleam"]


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=90).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--environment", choices=ENVIRONMENTS, action="append")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/system-info.json")
    args = ap.parse_args()
    cpu = next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                if s.startswith("model name")), platform.processor())
    mem = next(s for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemTotal:"))
    os_release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    result = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="Hardware and software installed on the reproduction machine at capture time",
        os=os_release.get("PRETTY_NAME", "").strip('"'), kernel=platform.release(),
        architecture=platform.machine(), cpu=cpu, logical_cpus=os.cpu_count(),
        ram_bytes=int(mem.split()[1])*1024, environments={})
    try:
        smi = command(["nvidia-smi"])
        result["nvidia_driver_cuda_max"] = re.search(r"CUDA Version:\s*([\d.]+)", smi).group(1)
        result["gpus_csv"] = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"])
    except (OSError, subprocess.SubprocessError) as exc:
        result["gpu_error"] = str(exc)
    inspector = '''import json,platform,importlib.metadata as metadata
import torch
packages={}
for name in ['numpy','torch','torchvision','gsplat','habitat-gs']:
 try: packages[name]=metadata.version(name)
 except metadata.PackageNotFoundError: pass
print(json.dumps(dict(python=platform.python_version(),packages=packages,
 pytorch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),
 cudnn=torch.backends.cudnn.version())))'''
    for name in args.environment or ENVIRONMENTS:
        python = Path(conda_python(name))
        try:
            value = json.loads(command([str(python), "-c", inspector]))
            nvcc = python.parent / "nvcc"
            value["nvcc"] = command([str(nvcc), "--version"]) if nvcc.exists() else None
            result["environments"][name] = value
            print(name, value["python"], "torch", value["packages"]["torch"], "CUDA", value["pytorch_cuda"], flush=True)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result["environments"][name] = dict(error=str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result["nvcc_note"] = "Only <environment>/bin/nvcc is queried; null does not imply absence of system or other compilers."
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if any("error" in v for v in result["environments"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
