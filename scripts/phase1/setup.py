"""Pinned source/environment setup. Prints commands unless --execute is given.

Create environments at NEW prefixes; existing environments are never replaced.
System GPU drivers, compiler compatibility and licensed dataset acquisition
remain machine-specific. See docs/SETUP.md for the validated machine.
"""
from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCKS = ROOT / "phase1/dependencies"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, help="New directory for pinned external checkouts")
    ap.add_argument("--environment", choices=["habitat", "habitat-gs", "bencheval", "r3con", "magician", "fisherrf", "gavis", "gleam"])
    ap.add_argument("--prefix", type=Path, help="New conda environment prefix")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    env = dict(os.environ)

    def run(command, cwd=None):
        print((f"(cwd={cwd}) " if cwd else "") + shlex.join(map(str, command)), flush=True)
        if args.execute:
            subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)

    sources = json.loads((LOCKS / "sources.json").read_text())
    if args.sources:
        args.sources = args.sources.resolve()
        if not args.environment:
            if args.sources.exists():
                ap.error("--sources must be a new directory when fetching checkouts")
            for name, record in sources.items():
                path = args.sources / name
                run(["git", "clone", record["url"], path])
                run(["git", "checkout", "--detach", record["commit"]], cwd=path)
                run(["git", "submodule", "update", "--init", "--recursive"], cwd=path)
                if record["local_source_patch"]:
                    run(["git", "apply", LOCKS / record["local_source_patch"]], cwd=path)
            return
    if not args.environment or not args.prefix:
        ap.error("select --sources to fetch code, or --environment and --prefix to build an env")
    prefix = args.prefix.resolve()
    if prefix.exists():
        ap.error("environment prefix already exists; choose a new prefix")
    name = args.environment
    if name not in ("habitat", "bencheval", "gleam") and not args.sources:
        ap.error("this environment requires --sources pointing to the pinned checkouts")
    run(["conda", "create", "-y", "--prefix", prefix, "--file", LOCKS / (name + "-conda-explicit.txt")])
    python = prefix / "bin/python"
    env.update(CUDA_HOME=str(prefix), NVCC_PREPEND_FLAGS="--pre-include cstdint --pre-include cfloat",
               CXXFLAGS="-include cstdint -include cfloat")
    # Both indexes are explicit; every top-level version is frozen in the lock.
    cuda = "cu130" if name in ("habitat-gs", "magician", "gleam") else "cu128"
    run([python, "-m", "pip", "install", "--no-deps", "-r", LOCKS / (name + "-pip.txt"),
         "--extra-index-url", "https://download.pytorch.org/whl/" + cuda])

    def build(path):
        run([python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", path])

    if name == "habitat-gs":
        env.update(HABITAT_WITH_CUDA="ON", HABITAT_WITH_BULLET="ON")
        build(args.sources / "habitat-gs")
    if name == "fisherrf":
        for rel in ("submodules/diff-gaussian-rasterization", "submodules/simple-knn", "diff"):
            build(args.sources / "FisherRF" / rel)
    if name == "r3con":
        glm = args.sources / "R3CON/envs/360-dn-diff-gaussian-rasterization/third_party/glm"
        run(["mkdir", "-p", glm])
        run(["cp", "-a", str(args.sources / "r3con-glm") + "/.", glm])
        build(args.sources / "R3CON/envs/360-dn-diff-gaussian-rasterization")
        build("git+https://github.com/liren-jin/diff-gaussian-rasterization_2d@53be5e91b2d81081ae0829d5b5c16d47b98293fb")
    if name == "gavis":
        build(args.sources / "gavis-dgr")
        build(args.sources / "gavis-rasterizer")
        build("git+https://github.com/YixunLiang/simple-knn.git@a019aef2d544fb8739f977ce09e6e22a90f58e39")
        code = "import site; from pathlib import Path; (Path(site.getsitepackages()[0])/'gavis_rasterizer.py').write_text('from gavis_rasterization import *\\nfrom gavis_rasterization import _C\\n')"
        run([python, "-c", code])
    if name == "magician":
        build("git+https://github.com/facebookresearch/pytorch3d.git@7f8a8a142fa1383a7895ed01dd7082105f9a81fb")
        build("git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38")
        for rel in ("submodules/diff-gaussian-rasterization", "submodules/simple-knn"):
            build(args.sources / "MAGICIAN/RaDe-GS" / rel)
    run([python, "-m", "pip", "install", "--no-deps", "-e", ROOT])


if __name__ == "__main__":
    main()
