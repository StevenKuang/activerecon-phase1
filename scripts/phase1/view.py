"""Launch the Phase 1 Spark catalog with portable data paths."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/phase1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--group", choices=["mesh", "gs"], action="append")
    ap.add_argument("--prepare-cache", action="store_true")
    args = ap.parse_args()
    data = args.data.resolve()
    # Add verified PSNR as a sidecar; preserve all original archive metadata.
    campaign = json.loads((ROOT / "phase1/campaign.json").read_text())
    for cell in campaign["cells"]:
        if cell["status"] != "retained" or args.group and cell["group"] not in args.group:
            continue
        rec = data / "runs" / cell["id"] / "reconstructions/gsplat"
        model = rec / "gaussians.npz"
        if not model.is_file():
            raise SystemExit(f"Missing model: {model}; restore the evaluation archive")
        verified = ROOT / "phase1/verification" / cell["id"] / "shared-legacy.json"
        payload = json.loads(verified.read_text())
        sidecar = rec / "psnr_verification.json"
        # A matching persisted sidecar avoids hashing the large model on every
        # launch. Archive verification remains available for subsequent damage.
        model_stat = dict(bytes=model.stat().st_size, mtime_ns=model.stat().st_mtime_ns)
        previous = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if previous.get("inputs") != payload["inputs"] or previous.get("model_stat") != model_stat:
            digest = hashlib.sha256()
            with model.open("rb") as f:
                for block in iter(lambda:f.read(8*1024*1024),b""):
                    digest.update(block)
            if digest.hexdigest() != payload["inputs"]["model_sha256"]:
                raise SystemExit(f"Model does not match frozen verification: {model}")
            payload["model_stat"] = model_stat
            sidecar.write_text(json.dumps(payload, indent=2) + "\n")
            print("Verified model:", cell["id"], flush=True)
    env = dict(os.environ)
    env["ACTIVEBENCH_PHASE1_DATA"] = str(data)
    for name, variable in [("habitat-sim", "HABITAT_SIM_ROOT"), ("habitat-gs", "HABITAT_GS_ROOT")]:
        bundled = data / "sim" / name
        if bundled.is_dir():
            env.setdefault(variable, str(bundled))
    command = [sys.executable, str(ROOT / "scripts/export_web_demo.py"),
               "--reconstruction", "gsplat", "--serve", str(args.port)]
    for group in args.group or ["gs", "mesh"]:
        command += ["--runs-dir", str(data / "runs" / group)]
    if args.prepare_cache:
        command += ["--prepare-cache"]
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()
