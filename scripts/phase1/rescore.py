"""Re-score frozen models without changing any historical model or result.

Only PSNR is recomputed here. Geometry, SSIM and LPIPS remain explicitly
historical fields; retrain_eval.py provides the full metric suite.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score(model_path, catalog, legacy):
    import numpy as np
    from PIL import Image
    from activebench.eval.reconstruction import ReconstructionCamera
    from activebench.eval.retrain import NpzGaussianReconstruction, psnr

    payload = json.loads((catalog / "transforms_eval_shared.json").read_text())
    camera = ReconstructionCamera(**{k: payload[v] for k, v in {
        "width": "w", "height": "h", "fx": "fl_x", "fy": "fl_y", "cx": "cx", "cy": "cy"}.items()})
    model = NpzGaussianReconstruction(model_path, force_legacy_dc=legacy)
    views = []
    for index, frame in enumerate(payload["frames"]):
        pose = np.asarray(frame["transform_matrix"], dtype=np.float64) @ np.diag([1., -1., -1., 1.])
        gt = np.asarray(Image.open(catalog / frame["file_path"]))[..., :3].astype(np.float32) / 255.
        pred = model.render(pose, camera).rgb
        if pred.shape != gt.shape:
            raise ValueError(f"Image size mismatch at {catalog}/{index}")
        views.append(dict(index=index, psnr=psnr(pred, gt),
                          label=frame.get("occlusion", {}).get("dyn", {}).get("class")))
    result = dict(psnr=sum(v["psnr"] for v in views) / len(views), n=len(views),
                  color_mode="raw" if model.raw_dc else "legacy", sh_degree=model.sh_degree,
                  views=views, class_psnr={})
    for label in ("severe", "clean"):
        values = [v["psnr"] for v in views if v["label"] == label]
        if values:
            result["class_psnr"][label] = dict(n=len(values), psnr=sum(values)/len(values))
    del model
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, default=ROOT / "phase1/campaign.json")
    ap.add_argument("--source", type=Path, help="Research archive root; omit for portable data layout")
    ap.add_argument("--data", type=Path, default=ROOT / "data/phase1")
    ap.add_argument("--out", type=Path, default=ROOT / "phase1/verification")
    ap.add_argument("--cell", action="append", help="Cell id; repeat to select several")
    ap.add_argument("--catalog", choices=["shared", "cube", "both"], default="both")
    ap.add_argument("--mode", choices=["recorded", "legacy", "both"], default="both")
    args = ap.parse_args()
    campaign = json.loads(args.campaign.read_text())
    failures = []
    args.out.mkdir(parents=True, exist_ok=True)
    for cell in campaign["cells"]:
        if cell["status"] != "retained" or args.cell and cell["id"] not in args.cell:
            continue
        run = args.source / cell["source_episode"] if args.source else args.data / "runs" / cell["id"]
        rec = cell["source_reconstruction"] if args.source else "gsplat"
        model = run / "reconstructions" / rec / "gaussians.npz"
        catalogs = ["shared"] if args.catalog == "shared" else (["cube"] if args.catalog == "cube" else ["shared", "cube"])
        model_sha = None
        for name in catalogs:
            if name == "cube" and cell["group"] != "gs":
                continue
            if args.source:
                cat = args.source / cell["source_eval_catalog" if name == "shared" else "source_cube_catalog"]
            else:
                cat = args.data / (cell["eval_catalog"] if name == "shared" else f"eval/cube/{cell['scene']}__s0")
            modes = [cell["recorded_color_mode"]] if args.mode == "recorded" else ["legacy"]
            if args.mode == "both":
                modes = sorted({"legacy", cell["recorded_color_mode"]})
            if name == "cube":
                modes = ["legacy"]
            for mode in modes:
                destination = args.out / cell["id"] / f"{name}-{mode}.json"
                try:
                    start = time.monotonic()
                    model_sha = model_sha or digest(model)
                    signature = dict(model_sha256=model_sha,
                        catalog_sha256=digest(cat / "transforms_eval_shared.json"), color_mode=mode,
                        evaluator_sha256=digest(ROOT / "src/activebench/eval/retrain.py"),
                        verifier_sha256=digest(Path(__file__)))
                    if destination.exists() and json.loads(destination.read_text()).get("inputs") == signature:
                        print(f"cached {cell['id']} {name} {mode}", flush=True)
                        continue
                    result = score(model, cat, mode == "legacy")
                    result.update(cell=cell["id"], catalog=name, inputs=signature,
                                  wall_time_s=time.monotonic()-start)
                    expected = cell["expected_psnr"]
                    if name == "cube":
                        old = args.campaign.parent / "evidence" / cell["id"] / "eval_cube.json"
                        expected = json.loads(old.read_text())["appearance_per_stratum"]["all"]["psnr"]
                    result["recorded_psnr"] = expected
                    result["delta_from_recorded_db"] = result["psnr"] - expected
                    result["matches_recorded_at_1e-5_db"] = abs(result["psnr"]-expected) <= 1e-5
                    result["recorded_scoring_path"] = cell["recorded_scoring_path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                    print(f"{cell['id']} {name} {mode}: {result['psnr']:.8f}, delta {result['psnr']-expected:+.8f} dB ({result['wall_time_s']:.1f}s)", flush=True)
                except Exception as exc:
                    failures.append(dict(cell=cell["id"], catalog=name, mode=mode, error=str(exc)))
                    print(f"FAILED {cell['id']} {name} {mode}: {exc}", flush=True)
    (args.out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
