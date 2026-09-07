"""Re-score frozen models without changing any historical model or result.

Only PSNR is recomputed here. Geometry, SSIM and LPIPS remain explicitly
historical fields; retrain_eval.py provides the full metric suite.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from activebench.selection import add_selection_arguments, select_cells


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


def catalog_images_digest(catalog):
    payload = json.loads((catalog / "transforms_eval_shared.json").read_text())
    hashes = [(frame["file_path"], digest(catalog / frame["file_path"])) for frame in payload["frames"]]
    return hashlib.sha256(json.dumps(hashes).encode()).hexdigest()


def check_report(result, reference, tolerance):
    for key in ("model_sha256", "catalog_sha256"):
        if result["inputs"][key] != reference["inputs"][key]:
            raise ValueError(f"Report input mismatch: {key}")
    if result["color_mode"] != "legacy" or result["n"] != reference["n"]:
        raise ValueError("Report color mode or view count mismatch")
    actual, expected = result["views"], reference["views"]
    if len(actual) != len(expected) or len(actual) != result["n"]:
        raise ValueError("Incomplete report view list")
    deltas = [result["psnr"] - reference["psnr"]]
    for a, b in zip(actual, expected):
        if (a["index"], a["label"]) != (b["index"], b["label"]):
            raise ValueError("Report view identity or class mismatch")
        deltas.append(a["psnr"] - b["psnr"])
    if set(result["class_psnr"]) != set(reference["class_psnr"]):
        raise ValueError("Report classes mismatch")
    for label, value in reference["class_psnr"].items():
        if result["class_psnr"][label]["n"] != value["n"]:
            raise ValueError("Report class count mismatch")
        deltas.append(result["class_psnr"][label]["psnr"] - value["psnr"])
    if any(not math.isfinite(d) or abs(d) > tolerance for d in deltas):
        raise ValueError(f"Report PSNR mismatch: maximum absolute error {max(map(abs,deltas)):.8g} dB; tolerance {tolerance:g}")
    return dict(delta_mean_db=deltas[0], max_absolute_error_db=max(map(abs, deltas)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, default=ROOT / "phase1/campaign.json")
    ap.add_argument("--source", type=Path, help="Research archive root; omit for portable data layout")
    ap.add_argument("--data", type=Path, default=ROOT / "data/phase1")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/rescore")
    add_selection_arguments(ap)
    ap.add_argument("--catalog", choices=["shared", "cube", "both"], default="both")
    ap.add_argument("--mode", choices=["recorded", "legacy", "both"], default="legacy")
    ap.add_argument("--check-report", action="store_true", help="Fail on any mismatch with frozen report PSNR, including per-view scores")
    ap.add_argument("--tolerance-db", type=float, default=1e-4)
    args = ap.parse_args()
    campaign = json.loads(args.campaign.read_text())
    try:
        cells = select_cells(campaign, args)
    except ValueError as exc:
        ap.error(str(exc))
    if args.catalog == "cube" and any(c["group"] != "gs" for c in cells):
        ap.error("cube evaluation is GS-only; use --group gs or a GS scene")
    if args.check_report and args.mode != "legacy":
        ap.error("--check-report requires --mode legacy, the report's loading regime")
    if not math.isfinite(args.tolerance_db) or args.tolerance_db <= 0:
        ap.error("tolerance must be positive and finite")
    frozen = args.campaign.parent.resolve()
    if args.out.resolve() == frozen or frozen in args.out.resolve().parents:
        ap.error("--out must be outside the frozen phase1 evidence directory")
    failures, checked, image_hashes = [], [], {}
    completed = 0
    args.out.mkdir(parents=True, exist_ok=True)
    for cell in cells:
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
                    if cat not in image_hashes:
                        image_hashes[cat] = catalog_images_digest(cat)
                    signature = dict(model_sha256=model_sha,
                        catalog_sha256=digest(cat / "transforms_eval_shared.json"),
                        target_images_sha256=image_hashes[cat], color_mode=mode,
                        evaluator_sha256=digest(ROOT / "src/activebench/eval/retrain.py"),
                        verifier_sha256=digest(Path(__file__)))
                    result = json.loads(destination.read_text()) if destination.exists() else {}
                    cached = result.get("inputs") == signature
                    if not cached:
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
                    message = f"{cell['id']} {name} {mode}: {result['psnr']:.8f} dB" + (" (cached)" if cached else f" ({result['wall_time_s']:.1f}s)")
                    if args.check_report:
                        reference = json.loads((frozen / "verification" / cell["id"] / f"{name}-legacy.json").read_text())
                        comparison = check_report(result, reference, args.tolerance_db)
                        checked.append(dict(cell=cell["id"], catalog=name, **comparison))
                        message += f"; REPORT MATCH, mean delta {comparison['delta_mean_db']:+.8f} dB"
                    print(message, flush=True)
                    completed += 1
                except Exception as exc:
                    failures.append(dict(cell=cell["id"], catalog=name, mode=mode, error=str(exc)))
                    print(f"FAILED {cell['id']} {name} {mode}: {exc}", flush=True)
    (args.out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    summary = dict(selected_cells=len(cells), completed_scores=completed, report_matches=checked,
                   tolerance_db=args.tolerance_db, failures=failures)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(cells)} cells; {completed} scores; {len(checked)} report matches; {len(failures)} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
