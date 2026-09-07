"""Build/verify portable data archives separately from the source repository."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def selected_files(campaign, source, profile):
    files = {}
    for cell in campaign["cells"]:
        if cell["status"] != "retained":
            continue
        run = source / cell["source_episode"]
        if profile == "viewer-assets":
            manifest = json.loads((run / "manifest.json").read_text())
            for item in manifest.get("distractors", []):
                template = Path(item["object_template"])
                payload = json.loads(template.read_text())
                paths = [template] + [(template.parent / payload[k]).resolve()
                    for k in ("render_asset", "collision_asset") if payload.get(k)
                    and (template.parent / payload[k]).is_file()]
                for p in paths:
                    files["sim/" + str(p.relative_to(source.parent))] = p
            continue
        prefix = "runs/" + cell["id"]
        for name in ("manifest.json", "transforms.json", "transforms_stream.json", "coverage.json"):
            if (run / name).is_file():
                files[f"{prefix}/{name}"] = run / name
        if profile == "evaluation":
            for name in ("gaussians.npz", "eval.json", "eval_shared.json"):
                p = run / "reconstructions" / cell["source_reconstruction"] / name
                if not p.is_file():
                    raise FileNotFoundError(p)
                files[f"{prefix}/reconstructions/gsplat/{name}"] = p
            catalog = source / cell["source_eval_catalog"]
            for p in catalog.rglob("*"):
                if p.is_file():
                    files[f"{cell['eval_catalog']}/{p.relative_to(catalog)}"] = p
            if cell["group"] == "gs":
                catalog = source / cell["source_cube_catalog"]
                for p in catalog.rglob("*"):
                    if p.is_file():
                        files[f"eval/cube/{cell['scene']}__s0/{p.relative_to(catalog)}"] = p
            samples = source / "eval_assets/surface" / (cell["scene"] + ".npz")
            if not samples.is_file():
                raise FileNotFoundError(samples)
            files[f"surface/{samples.name}"] = samples
        else:
            directories = ("frames",) if profile == "replay" else ("stream",)
            for directory in directories:
                for p in (run / directory).rglob("*"):
                    if p.is_file():
                        files[f"{prefix}/{directory}/{p.relative_to(run / directory)}"] = p
    return files


class HashReader:
    def __init__(self, f):
        self.f, self.hash = f, hashlib.sha256()

    def read(self, n):
        data = self.f.read(n)
        self.hash.update(data)
        return data


def archive(files, output, profile):
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    records = {}
    options = {"compresslevel": 1} if output.suffix == ".gz" else {}
    with tarfile.open(partial, "w:gz" if output.suffix == ".gz" else "w", **options) as tar:
        for i, (relative, path) in enumerate(sorted(files.items()), 1):
            info = tar.gettarinfo(str(path), arcname=relative)
            # Materialize hard-linked model aliases as ordinary tar files.
            info.type, info.linkname, info.size = tarfile.REGTYPE, "", path.stat().st_size
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as f:
                reader = HashReader(f)
                tar.addfile(info, reader)
            records[relative] = dict(bytes=info.size, sha256=reader.hash.hexdigest())
            if i % 100 == 0:
                print(f"archived {i}/{len(files)}", flush=True)
        payload = json.dumps(dict(schema_version=1, profile=profile, files=records), indent=2).encode()
        info = tarfile.TarInfo(f"{profile}-manifest.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    os.replace(partial, output)
    output.with_suffix(output.suffix + ".manifest.json").write_bytes(payload + b"\n")
    print(f"Wrote {output}: {len(files)} files, {output.stat().st_size / 1e9:.2f} GB")


def verify(root, manifest):
    failures = []
    for rel, expected in json.loads(manifest.read_text())["files"].items():
        p = root / rel
        if not p.is_file() or p.stat().st_size != expected["bytes"]:
            failures.append(rel)
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(8*1024*1024), b""):
                h.update(block)
        if h.hexdigest() != expected["sha256"]:
            failures.append(rel)
    if failures:
        raise ValueError(f"Failed archive integrity: {failures}")
    print("Every file matches its archived size and SHA-256")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, default=ROOT / "phase1/campaign.json")
    ap.add_argument("--source", type=Path, default=ROOT)
    ap.add_argument("--profile", choices=["evaluation", "replay", "streams", "viewer-assets"], default="evaluation")
    ap.add_argument("--out", type=Path, help="Write archive; omit to inspect selection only")
    ap.add_argument("--verify", type=Path, help="Extracted data root")
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    if args.verify:
        if not args.manifest:
            ap.error("--verify requires --manifest")
        return verify(args.verify, args.manifest)
    files = selected_files(json.loads(args.campaign.read_text()), args.source, args.profile)
    print(f"{args.profile}: {len(files)} files, {sum(p.stat().st_size for p in files.values())/1e9:.2f} GB", flush=True)
    if args.out:
        archive(files, args.out, args.profile)


if __name__ == "__main__":
    main()
