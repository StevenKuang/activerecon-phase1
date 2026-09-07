"""Decode every recorded frame of a run tree and report unreadable files.

Episodes are written once and read much later — by a retrain days after the
capture, by the viewer weeks after that. A file can pass its write-time check
and still be unreadable when it finally matters: the read-back in
``resample_stream`` is served from the page cache, so it validates the bytes in
memory rather than the bytes that reached the platter, and nothing revisits
them afterwards.

That is not hypothetical. campaign_v6 lost exactly one frame out of 18,990 —
``interior_0044__d0__s0/fisherrf/stream/frame_00093.png`` — with its PNG header
and IEND trailer intact, no zeroed block, and a normal size for its neighbours:
a few flipped bytes inside the zlib stream. It had decoded at write time and
again during the 30k retrain hours later; it surfaced only when the viewer
tried to open the scene, as an error with no indication of which file was at
fault. SMART reported zero media errors, so the drive never saw a problem.

Run this after a campaign, before trusting or publishing its results:

    python scripts/verify_stream_integrity.py runs_campaign_v6
    python scripts/verify_stream_integrity.py runs_campaign_v6 --include-depth

Exits non-zero when anything fails to decode, so it can gate a pipeline.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

IMAGE_GLOBS = ("*/*/stream/*.png", "*/*/frames/*.png", "*/*/stream/*.jpg")
DEPTH_GLOBS = ("*/*/stream/*.npy",)


def check_image(path: Path) -> Optional[str]:
    """Fully decode an image; returns an error string, or None when healthy."""

    from PIL import Image

    try:
        with Image.open(path) as handle:
            # .load() is what actually walks the compressed stream; opening
            # alone only reads the header and would miss interior corruption.
            handle.load()
            np.asarray(handle)
    except Exception as exc:  # noqa: BLE001 - any decoder failure is a finding
        return "%s: %s" % (type(exc).__name__, exc)
    return None


def check_depth(path: Path) -> Optional[str]:
    try:
        array = np.load(path)
        if not np.isfinite(array).all():
            return "contains non-finite values"
    except Exception as exc:  # noqa: BLE001
        return "%s: %s" % (type(exc).__name__, exc)
    return None


def collect(roots: List[Path], include_depth: bool) -> List[Tuple[Path, str]]:
    targets: List[Tuple[Path, str]] = []
    for root in roots:
        for pattern in IMAGE_GLOBS:
            targets += [(p, "image") for p in root.glob(pattern)]
        if include_depth:
            for pattern in DEPTH_GLOBS:
                targets += [(p, "depth") for p in root.glob(pattern)]
    return sorted(set(targets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="run tree(s) to scan")
    parser.add_argument(
        "--include-depth", action="store_true",
        help="also load the .npy depth maps (slower, and they are large)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--quiet", action="store_true", help="only print the summary and failures"
    )
    args = parser.parse_args()

    targets = collect(args.roots, args.include_depth)
    if not targets:
        print("no frames found under: %s" % ", ".join(str(r) for r in args.roots))
        raise SystemExit(0)
    if not args.quiet:
        print("checking %d files..." % len(targets), flush=True)

    def run(item):
        path, kind = item
        return path, (check_image(path) if kind == "image" else check_depth(path))

    failures: List[Tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path, error in pool.map(run, targets):
            if error is not None:
                failures.append((path, error))
                print("CORRUPT %s\n        %s" % (path, error), flush=True)

    print("\n%d file(s) checked, %d corrupt" % (len(targets), len(failures)))
    if failures:
        print(
            "\nA corrupt frame is regenerable: the poses are recorded, so re-render\n"
            "it from the episode's transforms rather than re-running the planner.\n"
            "Note that resample_stream cannot repair one, since it verifies its\n"
            "re-render against the stored pixels and dies on the file being replaced."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
