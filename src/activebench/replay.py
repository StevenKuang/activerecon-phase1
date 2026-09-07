"""UI-independent episode replay core for the demo viewer.

Loads a recorded episode and answers time queries so the Spark frontend can
replay the scan: interpolated camera pose at any sim time, which captures are
already taken, per-capture fused point-cloud chunks, and distractor poses
reconstructed from the manifest's trajectory configs. Kept UI-independent so
the replay logic is testable headlessly.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from activebench.convention import pose_to_c2w_cv
from activebench.distractors import make_trajectory
from activebench.episode import interpolate_pose
from activebench.eval.geometry import unproject_depth_cv
from activebench.eval.reconstruction import (
    resolve_reconstruction_eval,
    resolve_shared_eval,
)
from activebench.common.camera import CameraIntrinsics, CameraPose
from activebench.common.transforms import rotation_to_yaw_pitch

_GL_CV_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

# Shared-eval view classes (see eval/shared_eval.py): severe = red, clean =
# green, mixed = gray. Used by the viewer's shared-eval camera frustums.
SHARED_CLASS_COLORS = {
    "severe": (230, 60, 60),
    "clean": (70, 200, 90),
    "mixed": (170, 170, 170),
}


def shared_eval_set_dir(episode_dir: Path, shared_root: Path) -> Optional[Path]:
    """Shared eval set directory for an episode's (scene, seed) group, if built."""

    parts = Path(episode_dir).parent.name.rsplit("__", 2)
    if len(parts) != 3:
        return None
    scene, _difficulty, seed = parts
    candidate = Path(shared_root) / ("%s__%s" % (scene, seed))
    if (candidate / "transforms_eval_shared.json").exists():
        return candidate
    return None


def shared_view_class(frame: Dict[str, Any], difficulty: str) -> str:
    """Class label of a shared eval frame under one difficulty's occlusion.

    Falls back to the hardest labeled difficulty (e.g. for d0 episodes, which
    are compared against the dynamic difficulties' class splits).
    """

    occlusion = frame.get("occlusion", {})
    entry = occlusion.get(difficulty)
    if entry is None and occlusion:
        entry = occlusion[sorted(occlusion)[-1]]
    return entry.get("class", "mixed") if entry else "mixed"


@dataclass
class ReplayFrame:
    index: int
    time: float
    pose: CameraPose
    c2w_gl: np.ndarray
    rgb_path: Path
    depth_path: Path
    mask_path: Optional[Path]


@dataclass
class FrameChunk:
    """World-space points contributed by one capture."""

    points: np.ndarray
    colors: np.ndarray
    distractor_points: np.ndarray


def resolve_render_asset(object_template: str) -> Optional[Path]:
    """Path of the GLB render asset referenced by an object_config.json."""

    from activebench.runtime import relocated_asset_path

    template = relocated_asset_path(object_template)
    if not template.exists():
        return None
    try:
        asset = json.loads(template.read_text()).get("render_asset")
    except (json.JSONDecodeError, OSError):
        return None
    if not asset:
        return None
    path = (template.parent / asset).resolve()
    return path if path.exists() else None


class EpisodeReplay:
    def __init__(
        self,
        episode_dir: Path,
        point_stride: int = 6,
        max_depth: float = 8.0,
    ) -> None:
        self.episode_dir = Path(episode_dir)
        self.manifest = json.loads((self.episode_dir / "manifest.json").read_text())
        reconstruction = self.manifest.get("reconstruction") or {}
        transforms_name = reconstruction.get("transforms", "transforms.json")
        payload = json.loads((self.episode_dir / transforms_name).read_text())
        capture_entries = reconstruction.get("frames", self.manifest["captures"])
        self.capture_entries = capture_entries
        self.intrinsics = CameraIntrinsics(
            width=int(payload["w"]), height=int(payload["h"]),
            fx=float(payload["fl_x"]), fy=float(payload["fl_y"]),
            cx=float(payload["cx"]), cy=float(payload["cy"]),
        )
        self.frames: List[ReplayFrame] = []
        for capture, frame in zip(capture_entries, payload["frames"]):
            m = np.asarray(frame["transform_matrix"], dtype=np.float64)
            yaw, pitch = rotation_to_yaw_pitch(m[:3, :3])
            self.frames.append(
                ReplayFrame(
                    index=int(capture["index"]),
                    time=float(capture["time"]),
                    pose=CameraPose.from_xyz_yaw_pitch(m[:3, 3], yaw=yaw, pitch=pitch),
                    c2w_gl=m,
                    rgb_path=self.episode_dir / frame["file_path"],
                    depth_path=self.episode_dir / frame["depth_path"],
                    mask_path=(self.episode_dir / frame["mask_path"]) if frame.get("mask_path") else None,
                )
            )
        if len(capture_entries) != len(payload["frames"]):
            raise ValueError("replay capture metadata and transforms have different frame counts")
        self.t_end = self.frames[-1].time if self.frames else 0.0
        self.point_stride = point_stride
        self.max_depth = max_depth
        self._chunks: List[Optional[FrameChunk]] = [None] * len(self.frames)

        self.distractors: List[Tuple[str, Any, Optional[Path]]] = []
        for spec in self.manifest.get("distractors", []):
            try:
                trajectory = make_trajectory(dict(spec["trajectory"]))
            except Exception:
                continue
            self.distractors.append(
                (spec["name"], trajectory, resolve_render_asset(spec["object_template"]))
            )

    # -- time queries ---------------------------------------------------------

    def capture_index_at(self, t: float) -> int:
        """Index of the last capture taken at or before sim time ``t`` (>= 0)."""

        times = [f.time for f in self.frames]
        return max(0, int(np.searchsorted(times, t + 1e-9) - 1))

    def camera_pose_at(self, t: float) -> CameraPose:
        """Camera pose at sim time ``t``, interpolated between captures."""

        k = self.capture_index_at(t)
        if k >= len(self.frames) - 1:
            return self.frames[-1].pose
        a, b = self.frames[k], self.frames[k + 1]
        span = max(b.time - a.time, 1e-9)
        alpha = float(np.clip((t - a.time) / span, 0.0, 1.0))
        return interpolate_pose(a.pose, b.pose, alpha)

    def distractor_poses_at(self, t: float) -> List[Tuple[str, np.ndarray, float]]:
        return [
            (name, *trajectory.pose_at(t))
            for name, trajectory, _ in self.distractors
        ]

    # -- geometry --------------------------------------------------------------

    def chunk(self, index: int) -> FrameChunk:
        """Fused points of capture ``index`` (computed lazily, cached)."""

        if self._chunks[index] is None:
            from PIL import Image

            frame = self.frames[index]
            stride = self.point_stride
            depth = np.load(frame.depth_path)[::stride, ::stride].astype(np.float64)
            try:
                with Image.open(frame.rgb_path) as image:
                    rgb = np.asarray(image)[::stride, ::stride, :3]
            except OSError as exc:
                raise OSError("Cannot decode replay RGB frame %d: %s" % (
                    frame.index, frame.rgb_path)) from exc
            distractor = None
            if frame.mask_path is not None:
                mask = np.asarray(Image.open(frame.mask_path))[::stride, ::stride]
                distractor = (mask[..., 0] if mask.ndim == 3 else mask) > 0
            sub = CameraIntrinsics(
                width=depth.shape[1], height=depth.shape[0],
                fx=self.intrinsics.fx / stride, fy=self.intrinsics.fy / stride,
                cx=self.intrinsics.cx / stride, cy=self.intrinsics.cy / stride,
            )
            valid = (depth > 0.0) & (depth < self.max_depth)
            cam = unproject_depth_cv(depth, sub)
            c2w = frame.c2w_gl @ _GL_CV_FLIP
            world = cam[valid] @ c2w[:3, :3].T + c2w[:3, 3]
            colors = rgb[valid]
            if distractor is not None:
                dmask = distractor[valid]
                chunk = FrameChunk(world[~dmask], colors[~dmask], world[dmask])
            else:
                chunk = FrameChunk(world, colors, np.zeros((0, 3)))
            self._chunks[index] = chunk
        return self._chunks[index]

    def summary(self, reconstruction_run: Optional[str] = "auto") -> str:
        from activebench.audit import world_fingerprint

        method = self.manifest["method"]["name"]
        reconstruction = self.manifest.get("reconstruction") or {}
        frame_count = reconstruction.get("num_frames", self.manifest["num_captures"])
        text = "**%s** — %d reconstruction frames, %d agent captures, %.1fs sim time, world %s" % (
            method,
            frame_count,
            self.manifest["num_captures"],
            self.manifest["clock"]["final_sim_time"],
            world_fingerprint(self.manifest),
        )
        budget = self.manifest["clock"].get("max_sim_time")
        if budget is not None:
            text += " | budget %.0fs" % budget
            if self.manifest["clock"]["final_sim_time"] < budget - 1.0:
                text += " (ended early)"
        coverage = self.episode_dir / "coverage.json"
        if coverage.exists():
            overall = json.loads(coverage.read_text()).get("overall", {})
            if "observed_frac" in overall:
                text += " | observed surface %.1f%% (diagnostic)" % (100 * overall["observed_frac"])
        retrain = resolve_reconstruction_eval(self.episode_dir, reconstruction_run)
        verified_psnr = None
        if retrain is not None:
            verified_path = retrain.parent / "psnr_verification.json"
            if verified_path.exists():
                verified_psnr = json.loads(verified_path.read_text())
        if retrain is not None:
            rt = json.loads(retrain.read_text())
            text += " | recon PSNR %.1f, cmp@5cm %.0f%%" % (
                verified_psnr["psnr"] if verified_psnr else rt["appearance_per_stratum"]["all"]["psnr"],
                100 * rt["geometry"]["completeness@0.05"])
        shared = resolve_shared_eval(self.episode_dir, reconstruction_run)
        if shared is not None:
            payload = json.loads(shared.read_text())
            key = payload.get("difficulty")
            if key not in payload["class_psnr"] and payload["class_psnr"]:
                key = sorted(payload["class_psnr"])[-1]
            groups = payload["class_psnr"].get(key, {})
            if verified_psnr:
                groups = verified_psnr["class_psnr"]
            if groups:
                cell = lambda label: (
                    "%.1f" % groups[label]["psnr"] if label in groups else "-")
                text += " | shared[%s] sev/clean %s/%s" % (
                    key, cell("severe"), cell("clean"))
        return text


@dataclass
class RunIndex:
    """Scene/difficulty/seed/method index over supported run directories."""

    runs: Dict[str, Dict[str, Dict[str, Dict[str, Path]]]] = field(default_factory=dict)

    @classmethod
    def scan(cls, roots: List[Path]) -> "RunIndex":
        index = cls()
        for root in roots:
            root = Path(root)
            manifests = set(root.glob("*/*/manifest.json"))
            manifests.update(root.glob("*/manifest.json"))
            for manifest in sorted(manifests):
                episode_dir = manifest.parent
                if episode_dir.parent == root:
                    payload = json.loads(manifest.read_text())
                    scene, difficulty = episode_dir.name, "-"
                    raw_seed = payload.get("seed")
                    seed = "-" if raw_seed is None else "s%s" % raw_seed
                    method = payload.get("method", {}).get("name", episode_dir.name)
                else:
                    method = episode_dir.name
                    parts = episode_dir.parent.name.rsplit("__", 2)
                    if len(parts) == 3:
                        scene, difficulty, seed = parts
                    else:
                        scene, difficulty, seed = episode_dir.parent.name, "-", "-"
                index.runs.setdefault(scene, {}).setdefault(difficulty, {}).setdefault(seed, {})[
                    method
                ] = episode_dir
        return index

    def scenes(self) -> List[str]:
        return sorted(self.runs)

    def difficulties(self, scene: str) -> List[str]:
        return sorted(self.runs.get(scene, {}))

    def seeds(self, scene: str, difficulty: str) -> List[str]:
        seeds = self.runs.get(scene, {}).get(difficulty, {})

        def sort_key(seed: str) -> Tuple[int, Any]:
            if seed.startswith("s") and seed[1:].isdigit():
                return (0, int(seed[1:]))
            return (1, seed)

        return sorted(seeds, key=sort_key)

    def methods(self, scene: str, difficulty: str, seed: str) -> List[str]:
        return sorted(self.runs.get(scene, {}).get(difficulty, {}).get(seed, {}))

    def episode_dir(self, scene: str, difficulty: str, seed: str, method: str) -> Path:
        return self.runs[scene][difficulty][seed][method]


@dataclass(frozen=True)
class ExperimentRound:
    """A named, immutable campaign result set discoverable by the viewer."""

    round_id: str
    label: str
    runs_dir: Path
    manifest_path: Path

    @classmethod
    def load(cls, path: Path) -> "ExperimentRound":
        import yaml

        path = Path(path)
        payload = yaml.safe_load(path.read_text())
        results = payload.get("results", {})
        runs_dir = results.get("runs_dir")
        if not payload.get("id") or not runs_dir:
            raise ValueError("round manifest needs id and results.runs_dir: %s" % path)
        resolved = (path.parent / runs_dir).resolve()
        return cls(
            round_id=str(payload["id"]),
            label=str(payload.get("label", payload["id"])),
            runs_dir=resolved,
            manifest_path=path.resolve(),
        )

    @classmethod
    def scan(cls, rounds_dir: Path) -> List["ExperimentRound"]:
        rounds = [cls.load(path) for path in sorted(Path(rounds_dir).glob("*.yaml"))]
        ids = [round_.round_id for round_ in rounds]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment round ids must be unique")
        return rounds
