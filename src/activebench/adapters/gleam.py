"""ActiveAgent adapter for GLEAM (ICCV 2025) — generalizable indoor exploration.

Wraps the released policy from https://github.com/zjwzcx/GLEAM (checked out at
``repo_root``) *without* its simulator. GLEAM ships as an Isaac Gym task, and
Isaac Gym Preview 4 pins python 3.8 + torch 2.0/cu118 — neither runs on this
machine's sm_120 GPU. The policy itself needs none of that: it is a
LocoTransformer over an egocentric semantic map plus a pose history, in plain
torch. So this adapter imports their network modules verbatim, loads their
released checkpoint, and rebuilds the *environment-side* state construction
against Habitat observations.

What GLEAM actually consumes each step (``gleam/env/env_gleam_stage1.py``):

- ``ego_map_2D`` — a 128x128 egocentric crop at 10 cm/cell of a scene-sized
  128x128 top-down map whose cells are {-1 free, 0 unknown, 1 occupied,
  2 frontier};
- ``state`` — the last 30 world poses expressed relative to the current one.

and it emits a MultiDiscrete ``(ix, iy)`` in [0, 128] naming a *long-term
goal* at ``(index - 64) * cell_size`` metres away. Index 64 means "stay".

Faithfulness notes:

- The map update is transcribed cell-for-cell from their ``update_occ_map_2d``:
  back-project depth, keep only points in the voxel layer at flight height,
  Bresenham-raycast from the agent cell (path -0.05, hit =1.0), then
  threshold at >0.5 occupied / <0.0 free. Their raycast is a pycuda kernel;
  ``_bresenham_rays`` is a vectorized torch equivalent, unit-tested against a
  line-by-line transcription of that kernel.
- Their released config never rotates the agent (action dims 2..5 are clipped
  to a single value and ``update_pose`` only touches x/y/z), so their
  egocentric frame is world-axis-aligned and ``extract_ego_maps`` crops
  without rotation. GLEAM's internal pose therefore carries yaw=0 throughout,
  exactly as in their evaluation; the benchmark camera's yaw is a rendering
  choice that never enters the policy's state.
- Goal validity follows their *code*, which is looser than their prose. The
  only pre-move gate in ``update_pose`` is the target cell reading occupied,
  in which case the agent stays put and re-plans; the A*/BFS connectivity
  check runs after the move and merely raises a collision flag. Aiming into
  unknown space is therefore normal and intended — it is what exploration
  means — even though the paper says "only goals with collision-free and
  navigable paths are considered safe". Connectivity is logged here, not
  enforced. Their occupancy test reads ``scanned_gt_map`` (observed-occupied
  AND ground-truth-occupied); this adapter tests observed-occupied alone,
  the strictly more conservative GT-free reading.
- GLEAM's agent carries a 4-camera 360 deg depth ring (``camera_angles =
  [0, 90, 180, 270]``, 256x32 each) at *both* train and eval time, i.e. a
  ~11 deg band around the flight height that acts as a 2D scan. ActiveBench
  gives one forward camera per capture, so ``spin_legs`` splits each goal
  into legs that rotate 360 deg in total while translating; the benchmark
  charges ``max(translation, rotation)`` time, so this is free whenever
  translation dominates. The 1 Hz stream samples the spin and every frame
  feeds the map.

Runtime requirements (the ``gleam`` conda env): torch only. Habitat, Isaac
Gym, stable-baselines3 and gym are all unnecessary.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from activebench.api import AgentAction, MethodInfo, Observation
from activebench.convention import pose_to_c2w_cv
from activebench.common.camera import CameraPose

from activebench.runtime import external_repo

DEFAULT_REPO_ROOT = external_repo("GLEAM")

# Their released eval configuration (gleam/env/config_gleam_eval.py,
# gleam/env/env_gleam_eval.py, gleam/test/test_gleam_gleambench.py).
GRID_SIZE = 128
EGO_CELL_SIZE = 0.1  # metres per ego-map cell (their ``ego_cell_size``)
POSE_BUFFER = 30  # their ``--buffer_size`` default at eval
POSE_SIZE = 6  # (x, y, z, roll, pitch, yaw)
ACTION_NVEC = (129, 129, 1, 1, 1, 1)  # clip_actions_up - clip_actions_low + 1
STAY_INDEX = 64  # ``init_action``: "keep still"

# Habitat is y-up; GLEAM/Isaac Gym is z-up. This is the rotation taking a
# benchmark world point to theirs, chosen right-handed so the top-down map is
# not mirrored (a reflection would flip left/right under the policy).
_HAB_TO_GLEAM = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def hab_to_gleam(points: np.ndarray) -> np.ndarray:
    """Benchmark world coordinates -> GLEAM world coordinates (z-up)."""

    return np.asarray(points, dtype=np.float64) @ _HAB_TO_GLEAM.T


def gleam_to_hab(points: np.ndarray) -> np.ndarray:
    """GLEAM world coordinates -> benchmark world coordinates (y-up)."""

    return np.asarray(points, dtype=np.float64) @ _HAB_TO_GLEAM


def load_network_classes(repo_root: str):
    """Import GLEAM's pure-torch network modules.

    ``gleam/__init__.py`` registers Isaac Gym tasks, and importing any
    submodule would execute it. Pre-seeding a namespace-only ``gleam``
    package makes ``gleam.network.*`` resolve against the checkout while
    leaving that ``__init__`` unexecuted.
    """

    import types

    repo = str(Path(repo_root).expanduser())
    if "gleam" not in sys.modules:
        package = types.ModuleType("gleam")
        package.__path__ = [str(Path(repo) / "gleam")]
        sys.modules["gleam"] = package
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from gleam.network.base import LocoTransformerEncoder_Map
    from gleam.network.locotransformer import LocoTransformer_GLEAM

    return LocoTransformerEncoder_Map, LocoTransformer_GLEAM


# -- map construction, transcribed from gleam/utils/utils.py ----------------
# Those functions live behind a module-level ``import pycuda.autoinit``, so
# they cannot be imported on a machine without pycuda; the maths below is
# theirs, the raycast is a torch rewrite of their kernel.


def discretize_prob_map(prob_map, threshold_occu: float = 0.5, threshold_free: float = 0.0):
    """Their ``discretize_prob_map``: (occupied 0/1, tri-class -1/0/1)."""

    occupancy = (prob_map > threshold_occu).to(prob_map.dtype)
    free = (prob_map < threshold_free).to(prob_map.dtype)
    return occupancy, occupancy - free


def extract_ego_map(global_map, cell_size_xy, pose_idx, ego_cell: float = EGO_CELL_SIZE):
    """Their ``extract_ego_maps`` for a single environment.

    Resamples a patch of ``ego_cell``-sized cells centred on the agent out of
    the scene-sized global map. Despite the upstream docstring saying "cm",
    both ``cell_sizes`` and ``ego_cm`` are metres there, so the patch spans
    ``128 * 0.1 = 12.8`` m.
    """

    import torch
    import torch.nn.functional as F

    height, width = global_map.shape
    device = global_map.device
    patch_h = height * ego_cell / float(cell_size_xy[0])
    patch_w = width * ego_cell / float(cell_size_xy[1])

    t_y = torch.linspace(0, 1, steps=height, device=device)
    t_x = torch.linspace(0, 1, steps=width, device=device)
    y_coords = (pose_idx[0] - patch_h / 2.0) + patch_h * t_y
    x_coords = (pose_idx[1] - patch_w / 2.0) + patch_w * t_x

    grid_y = y_coords.unsqueeze(1).expand(height, width)
    grid_x = x_coords.unsqueeze(0).expand(height, width)
    norm_y = (grid_y / (height - 1)) * 2 - 1
    norm_x = (grid_x / (width - 1)) * 2 - 1
    batch_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

    ego = F.grid_sample(
        global_map[None, None], batch_grid, mode="nearest",
        padding_mode="zeros", align_corners=True,
    )
    return ego[0, 0]


def compute_frontier_mask(ego_map, frontier_kernel):
    """Their ``compute_frontier_map``: free cells touching unknown ones."""

    import torch
    import torch.nn.functional as F

    denoised = F.max_pool2d(ego_map[None, None], kernel_size=3, stride=1, padding=1)
    mask_free = (denoised == -1).float()
    mask_unknown = (denoised == 0).float()
    unknown_neighbors = F.conv2d(mask_unknown, frontier_kernel, padding=1)
    frontier = (mask_free == 1) & (unknown_neighbors >= 1)
    return frontier[0, 0]


def _bresenham_rays(source, targets, map_size: int):
    """All cells on the Bresenham lines source -> each target, inclusive.

    Vectorized rewrite of their ``bresenham_2d`` pycuda kernel: same integer
    DDA (``err = dx - dy``; ``e2 = 2*err``), same in-bounds filtering, same
    ``map_size * 2`` step cap, so the two enumerate identical cell sets.
    """

    import torch

    device = targets.device
    x0 = source[0].to(torch.long).expand(targets.shape[0]).clone()
    y0 = source[1].to(torch.long).expand(targets.shape[0]).clone()
    x1 = targets[:, 0].to(torch.long)
    y1 = targets[:, 1].to(torch.long)

    dx = (x1 - x0).abs()
    dy = (y1 - y0).abs()
    sx = torch.where(x0 < x1, 1, -1)
    sy = torch.where(y0 < y1, 1, -1)
    err = dx - dy

    x, y = x0.clone(), y0.clone()
    alive = torch.ones_like(x, dtype=torch.bool)
    collected = [torch.stack([x, y], dim=1)]
    for _ in range(int(map_size) * 2 - 1):
        alive = alive & ~((x == x1) & (y == y1))
        if not bool(alive.any()):
            break
        e2 = 2 * err
        step_x = alive & (e2 > -dy)
        step_y = alive & (e2 < dx)
        err = err - torch.where(step_x, dy, torch.zeros_like(dy))
        err = err + torch.where(step_y, dx, torch.zeros_like(dx))
        x = x + torch.where(step_x, sx, torch.zeros_like(sx))
        y = y + torch.where(step_y, sy, torch.zeros_like(sy))
        collected.append(torch.stack([x, y], dim=1)[alive])

    cells = torch.cat(collected, dim=0)
    inside = (
        (cells[:, 0] >= 0) & (cells[:, 0] < map_size)
        & (cells[:, 1] >= 0) & (cells[:, 1] < map_size)
    )
    return cells[inside]


def _reachable_mask(free_mask, start_idx):
    """Cells 4-connected to ``start_idx`` through ``free_mask``.

    Stands in for their A*/BFS goal check (``pathfinding`` with
    ``DiagonalMovement.never`` for visualisation, a custom ``bfs_cuda_2D``
    kernel for the collision flag). Reachability is all either is asked for,
    and iterated dilation computes it exactly without a CUDA extension.
    """

    import torch
    import torch.nn.functional as F

    if not bool(free_mask[start_idx[0], start_idx[1]]):
        return torch.zeros_like(free_mask)
    reached = torch.zeros_like(free_mask)
    reached[start_idx[0], start_idx[1]] = True
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        device=free_mask.device,
    )[None, None]
    # One dilation per unit of geodesic distance. The cap bounds the cost of
    # what is only a logged diagnostic; a truncated flood can under-report
    # connectivity on a pathologically serpentine map, never over-report it.
    for _ in range(4 * max(free_mask.shape)):
        grown = F.conv2d(reached.float()[None, None], kernel, padding=1)[0, 0] > 0
        grown = grown & free_mask
        if bool((grown == reached).all()):
            break
        reached = grown
    return reached


@dataclass
class _SceneGrid:
    """The scene-sized 128x128 top-down grid GLEAM maps into.

    Mirrors their per-scene ``range_gt`` (x_max, x_min, y_max, y_min, z_max,
    z_min) and ``voxel_size_gt`` (extent / grid_size), in GLEAM coordinates.
    """

    lower: np.ndarray  # (3,) GLEAM-frame minimum corner
    upper: np.ndarray  # (3,) GLEAM-frame maximum corner
    size: int = GRID_SIZE

    def __post_init__(self) -> None:
        self.voxel = (self.upper - self.lower) / float(self.size)

    @classmethod
    def from_scene_bbox(cls, scene_bbox, size: int = GRID_SIZE) -> "_SceneGrid":
        bbox = np.asarray(scene_bbox, dtype=np.float64)
        if bbox.shape != (2, 3):
            raise ValueError("scene_bbox must have shape (2, 3)")
        corners = np.array(
            [[bbox[i, 0], bbox[j, 1], bbox[k, 2]]
             for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        )
        mapped = hab_to_gleam(corners)
        return cls(lower=mapped.min(axis=0), upper=mapped.max(axis=0), size=size)

    def cell_of(self, xy: np.ndarray) -> np.ndarray:
        """Their ``pose_coord_to_2d_idx`` (half-voxel-shifted floor, clipped)."""

        origin = self.lower[:2] - 0.5 * self.voxel[:2]
        idx = np.floor((np.asarray(xy, dtype=np.float64) - origin) / self.voxel[:2])
        return np.clip(idx, 0, self.size - 1).astype(np.int64)

    def height_layer(self, height: float) -> int:
        return int((height - self.lower[2]) / self.voxel[2])


class GleamPolicy:
    """GLEAM's actor: their LocoTransformer plus the sb3 MultiDiscrete head.

    stable-baselines3 wraps the feature extractor with ``action_net`` /
    ``value_net`` linears (``net_arch=[]`` leaves the MLP extractor empty), so
    those two are all that must be rebuilt to run their checkpoint.
    """

    def __init__(self, repo_root: str, device: str = "cuda") -> None:
        import torch
        from torch import nn

        encoder_cls, transformer_cls = load_network_classes(repo_root)
        state_dim = POSE_BUFFER * POSE_SIZE
        encoder = encoder_cls(
            in_channels=1, state_input_dim=state_dim,
            hidden_shapes=[256, 256], visual_dim=256,
        )
        self.features = transformer_cls(
            encoder=encoder,
            state_input_shape=state_dim,
            visual_input_shape=(1, GRID_SIZE, GRID_SIZE),
            output_shape=256,
            transformer_params=[[1, 256], [1, 256]],
            append_hidden_shapes=[256],
        )
        self.action_net = nn.Linear(256, int(sum(ACTION_NVEC)))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.features.to(self.device).eval()
        self.action_net.to(self.device).eval()

    def load_checkpoint(self, path: str) -> None:
        """Load a stable-baselines3 ``.zip`` (or a bare ``policy.pth``).

        An sb3 archive stores the policy state dict as ``policy.pth``; the
        feature extractor lives under ``features_extractor.``, and the shared
        encoder is registered twice (``.encoder`` and ``.locotransformer
        .encoder``) so the duplicate keys are dropped, not merged.
        """

        import io
        import zipfile

        import torch

        source = Path(path).expanduser()
        if zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                name = "policy.pth"
                if name not in archive.namelist():
                    candidates = [n for n in archive.namelist() if n.endswith(".pth")]
                    if not candidates:
                        raise ValueError("no .pth inside %s" % source)
                    name = candidates[0]
                payload = torch.load(
                    io.BytesIO(archive.read(name)), map_location="cpu", weights_only=False
                )
        else:
            payload = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("unexpected checkpoint payload: %r" % type(payload))

        feature_state, head_state = {}, {}
        for key, value in payload.items():
            if key.startswith("features_extractor.locotransformer."):
                feature_state[key[len("features_extractor.locotransformer."):]] = value
            elif key.startswith("action_net."):
                head_state[key[len("action_net."):]] = value
        if not feature_state:
            raise ValueError(
                "checkpoint has no features_extractor.locotransformer.* keys "
                "(found: %s)" % ", ".join(sorted(payload)[:8])
            )
        self.features.load_state_dict(feature_state, strict=True)
        self.action_net.load_state_dict(head_state, strict=True)
        self.features.to(self.device).eval()
        self.action_net.to(self.device).eval()

    def act(self, state: np.ndarray, ego_map) -> np.ndarray:
        """Deterministic MultiDiscrete action, as their eval harness runs it."""

        import torch

        with torch.no_grad():
            flat_state = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).reshape(1, -1)
            flat_map = ego_map.to(self.device, torch.float32).reshape(1, -1)
            features = self.features(torch.cat([flat_state, flat_map], dim=1))
            logits = self.action_net(features)[0]
            action, offset = [], 0
            for count in ACTION_NVEC:
                action.append(int(torch.argmax(logits[offset:offset + count])))
                offset += count
        return np.asarray(action, dtype=np.int64)


@dataclass
class GleamAgent:
    """GLEAM's exploration policy as a benchmark ActiveAgent."""

    # Scene bounds in benchmark (Habitat) world coordinates, shape (2, 3).
    scene_bbox: Any
    repo_root: str = DEFAULT_REPO_ROOT
    checkpoint: Optional[str] = None
    # Legs each goal is split into; together they rotate a full turn, which
    # the 1 Hz stream samples to stand in for GLEAM's 360 deg depth ring.
    spin_legs: int = 4
    # Consecutive "stay" decisions tolerated before ending the episode; their
    # env terminates a wandering episode, ours only has the mission clock.
    max_consecutive_stays: int = 8
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.grid = _SceneGrid.from_scene_bbox(self.scene_bbox)
        self._policy: Optional[GleamPolicy] = None
        self._reset_state()

    def _reset_state(self) -> None:
        self._prob_map = None
        self._occ_map = None
        self._tri_map = None
        self._pose_history: List[np.ndarray] = []
        self._motion_height: Optional[float] = None
        self._height_layer: Optional[int] = None
        self._steps = 0
        self._stays = 0
        self._cells_written = 0
        self._frontier_kernel = None
        self._pixel_rays: Optional[np.ndarray] = None

    def info(self) -> MethodInfo:
        return MethodInfo(
            name="gleam",
            needs_depth=True,
            action_space="free",
            conda_env="gleam",
        )

    def reset(self, seed: int, task: Optional[str] = None) -> None:
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        self._reset_state()

    # -- lazy construction --------------------------------------------------

    def _ensure_ready(self, observation: Observation) -> None:
        import torch

        if self._prob_map is not None:
            return
        device = self.device if torch.cuda.is_available() else "cpu"
        self._prob_map = torch.zeros(
            self.grid.size, self.grid.size, dtype=torch.float32, device=device
        )
        self._frontier_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device
        )[None, None]
        # GLEAM flies at a fixed height and slices the world there; the
        # benchmark's start pose sets it.
        self._motion_height = float(hab_to_gleam(observation.pose.position)[2])
        self._height_layer = self.grid.height_layer(self._motion_height)
        intr = observation.intrinsics
        us, vs = np.meshgrid(
            np.arange(intr.width, dtype=np.float64),
            np.arange(intr.height, dtype=np.float64),
            indexing="xy",
        )
        # Habitat depth is planar-z, so a pixel's ray is (x, y, 1) * depth in
        # OpenCV camera axes.
        self._pixel_rays = np.stack(
            [(us - intr.cx) / intr.fx, (vs - intr.cy) / intr.fy, np.ones_like(us)],
            axis=-1,
        ).reshape(-1, 3)
        if self._policy is None:
            self._policy = GleamPolicy(self.repo_root, device=self.device)
            if self.checkpoint:
                self._policy.load_checkpoint(self.checkpoint)
            else:
                print(
                    "[gleam] WARNING: no checkpoint given — the policy is "
                    "randomly initialized and its actions are meaningless."
                )

    # -- map update ---------------------------------------------------------

    def _world_points(self, depth: np.ndarray, pose: CameraPose) -> np.ndarray:
        """Back-project one depth map to GLEAM-frame world points."""

        flat = depth.reshape(-1).astype(np.float64)
        valid = np.isfinite(flat) & (flat > 0.0)
        if not valid.any():
            return np.empty((0, 3))
        camera = self._pixel_rays[valid] * flat[valid, None]
        c2w = pose_to_c2w_cv(pose)
        world = camera @ c2w[:3, :3].T + c2w[:3, 3]
        return hab_to_gleam(world)

    def _ingest(self, points: np.ndarray, pose_cell: np.ndarray) -> None:
        """Their ``update_occ_map_2d`` body for one observation."""

        import torch

        if points.shape[0] == 0:
            return
        origin = self.grid.lower - 0.5 * self.grid.voxel
        upper = self.grid.upper + 0.5 * self.grid.voxel
        inside = np.all((points > origin) & (points < upper), axis=1)
        if not inside.any():
            return
        idx = np.floor((points[inside] - origin) / self.grid.voxel).astype(np.int64)
        # Only the voxel layer at flight height contributes, which is what
        # makes their 11 deg camera band behave like a 2D scan. The layer is
        # ``z_extent / 128`` thick (~2 cm in a 3 m room), so a camera with too
        # few rows can miss it entirely — hence the counter below.
        idx = idx[idx[:, 2] == self._height_layer]
        if idx.shape[0] == 0:
            return
        cells = np.unique(np.clip(idx[:, :2], 0, self.grid.size - 1), axis=0)
        self._cells_written += int(cells.shape[0])

        device = self._prob_map.device
        targets = torch.as_tensor(cells, dtype=torch.long, device=device)
        source = torch.as_tensor(pose_cell, dtype=torch.long, device=device)
        path = _bresenham_rays(source, targets, self.grid.size)
        # Duplicate indices collapse under advanced-index assignment, so a
        # cell is decremented once per update however many rays cross it —
        # matching their kernel's behaviour, not an accumulate.
        self._prob_map[path[:, 0], path[:, 1]] -= 0.05
        self._prob_map[targets[:, 0], targets[:, 1]] = 1.0

    def _observe(self, observation: Observation) -> None:
        frames: List[Tuple[np.ndarray, CameraPose]] = []
        for frame in observation.stream_frames:
            depth = frame.load_depth()
            if depth is not None and frame.pose is not None:
                frames.append((depth, frame.pose))
        if observation.depth is None:
            raise ValueError("GLEAM requires depth observations (needs_depth=True)")
        frames.append((observation.depth, observation.pose))

        self._cells_written = 0
        for depth, pose in frames:
            centre = hab_to_gleam(pose.position)
            self._ingest(self._world_points(depth, pose), self.grid.cell_of(centre[:2]))
        if self._cells_written == 0:
            # Silent starvation looks exactly like a well-behaved unexplored
            # map, so say it: the usual cause is a camera whose vertical
            # sampling is coarser than the flight-height layer.
            print(
                "[gleam] step %d: %d frame(s) contributed no cell at the "
                "flight-height layer (%.1f mm thick)"
                % (self._steps, len(frames), 1000.0 * self.grid.voxel[2]),
                flush=True,
            )

        self._occ_map, self._tri_map = discretize_prob_map(self._prob_map)

    # -- policy input / output ---------------------------------------------

    def _state_vector(self) -> np.ndarray:
        """Their ``ego_pose_buf``: recent world poses minus the current one."""

        buffer = np.zeros((POSE_BUFFER, POSE_SIZE), dtype=np.float32)
        history = self._pose_history[:POSE_BUFFER]
        if history:
            current = history[0]
            for row, pose in enumerate(history):
                buffer[row] = pose - current
        return buffer.reshape(-1)

    def _ego_map(self, pose_cell: np.ndarray):
        import torch

        device = self._prob_map.device
        ego = extract_ego_map(
            self._tri_map,
            self.grid.voxel[:2],
            torch.as_tensor(pose_cell, dtype=torch.float32, device=device),
        )
        occupied = (ego != 1.0)
        frontier = compute_frontier_mask(ego, self._frontier_kernel) & occupied
        ego = ego.clone()
        ego[frontier] = 2.0
        return ego

    def _goal_from_action(self, action: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Their ``update_pose``: index offset in cells -> world displacement.

        Their egocentric-to-world rotation is the identity here because the
        released config never yaws the agent, so this is a world-frame move.
        """

        offset = (action[:2].astype(np.float64) - self.grid.size / 2.0) * self.grid.voxel[:2]
        goal = position.copy()
        goal[:2] += offset
        # Their ``clip_pose_map_bound``.
        return np.clip(goal, self.grid.lower, self.grid.upper)

    def _goal_rejection(self, pose_cell: np.ndarray, goal_cell: np.ndarray) -> Optional[str]:
        """Why this goal is unusable, or None if it is fine.

        This is exactly their pre-move gate: ``update_pose`` translates unless
        the *target cell itself* reads occupied. Their A*/BFS connectivity
        check runs **after** the move and only raises a collision flag; it is
        not a veto on where the agent may aim.

        That distinction is load-bearing, and the paper reads more
        restrictively than the code ("only goals with collision-free and
        navigable paths are considered safe"). Gating on connectivity here
        cost this adapter two thirds of its mission clock: the policy aims
        into unknown space by design — that is what exploration *is* — and
        pre-rejecting those goals left the agent spinning until the
        consecutive-stay cap ended the episode at 99 s of 300 s.
        """

        if bool(self._occ_map[goal_cell[0], goal_cell[1]] == 1.0):
            return "goal cell occupied"  # their ``tar_no_collision``, GT-free
        return None

    def _goal_connectivity(self, pose_cell: np.ndarray, goal_cell: np.ndarray) -> str:
        """Their post-move check, kept as a logged diagnostic rather than a gate.

        Reports whether the goal is reachable through already-observed free
        space, which is what distinguishes a considered next-best-view from a
        leap across an unmapped wall.
        """

        import torch

        free = self._tri_map < 0.0
        state = {-1.0: "free", 0.0: "unknown", 1.0: "occupied"}.get(
            float(self._tri_map[goal_cell[0], goal_cell[1]]), "?"
        )
        if not bool(free[pose_cell[0], pose_cell[1]]):
            return "%s, agent off free space" % state
        reachable = _reachable_mask(free, torch.as_tensor(pose_cell, dtype=torch.long))
        connected = bool(reachable[goal_cell[0], goal_cell[1]])
        return "%s, %sconnected" % (state, "" if connected else "un")

    def _map_census(self) -> str:
        free = int((self._tri_map < 0).sum())
        occupied = int((self._tri_map > 0).sum())
        total = self.grid.size ** 2
        return "map free %d / occ %d / unknown %d" % (free, occupied, total - free - occupied)

    def _spin_trajectory(self, start: CameraPose, goal_hab: np.ndarray) -> AgentAction:
        """Split the move into legs that together turn a full circle.

        Rotation and translation share one clock (``AgentMotionModel``
        charges their max), so a leg that turns 360/spin_legs degrees is free
        whenever its translation takes at least as long.
        """

        legs = max(1, int(self.spin_legs))
        step = 2.0 * np.pi / legs
        waypoints = []
        for leg in range(1, legs + 1):
            alpha = leg / legs
            position = (1.0 - alpha) * start.position + alpha * goal_hab
            waypoints.append(
                CameraPose.from_xyz_yaw_pitch(
                    position,
                    yaw=float((start.yaw + leg * step + np.pi) % (2 * np.pi) - np.pi),
                    pitch=0.0,
                )
            )
        return AgentAction.trajectory(waypoints, capture_mode="last")

    # -- ActiveAgent protocol ----------------------------------------------

    def act(self, observation: Observation) -> AgentAction:
        self._ensure_ready(observation)
        self._observe(observation)

        centre = hab_to_gleam(observation.pose.position)
        position = np.array([centre[0], centre[1], self._motion_height], dtype=np.float64)
        pose_cell = self.grid.cell_of(position[:2])
        self._pose_history.insert(
            0, np.array([position[0], position[1], position[2], 0.0, 0.0, 0.0])
        )
        del self._pose_history[POSE_BUFFER:]

        if self._steps == 0:
            # Their env overrides the first action of every episode with
            # ``init_action`` (stay), so the episode opens by looking around.
            action = np.array([STAY_INDEX, STAY_INDEX, 0, 0, 0, 0], dtype=np.int64)
        else:
            action = self._policy.act(self._state_vector(), self._ego_map(pose_cell))
        self._steps += 1

        goal = self._goal_from_action(action, position)
        goal_cell = self.grid.cell_of(goal[:2])
        if np.array_equal(goal_cell, pose_cell):
            reason = "policy chose to stay"
        else:
            reason = self._goal_rejection(pose_cell, goal_cell)
        if reason is not None:
            # Their rule: a goal on an occupied cell leaves the agent
            # stationary to re-plan. Spinning in place both keeps the mission
            # clock moving and refreshes the map that produced the bad goal.
            self._stays += 1
            print(
                "[gleam] step %d: staying (%s); %s; %d consecutive"
                % (self._steps, reason, self._map_census(), self._stays),
                flush=True,
            )
            if self._stays > self.max_consecutive_stays:
                return AgentAction.done()
            return self._spin_trajectory(observation.pose, observation.pose.position)
        self._stays = 0

        goal_hab = gleam_to_hab(np.array([goal[0], goal[1], self._motion_height]))
        print(
            "[gleam] step %d: goal cell (%d, %d), %.2f m away [%s]; %s"
            % (self._steps, goal_cell[0], goal_cell[1],
               float(np.linalg.norm(goal_hab - observation.pose.position)),
               self._goal_connectivity(pose_cell, goal_cell), self._map_census()),
            flush=True,
        )
        return self._spin_trajectory(observation.pose, goal_hab)
