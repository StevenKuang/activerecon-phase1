"""ActiveAgent adapter for MAGICIAN (Li et al.) — imagined-Gaussian planning.

Wraps the official implementation from https://github.com/shiyao-li/MAGICIAN
(checked out at ``repo_root``) as a tier-2 full policy. Their published test
protocol (``test_magician_planning.py`` with ``use_perfect_depth_map: true``)
renders mesh scenes with pytorch3d and reads each frame back from a saved
``{n}.pt`` file; everything downstream — MACARONS surface/proxy scenes, SCONE
occupancy, imagined Gaussians, beam search over a discrete pose lattice — is
renderer-agnostic. This adapter replaces exactly that render/save step with
benchmark observations and leaves the rest of their pipeline untouched
(their ``Camera`` drives pose bookkeeping; their planning-cycle code is
transcribed from ``compute_magician_trajectory`` with the changes listed
below).

Coordinate/convention mapping (verified empirically against their
``get_camera_RT``):

- Their world is y-up like Habitat; positions map linearly by ``world_scale``
  (their scenes live at mesh×10 ≈ 30–60 units and constants like
  ``carving_tolerance`` and SCONE's training distribution assume that scale,
  so Habitat meters are scaled up accordingly). Depth maps scale the same.
- Their view direction for (elev, azim) degrees is
  ``(cos e·sin a, sin e, cos e·cos a)`` → benchmark ``pitch = elev``,
  ``yaw = azim + 180°``.
- Their cameras are pytorch3d ``FoVPerspectiveCameras`` with the default
  fov=60°, aspect 1 → the episode must render with
  ``fx = fy = (min(H, W)/2) / tan(30°)`` (the adapter refuses to run
  otherwise and prints the required hfov).
- pytorch3d zbuf is planar-z with −1 background; Habitat depth is planar-z
  with 0 at no-hit → scaled and remapped.

Documented deviations from their test harness:

- Step-1 collision checks approximate their trimesh ray intersector with a
  proximity test against the benchmark's dense GT surface samples using a
  ``collision_radius`` of 0.35 magician units (~surface thickness) — their
  1-unit threshold is calibrated to sparse imagined-occupancy clouds and
  blocks every move against dense samples in furnished scenes. If filtering
  still blocks every neighbor, the expansion retries unfiltered rather than
  ending the episode. Later beam steps use their occupancy points and their
  own function, unchanged.
- ``gt_scene`` (used only for their coverage printout) is filled from the
  benchmark's GT surface samples instead of mesh sampling.
- The pose lattice is authored from the benchmark scene bounds with their
  settings-file semantics (grid cell centers, interior elevations, wrapped
  azimuths); their per-scene lattice shapes are hand-authored per dataset
  scene, ours default to a similar spacing in their units.

Runtime requirements: the existing ``magician`` conda env (their repo's own
environment: torch + pytorch3d + RaDe-GS rasterizer). No habitat needed.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from activebench.api import AgentAction, MethodInfo, Observation
from activebench.common.camera import CameraPose

from activebench.runtime import external_repo

DEFAULT_REPO_ROOT = external_repo("MAGICIAN")


def pose_to_magician(pose: CameraPose, scale: float):
    """Benchmark pose → (X_cam(3,), elev_deg, azim_deg) in magician units."""

    x = np.asarray(pose.position, dtype=np.float64) * scale
    elev = float(np.rad2deg(pose.pitch))
    azim = float(np.rad2deg(pose.yaw - np.pi)) % 360.0
    return x, elev, azim


def magician_to_pose(x_cam, elev_deg: float, azim_deg: float, scale: float) -> CameraPose:
    position = np.asarray(x_cam, dtype=np.float64) / scale
    return CameraPose.from_xyz_yaw_pitch(
        position,
        yaw=float(np.deg2rad(azim_deg) + np.pi),
        pitch=float(np.deg2rad(elev_deg)),
    )


@dataclass
class MagicianAgent:
    """Official MAGICIAN planner as a benchmark ActiveAgent."""

    # Habitat-world scene bounds, shape (2, 3).
    scene_bbox: Any
    # GT surface samples npz (activebench.eval.surface_samples) for gt_scene
    # fill and early-step collision checks.
    surface_samples_path: str
    repo_root: str = DEFAULT_REPO_ROOT
    # Habitat meters → magician units. None: scale so the mean bbox extent
    # matches their dataset's typical ~30 units.
    world_scale: Optional[float] = None
    beam_width: int = 10
    beam_steps: int = 10
    max_captures: int = 20
    # Lattice authoring: target spacing in magician units, matching the
    # spacing of their hand-authored scene settings (~6-7 units).
    pose_spacing: float = 6.5
    # Step-1 collision: block a move when the segment passes within this many
    # magician units of a GT surface sample (see module docstring).
    collision_radius: float = 0.35
    pose_n_elev: int = 4
    pose_n_azim: int = 6
    scratch_dir: Optional[str] = None
    seed: int = 0

    def __post_init__(self) -> None:
        repo = str(Path(self.repo_root).expanduser())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        bbox = np.asarray(self.scene_bbox, dtype=np.float64)
        if bbox.shape != (2, 3):
            raise ValueError("scene_bbox must have shape (2, 3)")
        self._bbox_m = bbox
        if self.world_scale is None:
            extent = float(np.mean(bbox[1] - bbox[0]))
            self.world_scale = 30.0 / max(extent, 1e-6)
        self._initialized = False

    def info(self) -> MethodInfo:
        return MethodInfo(
            name="magician",
            needs_depth=True,  # their published protocol: perfect depth maps
            action_space="free",
            conda_env="magician",
        )

    def reset(self, seed: int, task: Optional[str] = None) -> None:
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        self._initialized = False

    # -- intrinsics contract ---------------------------------------------------

    def _check_intrinsics(self, observation: Observation) -> None:
        intr = observation.intrinsics
        expected = (min(intr.width, intr.height) / 2.0) / np.tan(np.deg2rad(30.0))
        if abs(intr.fx - expected) > 0.01 * expected or abs(intr.fy - expected) > 0.01 * expected:
            required_hfov = float(np.rad2deg(2.0 * np.arctan(intr.width / (2.0 * expected))))
            raise ValueError(
                "MAGICIAN uses pytorch3d FoVPerspectiveCameras with fov=60deg "
                "(fx=fy=min(H,W)/2/tan30). Episode intrinsics fx=%.2f do not "
                "match the required %.2f; set habitat hfov to %.3f deg."
                % (intr.fx, expected, required_hfov)
            )

    # -- stack construction ------------------------------------------------------

    def _author_settings(self, device):
        from macarons.utility.macarons_utils import Settings

        s = self.world_scale
        lo = (self._bbox_m[0] * s).tolist()
        hi = (self._bbox_m[1] * s).tolist()
        extent = np.asarray(hi) - np.asarray(lo)
        grid = [max(2, int(np.ceil(e / 9.0))) for e in extent]
        pose_dims = [max(2, int(round(e / self.pose_spacing))) for e in extent]
        settings_dict = {
            "scene": {
                "x_min": lo,
                "x_max": hi,
                "visibility_ratio": 1.0,
                "grid_l": grid[0],
                "grid_w": grid[1],
                "grid_h": grid[2],
                "cell_capacity": 1000,
                "cell_resolution": 0.05,
            },
            "camera": {
                "x_min": lo,
                "x_max": hi,
                "pose_l": pose_dims[0],
                "pose_w": pose_dims[1],
                "pose_h": pose_dims[2],
                "pose_n_theta": self.pose_n_elev,
                "pose_n_azim": self.pose_n_azim,
                "start_positions": [[0, 0, 0, self.pose_n_elev // 2, 0]],
                "contrast_factor": 1.0,
            },
        }
        # Settings applies scene_scale_factor; ours is pre-applied.
        return Settings(settings_dict, device, scene_scale_factor=1.0)

    def _init_stack(self, observation: Observation) -> None:
        import torch
        from types import SimpleNamespace

        cwd = os.getcwd()
        os.chdir(self.repo_root)  # their configs/weights paths are repo-relative
        try:
            from macarons.testers.magician_planning import (
                load_params,
                load_pretrained_macarons,
            )
            from macarons.utility.macarons_utils import Camera, Scene

            params = load_params(
                os.path.join(self.repo_root, "configs/macarons/macarons_default_training_config.json")
            )
            params.jz = False
            params.numGPU = 0
            params.anomaly_detection = False
            device = torch.device("cuda")
            intr = observation.intrinsics
            params.image_height = intr.height
            params.image_width = intr.width
            params.n_interpolation_steps = 1
            params.n_poses_in_trajectory = self.max_captures
            params.beam_width = self.beam_width
            params.beam_steps = self.beam_steps
            self._params = params
            self._device = device

            macarons = load_pretrained_macarons(
                pretrained_model_path=params.pretrained_model_path,
                device=device,
                learn_pose=params.learn_pose,
            )
            weights = torch.load(
                os.path.join(self.repo_root, "weights/macarons/trained_macarons.pth"),
                map_location=device,
                weights_only=False,
            )
            macarons.load_state_dict(weights["model_state_dict"], ddp=True)
            macarons.eval()
            self._macarons = macarons

            settings = self._author_settings(device)
            self._settings = settings

            def make_scene(feature_dim, cell_capacity, cell_resolution, score_threshold=None):
                kwargs = dict(
                    x_min=settings.scene.x_min,
                    x_max=settings.scene.x_max,
                    grid_l=settings.scene.grid_l,
                    grid_w=settings.scene.grid_w,
                    grid_h=settings.scene.grid_h,
                    cell_capacity=cell_capacity,
                    cell_resolution=cell_resolution,
                    n_proxy_points=params.n_proxy_points,
                    device=device,
                    view_state_n_elev=params.view_state_n_elev,
                    view_state_n_azim=params.view_state_n_azim,
                    feature_dim=feature_dim,
                    mirrored_scene=False,
                    mirrored_axis=None,
                )
                if score_threshold is not None:
                    kwargs["score_threshold"] = score_threshold
                return Scene(**kwargs)

            test_resolution = 0.05
            self._gt_scene = make_scene(3, params.surface_cell_capacity,
                                        test_resolution * params.scene_scale_factor)
            self._covered_scene = make_scene(1, params.surface_cell_capacity,
                                             test_resolution * params.scene_scale_factor)
            self._surface_scene = make_scene(1, params.surface_cell_capacity, None)
            self._proxy_scene = make_scene(1, params.proxy_cell_capacity,
                                           params.proxy_cell_resolution,
                                           score_threshold=params.score_threshold)
            self._proxy_scene.initialize_proxy_points()
            self._test_resolution = test_resolution

            samples = np.load(self.surface_samples_path)
            gt_points = torch.from_numpy(
                samples["points"].astype(np.float32) * self.world_scale
            ).to(device)
            self._gt_points = gt_points
            self._gt_scene.fill_cells(
                gt_points, features=torch.full((len(gt_points), 3), 0.5, device=device)
            )

            frames_dir = Path(self.scratch_dir or (Path(self.repo_root) / "results" / "bench_frames"))
            frames_dir.mkdir(parents=True, exist_ok=True)
            (frames_dir.parent / "imgs").mkdir(parents=True, exist_ok=True)
            renderer_stub = SimpleNamespace(
                rasterizer=SimpleNamespace(
                    raster_settings=SimpleNamespace(image_size=(intr.height, intr.width))
                )
            )
            camera = Camera(
                x_min=settings.camera.x_min,
                x_max=settings.camera.x_max,
                pose_l=settings.camera.pose_l,
                pose_w=settings.camera.pose_w,
                pose_h=settings.camera.pose_h,
                pose_n_elev=settings.camera.pose_n_elev,
                pose_n_azim=settings.camera.pose_n_azim,
                n_interpolation_steps=1,
                zfar=params.zfar,
                renderer=renderer_stub,
                device=device,
                contrast_factor=1.0,
                gathering_factor=params.gathering_factor,
                occupied_pose_data=None,
                save_dir_path=str(frames_dir),
            )
            start_idx = self._nearest_pose_idx(camera, observation.pose)
            camera.initialize_camera(start_cam_idx=start_idx)
            # The episode's first frame was captured at the episode start
            # pose, which is generally off-lattice; overwrite the camera's
            # current state and last history entry with the true pose so the
            # stored R/T match the pixels. Motion continues on the lattice.
            x, elev, azim = pose_to_magician(observation.pose, self.world_scale)
            self._set_camera_state(camera, x, elev, azim)
            self._camera = camera

            self._full_pc = torch.zeros(0, 3, device=device)
            self._pose_i = 0
            self._initialized = True
        finally:
            os.chdir(cwd)

    def _set_camera_state(self, camera, x, elev, azim) -> None:
        import torch
        from macarons.utility.macarons_utils import get_camera_RT
        from pytorch3d.renderer import FoVPerspectiveCameras

        device = camera.device
        camera.X_cam = torch.tensor([list(x)], dtype=torch.float32, device=device)
        camera.V_cam = torch.tensor([[elev, azim]], dtype=torch.float32, device=device)
        camera.X_cam_history[-1] = camera.X_cam[0]
        camera.V_cam_history[-1] = camera.V_cam[0]
        r, t = get_camera_RT(camera.X_cam, camera.V_cam)
        camera.fov_camera = FoVPerspectiveCameras(R=r, T=t, zfar=camera.zfar, device=device)
        camera.fov_camera_0 = camera.fov_camera

    def _nearest_pose_idx(self, camera, pose: CameraPose):
        import torch

        x, elev, azim = pose_to_magician(pose, self.world_scale)
        best_key, best_cost = None, None
        target = np.asarray([*x, elev, azim])
        for key, lattice_pose in camera.pose_space.items():
            p = lattice_pose.cpu().numpy()
            azim_diff = min(abs(p[4] - azim), 360.0 - abs(p[4] - azim))
            cost = np.linalg.norm(p[:3] - target[:3]) + 0.05 * abs(p[3] - elev) + 0.05 * azim_diff
            if best_cost is None or cost < best_cost:
                best_key, best_cost = key, cost
        return camera.get_idx_from_key(best_key)

    # -- frame ingestion ---------------------------------------------------------

    def _write_frame(self, observation: Observation) -> None:
        """Store the benchmark observation exactly as their capture_image does."""

        self._write_frame_arrays(observation.rgb, observation.depth)

    def _write_frame_arrays(self, rgb_array: np.ndarray, depth_m: Optional[np.ndarray]) -> None:
        """Persist one frame under the camera's CURRENT fov state."""

        import torch

        camera = self._camera
        rgb = torch.from_numpy(
            (rgb_array[..., :3].astype(np.float32) / 255.0)
        ).unsqueeze(0)  # (1, H, W, 3)
        if depth_m is None:
            raise ValueError("MAGICIAN adapter requires depth observations")
        zbuf = depth_m.astype(np.float32) * self.world_scale
        zbuf[depth_m <= 0.0] = -1.0  # pytorch3d background convention
        zbuf_t = torch.from_numpy(zbuf).unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
        frame = {
            "rgb": rgb.to(self._device),
            "zbuf": zbuf_t.to(self._device),
            "mask": (zbuf_t > -1).to(self._device),
            "R": camera.fov_camera.R,
            "T": camera.fov_camera.T,
            "zfar": camera.zfar,
        }
        path = os.path.join(camera.save_dir_path, str(camera.n_frames_captured) + ".pt")
        torch.save(frame, path)
        camera.n_frames_captured += 1

    def _segment_hits_gt(self, start, end) -> bool:
        """Segment-vs-GT-surface proximity test (their math, tighter radius)."""

        import torch

        points = self._gt_points
        line = end - start
        length_sq = torch.norm(line) ** 2
        t = torch.sum((points - start) * line, dim=1) / length_sq
        within = (t >= 0) & (t <= 1)
        closest = start + t.unsqueeze(1) * line
        d_in = torch.norm(points - closest, dim=1)
        d_out = torch.minimum(
            torch.norm(points - start, dim=1), torch.norm(points - end, dim=1)
        )
        distances = torch.where(within, d_in, d_out)
        return bool(torch.min(distances).item() < self.collision_radius)

    # -- one planning cycle (transcribed from compute_magician_trajectory) --------

    def act(self, observation: Observation) -> AgentAction:
        import torch

        self._check_intrinsics(observation)
        if not self._initialized:
            self._init_stack(observation)
        cwd = os.getcwd()
        os.chdir(self.repo_root)
        try:
            with torch.no_grad():
                for frame in observation.stream_frames:
                    self._ingest_stream_frame(frame)
                self._write_frame(observation)
                next_idx = self._plan_step()
        finally:
            os.chdir(cwd)
        if next_idx is None:
            return AgentAction.done()
        camera = self._camera
        camera.update_camera(next_idx, interpolation_step=1)
        x = camera.X_cam[0].cpu().numpy()
        elev, azim = float(camera.V_cam[0, 0]), float(camera.V_cam[0, 1])
        return AgentAction.move_to(magician_to_pose(x, elev, azim, self.world_scale))

    def _process_current_frame(self) -> None:
        """Fold the newest written frame into their surface/proxy scenes
        (transcribed from their process_current_frame)."""

        import torch
        from macarons.testers.magician_planning import (
            apply_perfect_depth_simple,
            load_current_frame_perfect_depth,
        )

        params, camera, device = self._params, self._camera, self._device
        surface_scene, proxy_scene = self._surface_scene, self._proxy_scene
        covered_scene = self._covered_scene

        current_frame = load_current_frame_perfect_depth(camera, device)
        depth, mask, error_mask, r, t = apply_perfect_depth_simple(current_frame, device)
        fov_camera = camera.get_fov_camera_from_RT(R_cam=r, T_cam=t)
        x_cam = fov_camera.get_camera_center()
        part_pc, part_pc_features = camera.compute_partial_point_cloud(
            depth=depth,
            mask=(mask * error_mask).bool(),
            images=current_frame["rgb"],
            fov_cameras=fov_camera,
            gathering_factor=params.gathering_factor * 2,
            fov_range=params.sensor_range,
        )
        fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(
            proxy_scene.proxy_points, return_mask=True, fov_camera=None, fov_range=params.sensor_range
        )
        sgn_dists = None
        if fov_proxy_mask.any():
            sgn_dists = camera.get_signed_distance_to_depth_maps(
                pts=fov_proxy_points, depth_maps=depth, mask=mask, fov_camera=None
            )

        zero_features = torch.zeros(len(part_pc), 1, device=device)
        covered_scene.fill_cells(part_pc, features=zero_features)
        surface_scene.fill_cells(part_pc, features=zero_features)
        self._full_pc = torch.vstack((self._full_pc, part_pc))
        if fov_proxy_mask.any():
            fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(fov_proxy_mask)
            proxy_scene.fill_cells(fov_proxy_points, features=fov_proxy_indices.view(-1, 1))
            proxy_scene.update_proxy_view_states(
                camera, fov_proxy_mask, signed_distances=sgn_dists,
                distance_to_surface=None, X_cam=x_cam,
            )
            proxy_scene.update_proxy_supervision_occ(
                fov_proxy_mask, sgn_dists, tol=params.carving_tolerance
            )
            proxy_scene.update_proxy_out_of_field(fov_proxy_mask)
        surface_scene.set_all_features_to_value(value=1.0)

    def _ingest_stream_frame(self, frame) -> None:
        """Protocol v3: run one passively-streamed trajectory frame through
        their per-frame pipeline (write + process_current_frame) under a
        temporarily re-pointed camera. Camera history stays
        decision-poses-only: novelty is their decision-level concept; the
        stream contributes geometry through the surface/proxy scenes.
        """

        import torch
        from macarons.utility.macarons_utils import get_camera_RT
        from pytorch3d.renderer import FoVPerspectiveCameras

        depth = frame.load_depth()
        if depth is None:
            raise ValueError("MAGICIAN stream ingestion requires depth")
        camera = self._camera
        device = camera.device
        saved = (camera.X_cam, camera.V_cam, camera.fov_camera)
        x, elev, azim = pose_to_magician(frame.pose, self.world_scale)
        camera.X_cam = torch.tensor([list(x)], dtype=torch.float32, device=device)
        camera.V_cam = torch.tensor([[elev, azim]], dtype=torch.float32, device=device)
        r, t = get_camera_RT(camera.X_cam, camera.V_cam)
        camera.fov_camera = FoVPerspectiveCameras(R=r, T=t, zfar=camera.zfar, device=device)
        try:
            self._write_frame_arrays(frame.load_rgb(), depth)
            self._process_current_frame()
        finally:
            camera.X_cam, camera.V_cam, camera.fov_camera = saved

    def _plan_step(self):
        import torch
        from macarons.testers.magician_planning import (
            fill_surface_scene,
            compute_scene_occupancy_probability_field,
            render_gaussian_depth,
            update_gaussian_colors_from_novelty,
            line_segment_intersects_point_cloud_region,
        )
        from macarons.utility.macarons_utils import get_camera_RT
        from pytorch3d.renderer import FoVPerspectiveCameras

        params, camera, device = self._params, self._camera, self._device
        surface_scene, proxy_scene = self._surface_scene, self._proxy_scene
        covered_scene, gt_scene = self._covered_scene, self._gt_scene

        camera.fov_camera_0 = camera.fov_camera
        if self._pose_i > 0 and self._pose_i % params.recompute_surface_every_n_loop == 0:
            fill_surface_scene(
                surface_scene,
                self._full_pc,
                random_sampling_max_size=params.n_gt_surface_points,
                min_n_points_per_cell_fill=3,
                progressive_fill=params.progressive_fill,
                max_n_points_per_fill=params.max_points_per_progressive_fill,
            )

        self._process_current_frame()

        coverage = gt_scene.scene_coverage(
            covered_scene,
            surface_epsilon=2 * self._test_resolution * params.scene_scale_factor,
        )
        print("[magician] pose %d coverage %.4f, surface pts %d"
              % (self._pose_i, float(coverage[0]), len(self._full_pc)))

        # --- occupancy field -> imagined gaussians ---
        x_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(
            params, self._macarons.scone, camera, surface_scene, proxy_scene, device
        )
        filtered_x_world = x_world[occ_probs.squeeze() > 0.5]
        n_points = filtered_x_world.shape[0]
        gaussian_means = filtered_x_world
        gaussian_opacities = occ_probs[occ_probs.squeeze() > 0.5]
        gaussian_scales = torch.ones(n_points, 3, device=device) * (0.7154 / 2)
        gaussian_rotations = torch.tensor(
            [[1, 0, 0, 0]], device=device, dtype=torch.float32
        ).repeat(n_points, 1)

        sample_x = camera.X_cam_history[0].view(1, 3)
        sample_v = camera.V_cam_history[0].view(1, 2)
        r0, t0 = get_camera_RT(sample_x, sample_v)
        sample_camera = FoVPerspectiveCameras(R=r0, T=t0, zfar=camera.zfar, device=device)
        k_matrix = sample_camera.get_projection_transform().get_matrix().transpose(-1, -2)

        # --- novelty from camera history ---
        from macarons.testers.magician_planning import convert_camera_from_pytorch3d_to_gs

        novelty_values = torch.zeros(n_points, device=device)
        for cam_idx in range(len(camera.X_cam_history)):
            xc = camera.X_cam_history[cam_idx].view(1, 3)
            vc = camera.V_cam_history[cam_idx].view(1, 2)
            rc, tc = get_camera_RT(xc, vc)
            hist_camera = FoVPerspectiveCameras(R=rc, T=tc, zfar=camera.zfar, device=device)
            hist_camera.K = k_matrix
            gs_camera = convert_camera_from_pytorch3d_to_gs(
                hist_camera, height=camera.image_height, width=camera.image_width, device=device
            )[0]
            gaussian_colors = update_gaussian_colors_from_novelty(novelty_values)
            rendered_depth, _ = render_gaussian_depth(
                gaussian_means=gaussian_means,
                gaussian_opacities=gaussian_opacities,
                gaussian_scales=gaussian_scales,
                gaussian_rotations=gaussian_rotations,
                gaussian_colors=gaussian_colors,
                gs_camera=gs_camera,
                device=device,
                bg_color=torch.tensor([1.0, 1.0, 1.0], device=device),
                kernel_size=0.01,
            )
            visible = camera.check_point_visibility_from_depth(
                filtered_x_world, hist_camera, rendered_depth[0], depth_tolerance=1.0
            )
            novelty_values[visible] = 1.0

        # --- beam search (their loop, with pc-based step-1 collision) ---
        scene_scale = float(
            (self._settings.scene.x_max - self._settings.scene.x_min).mean()
        )
        beams = [{
            "trajectory": [],
            "novelty_values": novelty_values.clone(),
            "score": novelty_values.sum().item(),
            "total_coverage_gain": 0.0,
            "current_pose_idx": camera.cam_idx,
        }]
        for bs_i in range(params.beam_steps):
            all_candidates = []
            for beam in beams:
                neighbor_indices = camera.get_neighboring_poses(pose_idx=beam["current_pose_idx"])
                valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=None)
                current_pose, _ = camera.get_pose_from_idx(beam["current_pose_idx"])
                x_current, _, _ = camera.get_camera_parameters_from_pose(current_pose)

                def expand(apply_collision_filter):
                    rendering, indices = [], []
                    for row in valid_neighbors:
                        neighbor_pose, _ = camera.get_pose_from_idx(row)
                        x_neighbor, _, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
                        if apply_collision_filter:
                            if bs_i == 0:
                                blocked = self._segment_hits_gt(x_current[0], x_neighbor[0])
                            else:
                                blocked = line_segment_intersects_point_cloud_region(
                                    filtered_x_world, x_current[0], x_neighbor[0]
                                )
                            if blocked:
                                continue
                        rendering.append(fov_neighbor)
                        indices.append(row)
                    return rendering, indices

                rendering_candidate, idx_candidate = expand(True)
                if not rendering_candidate and bs_i == 0:
                    # Never let collision filtering end the episode from the
                    # start pose; risk a wall-graze instead.
                    print("[magician] all neighbors collision-blocked; retrying unfiltered")
                    rendering_candidate, idx_candidate = expand(False)
                if not rendering_candidate:
                    continue

                for j, pose_idx in enumerate(idx_candidate):
                    fov_camera_j = rendering_candidate[j]
                    fov_camera_j.K = k_matrix
                    gs_camera = convert_camera_from_pytorch3d_to_gs(
                        fov_camera_j, height=camera.image_height, width=camera.image_width, device=device
                    )[0]
                    gaussian_colors = update_gaussian_colors_from_novelty(beam["novelty_values"])
                    rendered_depth, rendered_image = render_gaussian_depth(
                        gaussian_means=gaussian_means,
                        gaussian_opacities=gaussian_opacities,
                        gaussian_scales=gaussian_scales,
                        gaussian_rotations=gaussian_rotations,
                        gaussian_colors=gaussian_colors,
                        gs_camera=gs_camera,
                        device=device,
                        bg_color=torch.tensor([1.0, 1.0, 1.0], device=device),
                        kernel_size=0.01,
                    )
                    depth_map = rendered_depth[0]
                    visible_mask = camera.check_point_visibility_from_depth(
                        filtered_x_world, fov_camera_j, depth_map, depth_tolerance=1.0
                    )
                    valid_depth_mask = depth_map > 0
                    grayscale = rendered_image.mean(dim=0)
                    if valid_depth_mask.any():
                        depth_weight = ((depth_map / (scene_scale / 2.0)) ** 2).clamp_max(1.0)
                        coverage_gain = (
                            grayscale * depth_weight * valid_depth_mask.float()
                        ).sum().item()
                    else:
                        coverage_gain = 0.0
                    new_novelty = beam["novelty_values"].clone()
                    new_novelty[visible_mask] = 1.0
                    all_candidates.append({
                        "trajectory": beam["trajectory"] + [pose_idx],
                        "novelty_values": new_novelty,
                        "coverage_gain": coverage_gain,
                        "total_coverage_gain": beam["total_coverage_gain"] + coverage_gain,
                        "current_pose_idx": pose_idx,
                    })
            if not all_candidates:
                break
            all_candidates.sort(key=lambda c: c["total_coverage_gain"], reverse=True)
            beams = all_candidates[: params.beam_width]

        self._pose_i += 1
        if not beams or not beams[0]["trajectory"]:
            return None
        return beams[0]["trajectory"][0]
