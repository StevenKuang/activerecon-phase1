import numpy as np
import pytest

from activebench.convention import pose_to_c2w_cv
from activebench.eval.coverage import (
    CoverageParams,
    FrameObservation,
    coverage_from_frames,
)
from activebench.eval.geometry import normals_from_depth_cv, unproject_depth_cv
from activebench.common.camera import CameraIntrinsics, CameraPose

INTR = CameraIntrinsics.from_hfov(64, 64, 90.0)


def render_horizontal_planes(pose: CameraPose, planes):
    """Analytic planar-z depth of horizontal planes seen from a camera.

    ``planes``: list of (y, x_range, z_range) rectangles; nearest hit wins.
    """

    c2w = pose_to_c2w_cv(pose)
    u, v = np.meshgrid(np.arange(INTR.width, dtype=np.float64), np.arange(INTR.height, dtype=np.float64))
    dirs_cam = np.stack([(u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy, np.ones_like(u)], -1)
    dirs_world = dirs_cam @ c2w[:3, :3].T
    origin = c2w[:3, 3]
    depth = np.full(u.shape, np.inf)
    for plane_y, (x0, x1), (z0, z1) in planes:
        dy = dirs_world[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (plane_y - origin[1]) / dy
        hit = dirs_world * t[..., None] + origin
        ok = (t > 0) & (hit[..., 0] >= x0) & (hit[..., 0] <= x1) & (hit[..., 2] >= z0) & (hit[..., 2] <= z1)
        # planar depth = t * (camera-space z of unit-z dir) = t
        depth = np.where(ok & (t < depth), t, depth)
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def grid_points(y, extent, n, normal):
    xs = np.linspace(-extent, extent, n)
    pts = np.array([[x, y, z] for x in xs for z in xs])
    nrm = np.tile(np.asarray(normal, dtype=np.float64), (len(pts), 1))
    return pts, nrm


class TestPitchConvention:
    def test_positive_pitch_looks_up(self):
        pose = CameraPose.from_xyz_yaw_pitch([0, 0, 0], yaw=0.0, pitch=0.5)
        # Viewing direction is -forward in the benchmark convention.
        view = -pose.forward()
        assert view[1] == pytest.approx(np.sin(0.5))


class TestCoverage:
    def make_case(self):
        # Floor at y=0 (normal up), ceiling at y=3 (normal down),
        # camera at y=2 looking straight down-ish.
        cam = CameraPose.from_xyz_yaw_pitch([0.0, 2.0, 0.0], yaw=0.0, pitch=np.deg2rad(-80.0))
        floor_pts, floor_n = grid_points(0.0, 1.0, 12, [0, 1, 0])
        ceil_pts, ceil_n = grid_points(3.0, 1.0, 12, [0, -1, 0])
        points = np.vstack([floor_pts, ceil_pts])
        normals = np.vstack([floor_n, ceil_n])
        return cam, points, normals, len(floor_pts)

    def test_floor_observed_ceiling_not(self):
        cam, points, normals, n_floor = self.make_case()
        depth = render_horizontal_planes(cam, [(0.0, (-50, 50), (-50, 50))])
        frames = [FrameObservation(c2w_gl=cam.as_matrix(), depth=depth)]
        result = coverage_from_frames(frames, points, normals, INTR)
        assert result["bins"]["up"]["observed_frac"] > 0.8
        assert result["bins"]["down"]["observed_frac"] == 0.0
        # Looking almost straight down at the floor: incidence is excellent.
        assert result["bins"]["up"]["mean_best_cos_observed"] > 0.85

    def test_occluder_blocks_points_beneath_it(self):
        cam, points, normals, n_floor = self.make_case()
        # A table at y=1 covering x,z in [-0.4, 0.4] occludes the floor under it.
        table = (1.0, (-0.4, 0.4), (-0.4, 0.4))
        depth = render_horizontal_planes(cam, [(0.0, (-50, 50), (-50, 50)), table])
        frames = [FrameObservation(c2w_gl=cam.as_matrix(), depth=depth)]
        result = coverage_from_frames(frames, points, normals, INTR)

        floor_pts = points[:n_floor]
        under = (np.abs(floor_pts[:, 0]) < 0.3) & (np.abs(floor_pts[:, 2]) < 0.3)
        # Recompute per-point observation via a targeted query: run coverage
        # on only the under-table points.
        under_result = coverage_from_frames(
            frames, floor_pts[under], normals[:n_floor][under], INTR
        )
        assert under_result["overall"]["observed_frac"] == 0.0
        # Points clearly outside the table's shadow are unaffected by it. The
        # camera at y=2 projects the table edge (0.4 at y=1) to 0.8 on the
        # floor, so "outside" means beyond 0.85.
        outside = (np.abs(floor_pts[:, 0]) > 0.85) | (np.abs(floor_pts[:, 2]) > 0.85)
        depth_clear = render_horizontal_planes(cam, [(0.0, (-50, 50), (-50, 50))])
        clear_frames = [FrameObservation(c2w_gl=cam.as_matrix(), depth=depth_clear)]
        with_table = coverage_from_frames(
            frames, floor_pts[outside], normals[:n_floor][outside], INTR
        )
        without_table = coverage_from_frames(
            clear_frames, floor_pts[outside], normals[:n_floor][outside], INTR
        )
        assert with_table["overall"]["observed_frac"] == pytest.approx(
            without_table["overall"]["observed_frac"]
        )
        assert without_table["overall"]["observed_frac"] > 0.5

    def test_looking_up_sees_ceiling(self):
        cam = CameraPose.from_xyz_yaw_pitch([0.0, 1.0, 0.0], yaw=0.0, pitch=np.deg2rad(80.0))
        _, points, normals, n_floor = self.make_case()
        depth = render_horizontal_planes(cam, [(3.0, (-50, 50), (-50, 50))])
        frames = [FrameObservation(c2w_gl=cam.as_matrix(), depth=depth)]
        result = coverage_from_frames(frames, points, normals, INTR)
        assert result["bins"]["down"]["observed_frac"] > 0.8
        assert result["bins"]["up"]["observed_frac"] == 0.0

    def test_range_limit(self):
        cam, points, normals, _ = self.make_case()
        depth = render_horizontal_planes(cam, [(0.0, (-50, 50), (-50, 50))])
        frames = [FrameObservation(c2w_gl=cam.as_matrix(), depth=depth)]
        result = coverage_from_frames(
            frames, points, normals, INTR, CoverageParams(max_range=1.0)
        )
        assert result["overall"]["observed_frac"] == 0.0  # floor is >=2m away


class TestNormalsFromDepth:
    def test_flat_floor_normals(self):
        cam = CameraPose.from_xyz_yaw_pitch([0.0, 2.0, 0.0], yaw=0.3, pitch=np.deg2rad(-70.0))
        depth = render_horizontal_planes(cam, [(0.0, (-50, 50), (-50, 50))])
        normals_cam, valid = normals_from_depth_cv(depth.astype(np.float64), INTR)
        assert valid.sum() > 0.5 * valid.size
        c2w = pose_to_c2w_cv(cam)
        n_world = normals_cam[valid] @ c2w[:3, :3].T
        # Floor normals in world space point up.
        assert np.mean(n_world[:, 1] > 0.95) > 0.95

    def test_unproject_round_trip(self):
        cam = CameraPose.from_xyz_yaw_pitch([0.4, 1.7, -0.2], yaw=1.1, pitch=-0.4)
        depth = render_horizontal_planes(cam, [(0.0, (-50, 50), (-50, 50))])
        pts_cam = unproject_depth_cv(depth.astype(np.float64), INTR)
        c2w = pose_to_c2w_cv(cam)
        valid = depth > 0
        pts_world = pts_cam[valid] @ c2w[:3, :3].T + c2w[:3, 3]
        # Every unprojected floor pixel lies on the y=0 plane.
        assert np.abs(pts_world[:, 1]).max() < 1e-6
