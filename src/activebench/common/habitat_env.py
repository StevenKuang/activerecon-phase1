"""Habitat-Sim wrapper for direct 5-DoF camera control.

Habitat-Sim is imported lazily so the rest of the package remains importable in
planner-only environments.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .camera import CameraIntrinsics, CameraPose
from .habitat_coordinates import (
    habitat_quaternion_to_rotation_matrix,
    pose_to_habitat_quaternion,
)
from .transforms import rotation_to_yaw_pitch


@dataclass
class HabitatEnvConfig:
    """Configuration for a minimal Habitat-Sim RGB/depth renderer."""

    scene_path: str
    scene_dataset_config_file: str = "default"
    width: int = 320
    height: int = 240
    hfov: float = 90.0
    sensor_height: float = 1.5
    enable_physics: bool = False
    use_default_lighting: bool = False
    enable_hbao: bool = False
    use_viewer_config: bool = True
    physics_config_file: Optional[str] = None
    resolve_habitat_data_paths: bool = True
    use_habitat_sim_cwd: bool = True

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """Return pinhole intrinsics implied by the render config."""

        return CameraIntrinsics.from_hfov(self.width, self.height, self.hfov)

    def uses_scene_dataset_handle(self) -> bool:
        """True when scene_path is a dataset scene handle (e.g. "apt_0"),
        resolved by the scene dataset config rather than the filesystem."""

        if self.scene_dataset_config_file == "default":
            return False
        return not Path(self.scene_path).expanduser().exists()

    def resolved_scene_path(self) -> Path:
        """Return an expanded scene path and fail early when it is missing."""

        if not self.scene_path or self.scene_path.startswith("/path/to/"):
            raise FileNotFoundError(
                "Habitat scene_path is not configured. Set scene_path in the "
                "config file or pass --scene with a valid .glb/.ply scene."
            )
        path = Path(self.scene_path).expanduser()
        if not path.exists():
            if self.uses_scene_dataset_handle():
                return Path(self.scene_path)
            raise FileNotFoundError(
                "Habitat scene_path does not exist: %s. Set scene_path in the "
                "config file or pass --scene with a valid local scene path." % path
            )
        return path

    def inferred_habitat_data_dir(self) -> Optional[Path]:
        """Infer the Habitat checkout `data` directory from the scene path.

        Habitat's viewer is usually launched from the Habitat-Sim checkout,
        where relative defaults like `data/default.physics_config.json` resolve
        correctly. Project scripts are launched from this repository, so we
        resolve those defaults relative to the scene path when possible.
        """

        if not self.resolve_habitat_data_paths:
            return None
        # For dataset scene handles ("apt_0") the scene path carries no
        # location; anchor the search on the dataset config file instead.
        if self.uses_scene_dataset_handle():
            anchor = Path(self.scene_dataset_config_file).expanduser()
        else:
            anchor = self.resolved_scene_path()
        search_roots = (anchor.parent,) + tuple(anchor.parents)
        for root in search_roots:
            if root.name == "data" and (root / "default.physics_config.json").exists():
                return root
            candidate = root / "data"
            if (candidate / "default.physics_config.json").exists():
                return candidate
        return None

    def resolved_scene_dataset_config_file(self) -> str:
        """Return the dataset config path, resolving Habitat's default if found."""

        if self.scene_dataset_config_file != "default":
            return self.scene_dataset_config_file
        data_dir = self.inferred_habitat_data_dir()
        if data_dir is None:
            return self.scene_dataset_config_file
        candidate = data_dir / "default.scene_dataset_config.json"
        if candidate.exists():
            return str(candidate)
        return self.scene_dataset_config_file

    def resolved_physics_config_file(self) -> Optional[str]:
        """Return an absolute physics config path when one can be resolved."""

        if self.physics_config_file is not None:
            path = Path(self.physics_config_file).expanduser()
            if not path.exists():
                raise FileNotFoundError("Habitat physics_config_file does not exist: %s" % path)
            return str(path)
        data_dir = self.inferred_habitat_data_dir()
        if data_dir is None:
            return None
        candidate = data_dir / "default.physics_config.json"
        if candidate.exists():
            return str(candidate)
        return None

    def inferred_habitat_root(self) -> Optional[Path]:
        """Infer the Habitat-Sim checkout root that owns the `data` directory."""

        data_dir = self.inferred_habitat_data_dir()
        if data_dir is None:
            return None
        return data_dir.parent


def _import_habitat_sim() -> Any:
    try:
        import habitat_sim  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "habitat_sim is required for Habitat rendering but was not found. "
            "Run this smoke test with the Habitat conda environment, for example: "
            "/home/steven/miniconda3/envs/habitat/bin/python scripts/run_habitat_smoke_test.py"
        ) from exc
    return habitat_sim


class HabitatSimWrapper:
    """Small Habitat-Sim renderer with arbitrary 5-DoF camera pose control."""

    def __init__(self, config: HabitatEnvConfig) -> None:
        self.config = config
        self.scene_path = config.resolved_scene_path()
        self.habitat_sim = _import_habitat_sim()
        self.sim = self._make_simulator()
        self.agent = self.sim.get_agent(0)
        self.current_pose: Optional[CameraPose] = None

    def _make_simulator(self) -> Any:
        if self.config.use_viewer_config:
            return self._make_viewer_style_simulator()
        return self._make_minimal_simulator()

    def _construct_simulator(self, cfg: Any) -> Any:
        """Construct Habitat-Sim with the same cwd-sensitive defaults as viewer.py.

        Habitat-Sim 0.3.3 has a compiled default PBR config path of
        `./data/default.pbr_config.json`. `examples/viewer.py` is normally run
        from the Habitat-Sim checkout, so PBR/IBL resolves correctly there. We
        temporarily chdir only while constructing the simulator so project
        scripts launched from this repo load the same material/lighting defaults.
        """

        habitat_root = self.config.inferred_habitat_root()
        if not self.config.use_habitat_sim_cwd or habitat_root is None:
            return self.habitat_sim.Simulator(cfg)

        original_cwd = Path.cwd()
        try:
            os.chdir(str(habitat_root))
            return self.habitat_sim.Simulator(cfg)
        finally:
            os.chdir(str(original_cwd))

    def _make_viewer_style_simulator(self) -> Any:
        """Create a simulator using Habitat-Sim's standard viewer settings helper."""

        try:
            from habitat_sim.utils.settings import default_sim_settings, make_cfg  # type: ignore
        except ImportError as exc:
            raise ImportError("Could not import habitat_sim.utils.settings.make_cfg.") from exc

        settings = dict(default_sim_settings)
        settings["scene"] = str(self.scene_path)
        settings["scene_dataset_config_file"] = self.config.resolved_scene_dataset_config_file()
        settings["enable_physics"] = self.config.enable_physics
        settings["color_sensor"] = True
        settings["depth_sensor"] = True
        settings["semantic_sensor"] = False
        settings["width"] = self.config.width
        settings["height"] = self.config.height
        settings["hfov"] = self.config.hfov
        settings["sensor_height"] = self.config.sensor_height
        settings["enable_hbao"] = self.config.enable_hbao
        settings["default_agent_navmesh"] = False
        physics_config_file = self.config.resolved_physics_config_file()
        if physics_config_file is None:
            settings.pop("physics_config_file", None)
        else:
            settings["physics_config_file"] = physics_config_file
        if self.config.use_default_lighting:
            settings["scene_light_setup"] = self.habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        cfg = make_cfg(settings)
        if self.config.use_default_lighting:
            cfg.sim_cfg.override_scene_light_defaults = True
            cfg.sim_cfg.scene_light_setup = self.habitat_sim.gfx.DEFAULT_LIGHTING_KEY
        return self._construct_simulator(cfg)

    def _make_minimal_simulator(self) -> Any:
        habitat_sim = self.habitat_sim

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(self.scene_path)
        sim_cfg.scene_dataset_config_file = self.config.resolved_scene_dataset_config_file()
        sim_cfg.enable_physics = self.config.enable_physics
        sim_cfg.enable_hbao = self.config.enable_hbao
        physics_config_file = self.config.resolved_physics_config_file()
        if physics_config_file is not None:
            sim_cfg.physics_config_file = physics_config_file
        if self.config.use_default_lighting:
            sim_cfg.override_scene_light_defaults = True
            sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [self.config.height, self.config.width]
        rgb_spec.hfov = self.config.hfov
        rgb_spec.position = [0.0, 0.0, 0.0]

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [self.config.height, self.config.width]
        depth_spec.hfov = self.config.hfov
        depth_spec.position = [0.0, 0.0, 0.0]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

        return self._construct_simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    def _to_habitat_quaternion(self, pose: CameraPose) -> Any:
        return pose_to_habitat_quaternion(pose)

    def current_camera_pose(self) -> CameraPose:
        """Return the current Habitat sensor pose as a roll-free `CameraPose5D`."""

        agent_state = self.agent.get_state()
        sensor_state = None
        if "color_sensor" in agent_state.sensor_states:
            sensor_state = agent_state.sensor_states["color_sensor"]
        elif "rgb" in agent_state.sensor_states:
            sensor_state = agent_state.sensor_states["rgb"]
        elif agent_state.sensor_states:
            sensor_state = next(iter(agent_state.sensor_states.values()))

        if sensor_state is None:
            position = np.asarray(agent_state.position, dtype=np.float64)
            rotation = habitat_quaternion_to_rotation_matrix(agent_state.rotation)
        else:
            position = np.asarray(sensor_state.position, dtype=np.float64)
            rotation = habitat_quaternion_to_rotation_matrix(sensor_state.rotation)

        yaw, pitch = rotation_to_yaw_pitch(rotation)
        return CameraPose.from_xyz_yaw_pitch(position, yaw=yaw, pitch=pitch)

    def set_camera_pose(self, pose: CameraPose) -> None:
        """Set the Habitat camera to an arbitrary world-space `CameraPose5D`.

        `CameraPose5D.position` is the desired camera/sensor position. In the
        viewer-style configuration, Habitat stores the camera as a sensor offset
        from the agent body at `[0, sensor_height, 0]`, so we place the agent
        body below the desired camera and let Habitat infer the sensor state.
        This matches the initial state used by `examples/viewer.py`.
        """

        agent_state = self.agent.get_state()
        rotation = self._to_habitat_quaternion(pose)
        sensor_offset = np.asarray([0.0, self.config.sensor_height, 0.0], dtype=np.float64)
        if not self.config.use_viewer_config:
            sensor_offset[:] = 0.0
        agent_position = pose.position - pose.rotation_matrix() @ sensor_offset
        agent_state.position = agent_position.astype(np.float32)
        agent_state.rotation = rotation
        self.agent.set_state(agent_state, infer_sensor_states=True)
        self.current_pose = pose.copy()

    def render(self, pose: Optional[CameraPose] = None) -> Dict[str, np.ndarray]:
        """Render RGB and depth observations at the current or provided pose."""

        if pose is not None:
            self.set_camera_pose(pose)
        observations = self.sim.get_sensor_observations()
        rgb_key = "color_sensor" if "color_sensor" in observations else "rgb"
        depth_key = "depth_sensor" if "depth_sensor" in observations else "depth"
        rgb = np.asarray(observations[rgb_key])
        depth = np.asarray(observations[depth_key], dtype=np.float32)
        return {"rgb": rgb, "depth": depth}

    def close(self) -> None:
        """Close the underlying simulator."""

        self.sim.close()
