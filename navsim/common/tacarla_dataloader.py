"""
TaCarla (Bench2Drive) data adapter for SparseDriveV2.

Reads TaCarla parquet label files and converts them into the NAVSIM-compatible
frame dict format used by the existing pipeline (SceneLoader, Scene, AgentInput).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from navsim.common.dataclasses import (
    AgentInput, Camera, Cameras, EgoStatus, Frame, Lidar, Scene,
    SceneFilter, SceneMetadata, SensorConfig,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from nuplan.common.actor_state.state_representation import StateSE2

FrameList = List[Dict[str, Any]]

# TaCarla camera name -> NAVSIM camera name
TACARLA_CAM_MAPPING = {
    "front": "CAM_F0",
    "front_left": "CAM_L0",
    "front_right": "CAM_R0",
    "back": "CAM_B0",
    "back_left": "CAM_L1",
    "back_right": "CAM_R1",
}

# CARLA driving command mapping to 4-element one-hot
# CARLA: 1=LEFT, 2=RIGHT, 3=STRAIGHT, 4=FOLLOW_LANE, 5=CHANGE_LEFT, 6=CHANGE_RIGHT
# NAVSIM uses 4-dim: we encode as [left_turn, right_turn, keep_forward, follow_lane]
CARLA_COMMAND_TO_ONEHOT = {
    1: np.array([1, 0, 0, 0], dtype=np.float32),  # LEFT
    2: np.array([0, 1, 0, 0], dtype=np.float32),  # RIGHT
    3: np.array([0, 0, 1, 0], dtype=np.float32),  # STRAIGHT
    4: np.array([0, 0, 0, 1], dtype=np.float32),  # FOLLOW_LANE
    5: np.array([1, 0, 0, 0], dtype=np.float32),  # CHANGE_LEFT -> LEFT
    6: np.array([0, 1, 0, 0], dtype=np.float32),  # CHANGE_RIGHT -> RIGHT
}


def load_tacarla_camera_params(camera_params_dir: Path) -> Dict[str, Any]:
    """Load TaCarla camera intrinsics and extrinsics."""
    with open(camera_params_dir / "new_intrinsics_dict.json") as f:
        intrinsics_raw = json.load(f)
    with open(camera_params_dir / "new_extrinsics_dict.json") as f:
        extrinsics_raw = json.load(f)

    camera_params = {}
    for cam_name in TACARLA_CAM_MAPPING:
        intrinsic = np.array(intrinsics_raw[cam_name], dtype=np.float32)
        extrinsic = np.array(extrinsics_raw[cam_name], dtype=np.float32)
        camera_params[cam_name] = {
            "intrinsic": intrinsic,       # 3x3
            "extrinsic": extrinsic,       # 4x4 cam-to-ego transform
        }
    return camera_params


def _resolve_image_path(parquet_path_str: str, sensor_root: Path, episode_name: str) -> Path:
    """Resolve a TaCarla parquet image path to an actual filesystem path.

    Parquet stores: /leaderboard_plant_pdm_Town12/{episode}/detection/rgb_camera/front/front_10_.jpg
    We need: {sensor_root}/{episode}/detection/rgb_camera/front/front_10_.jpg
    """
    # Strip leading path prefix up to and including the episode name
    path_str = parquet_path_str.strip()
    # Find the detection/ part which is the relative path within the episode
    match = re.search(r"(detection/.+)$", path_str)
    if match:
        rel_path = match.group(1)
    else:
        # fallback: strip leading slashes and first two components
        parts = path_str.strip("/").split("/")
        # skip leaderboard_plant_pdm_TownXX/{episode}/ prefix
        rel_path = "/".join(parts[2:]) if len(parts) > 2 else path_str
    return sensor_root / episode_name / rel_path


def _compute_velocity_acceleration_direct(
    measurements_at_indices: Dict[int, Dict],
    indices: List[int],
    dt: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute ego velocity (vx, vy) and acceleration (ax, ay) in global frame."""
    results = []
    for i, idx in enumerate(indices):
        m = measurements_at_indices[idx]
        speed = float(m["speed"])
        theta = float(m["theta"])

        # Velocity in global frame
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)

        # Acceleration from finite differences
        if i > 0:
            m_prev = measurements_at_indices[indices[i - 1]]
            speed_prev = float(m_prev["speed"])
            theta_prev = float(m_prev["theta"])
            vx_prev = speed_prev * np.cos(theta_prev)
            vy_prev = speed_prev * np.sin(theta_prev)
            ax = (vx - vx_prev) / dt
            ay = (vy - vy_prev) / dt
        else:
            ax, ay = 0.0, 0.0

        results.append((
            np.array([vx, vy], dtype=np.float32),
            np.array([ax, ay], dtype=np.float32),
        ))
    return results


def tacarla_parquet_to_frame_dicts(
    parquet_path: Path,
    sensor_root: Path,
    camera_params: Dict[str, Any],
    subsample: int = 5,
) -> List[Dict[str, Any]]:
    """Convert one TaCarla parquet episode to a list of NAVSIM-compatible frame dicts.

    Args:
        parquet_path: Path to the parquet label file.
        sensor_root: Root directory with extracted sensor data.
        camera_params: Camera intrinsics/extrinsics from load_tacarla_camera_params().
        subsample: Subsample factor (TaCarla is ~10Hz, subsample=5 gives ~2Hz).

    Returns:
        List of frame dicts compatible with NAVSIM SceneLoader pipeline.
    """
    df = pd.read_parquet(parquet_path)
    episode_name = parquet_path.stem
    log_name = episode_name

    # Determine subsampled frame indices
    all_indices = list(range(0, len(df), subsample))
    if len(all_indices) < 2:
        return []

    # Only read measurements for subsampled frames (avoid O(N) dict deserialization)
    measurements_at_indices = {idx: df["measurements"].iloc[idx] for idx in all_indices}
    dt = subsample * 0.1  # 10Hz * subsample frames = dt seconds

    # Compute velocity and acceleration for subsampled frames
    vel_acc = _compute_velocity_acceleration_direct(measurements_at_indices, all_indices, dt)

    frame_dicts = []
    for frame_idx, raw_idx in enumerate(all_indices):
        m = measurements_at_indices[raw_idx]

        # Ego pose from 4x4 matrix
        ego_matrix = np.array(m["ego_matrix"].tolist(), dtype=np.float64)
        ego_translation = ego_matrix[:3, 3].tolist()
        # Use scipy to handle non-perfectly-orthogonal rotation matrices
        rot = R.from_matrix(ego_matrix[:3, :3])
        quat_xyzw = rot.as_quat()  # scipy returns [x, y, z, w]
        ego_rotation = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # [w, x, y, z]

        # Driving command
        command = int(m.get("command", 4))
        driving_command = CARLA_COMMAND_TO_ONEHOT.get(
            command, np.array([0, 0, 0, 1], dtype=np.float32)
        )

        # Velocity and acceleration
        velocity, acceleration = vel_acc[frame_idx]
        ego_dynamic_state = np.concatenate([velocity, acceleration]).tolist()

        # Build camera dict
        cams = {}
        for tc_cam, navsim_cam in TACARLA_CAM_MAPPING.items():
            # Image path
            img_path_str = df[tc_cam].iloc[raw_idx]
            img_path = _resolve_image_path(str(img_path_str), sensor_root, episode_name)

            cam_params = camera_params[tc_cam]
            extrinsic = cam_params["extrinsic"]  # 4x4 cam-to-ego

            cams[navsim_cam] = {
                "data_path": str(img_path),
                "cam_intrinsic": cam_params["intrinsic"].copy(),
                "sensor2lidar_rotation": extrinsic[:3, :3].astype(np.float32),
                "sensor2lidar_translation": extrinsic[:3, 3].astype(np.float32),
                "distortion": np.zeros(5, dtype=np.float32),
            }

        token = f"{episode_name}_{raw_idx}"
        frame_dict = {
            "token": token,
            "timestamp": int(raw_idx),
            "log_name": log_name,
            "scene_token": log_name,
            "map_location": "tacarla",
            "ego2global_translation": ego_translation,
            "ego2global_rotation": ego_rotation,
            "ego_dynamic_state": ego_dynamic_state,
            "driving_command": driving_command,
            "cams": cams,
            "lidar_path": None,
            "anns": {
                "gt_boxes": np.zeros((0, 7), dtype=np.float32),
                "gt_names": [],
                "gt_velocity_3d": np.zeros((0, 3), dtype=np.float32),
                "instance_tokens": [],
                "track_tokens": [],
            },
            "roadblock_ids": ["dummy"],
            "traffic_lights": [],
        }
        frame_dicts.append(frame_dict)

    return frame_dicts


def filter_tacarla_scenes(
    label_dirs: List[Path],
    sensor_root: Path,
    camera_params: Dict[str, Any],
    scene_filter: SceneFilter,
    subsample: int = 5,
    max_episodes: Optional[int] = None,
) -> Tuple[Dict[str, FrameList], List[str]]:
    """Load and filter TaCarla scenes from parquet files.

    Returns:
        filtered_scenes: Dict[token, frame_list]
        final_frame_tokens: List of final frame tokens
    """
    filtered_scenes: Dict[str, FrameList] = {}
    final_frame_tokens: List[str] = []

    parquet_files = []
    for label_dir in label_dirs:
        parquet_files.extend(sorted(label_dir.glob("*.parquet")))

    if scene_filter.log_names is not None:
        allowed_logs = set(scene_filter.log_names)
        parquet_files = [p for p in parquet_files if p.stem in allowed_logs]

    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]

    num_frames = scene_filter.num_frames
    frame_interval = scene_filter.frame_interval

    for parquet_path in tqdm(parquet_files, desc="Loading TaCarla episodes"):
        frame_dicts = tacarla_parquet_to_frame_dicts(
            parquet_path, sensor_root, camera_params, subsample=subsample,
        )
        if not frame_dicts:
            continue

        # Window the episode into scenes
        for start in range(0, len(frame_dicts), frame_interval):
            window = frame_dicts[start : start + num_frames]
            if len(window) < num_frames:
                continue

            token = window[scene_filter.num_history_frames - 1]["token"]

            if scene_filter.tokens is not None and token not in set(scene_filter.tokens):
                continue

            filtered_scenes[token] = window
            final_frame_tokens.append(window[-1]["token"])

            if scene_filter.max_scenes is not None and len(filtered_scenes) >= scene_filter.max_scenes:
                return filtered_scenes, final_frame_tokens

    return filtered_scenes, final_frame_tokens


class TaCarlaSceneLoader:
    """Scene loader for TaCarla dataset, mimics SceneLoader interface."""

    def __init__(
        self,
        label_dirs: List[Path],
        sensor_root: Path,
        camera_params_dir: Path,
        scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
        subsample: int = 5,
        max_episodes: Optional[int] = None,
    ):
        self._sensor_root = sensor_root
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config
        self._camera_params = load_tacarla_camera_params(camera_params_dir)

        self.scene_frames_dicts, _ = filter_tacarla_scenes(
            label_dirs=label_dirs,
            sensor_root=sensor_root,
            camera_params=self._camera_params,
            scene_filter=scene_filter,
            subsample=subsample,
            max_episodes=max_episodes,
        )
        self.synthetic_scenes = {}
        self.synthetic_scenes_tokens = set()

    @property
    def tokens(self) -> List[str]:
        return list(self.scene_frames_dicts.keys())

    @property
    def tokens_stage_one(self) -> List[str]:
        return list(self.scene_frames_dicts.keys())

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx) -> str:
        return self.tokens[idx]

    def _build_cameras(self, scene_dict: Dict, sensor_names: List[str]) -> Cameras:
        """Build Cameras dataclass from frame dict without loading images."""
        cam_dict = scene_dict["cams"]
        data_dict: Dict[str, Camera] = {}
        for navsim_name in ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]:
            upper_name = navsim_name.upper()
            if upper_name in cam_dict and navsim_name in sensor_names:
                cam_info = cam_dict[upper_name]
                data_dict[navsim_name] = Camera(
                    image=None,
                    image_path=Path(cam_info["data_path"]),
                    sensor2lidar_rotation=cam_info["sensor2lidar_rotation"],
                    sensor2lidar_translation=cam_info["sensor2lidar_translation"],
                    intrinsics=cam_info["cam_intrinsic"],
                    distortion=cam_info["distortion"],
                    camera_path=cam_info["data_path"],
                )
            else:
                data_dict[navsim_name] = Camera()

        return Cameras(**data_dict)

    def get_scene_from_token(self, token: str) -> Scene:
        """Build Scene without nuPlan map or eager image loading."""
        assert token in self.scene_frames_dicts
        scene_dict_list = self.scene_frames_dicts[token]
        num_history = self._scene_filter.num_history_frames
        num_future = self._scene_filter.num_future_frames

        scene_metadata = SceneMetadata(
            log_name=scene_dict_list[num_history - 1]["log_name"],
            scene_token=scene_dict_list[num_history - 1]["scene_token"],
            map_name=scene_dict_list[num_history - 1]["map_location"],
            initial_token=scene_dict_list[num_history - 1]["token"],
            num_history_frames=num_history,
            num_future_frames=num_future,
        )

        frames: List[Frame] = []
        for frame_idx, sd in enumerate(scene_dict_list):
            ego_translation = sd["ego2global_translation"]
            ego_quaternion = Quaternion(*sd["ego2global_rotation"])
            global_ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )
            ego_dynamic_state = sd["ego_dynamic_state"]
            ego_status = EgoStatus(
                ego_pose=global_ego_pose,
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=sd["driving_command"],
                in_global_frame=True,
            )

            sensor_names = self._sensor_config.get_sensors_at_iteration(frame_idx)
            cameras = self._build_cameras(sd, sensor_names)

            frame = Frame(
                token=sd["token"],
                timestamp=sd["timestamp"],
                roadblock_ids=sd["roadblock_ids"],
                traffic_lights=sd["traffic_lights"],
                annotations=None,
                ego_status=ego_status,
                lidar=Lidar(),
                cameras=cameras,
            )
            frames.append(frame)

        return Scene(scene_metadata=scene_metadata, map_api=None, frames=frames)

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """Build AgentInput without eager image loading."""
        assert token in self.scene_frames_dicts
        scene_dict_list = self.scene_frames_dicts[token]
        num_history = self._scene_filter.num_history_frames

        # Compute local ego poses relative to the current (last history) frame
        global_ego_poses = []
        for frame_idx in range(num_history):
            sd = scene_dict_list[frame_idx]
            ego_translation = sd["ego2global_translation"]
            ego_quaternion = Quaternion(*sd["ego2global_rotation"])
            global_ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )
            global_ego_poses.append(global_ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[-1]),
            np.array(global_ego_poses, dtype=np.float64),
        )

        ego_statuses = []
        cameras_list = []
        lidars_list = []
        for frame_idx in range(num_history):
            sd = scene_dict_list[frame_idx]
            ego_dynamic_state = sd["ego_dynamic_state"]
            ego_status = EgoStatus(
                ego_pose=np.array(local_ego_poses[frame_idx], dtype=np.float32),
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=sd["driving_command"],
            )
            ego_statuses.append(ego_status)

            sensor_names = self._sensor_config.get_sensors_at_iteration(frame_idx)
            cameras_list.append(self._build_cameras(sd, sensor_names))
            lidars_list.append(Lidar())

        return AgentInput(ego_statuses, cameras_list, lidars_list)

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        tokens_per_logs: Dict[str, List[str]] = {}
        for token, scene_dict_list in self.scene_frames_dicts.items():
            log_name = scene_dict_list[0]["log_name"]
            if log_name in tokens_per_logs:
                tokens_per_logs[log_name].append(token)
            else:
                tokens_per_logs[log_name] = [token]
        return tokens_per_logs
