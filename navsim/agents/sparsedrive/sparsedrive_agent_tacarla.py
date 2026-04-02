"""SparseDrive agent and target builder for TaCarla (Bench2Drive) dataset."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig
from pyquaternion import Quaternion
from pytorch_lightning.callbacks import ModelCheckpoint

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

from .sparsedrive_config_tacarla import SparseDriveConfigTaCarla
from .sparsedrive_model import SparseDriveModel
from .sparsedrive_features import SparseDriveFeatureBuilder
from .sparsedrive_callback import CheckpointCallback


class TaCarlaTargetBuilder(AbstractTargetBuilder):
    """Target builder for TaCarla dataset.

    Unlike the NAVSIM version, this computes path/velocity targets
    from the Scene's frame data directly instead of re-reading pickle files.
    """

    def __init__(self, config: SparseDriveConfigTaCarla):
        self._config = config

    def get_unique_name(self) -> str:
        return "sparsedrive_target"

    def compute_targets(self, scene: Scene, cfg: DictConfig) -> Dict[str, torch.Tensor]:
        num_poses = self._config.trajectory_sampling.num_poses  # 6

        # Build trajectory from future ego poses
        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=num_poses).poses
        )

        # Compute path and velocity from the scene's frames directly
        path, path_mask = self._get_future_path(scene)

        pad_trajectory = torch.cat([torch.zeros(1, 2), trajectory[:, :2]], dim=0)
        velocity = torch.norm(
            pad_trajectory[1:] - pad_trajectory[:-1], dim=-1
        ) / self._config.vel_time_interval

        return {
            "trajectory": trajectory,
            "path": path,
            "path_mask": path_mask,
            "velocity": velocity,
        }

    def _get_future_path(self, scene: Scene):
        """Compute future path by interpolating ego poses at fixed distance intervals."""
        num_pts = self._config.len_path
        interval = self._config.path_interval
        max_dis = num_pts * interval

        num_history = scene.scene_metadata.num_history_frames
        frames = scene.frames

        # Collect global ego poses from the current frame onward
        global_ego_poses = []
        distances = [0.0]
        accumulated_distance = 0.0

        for frame_idx in range(num_history - 1, len(frames)):
            ego_status = frames[frame_idx].ego_status
            if ego_status.in_global_frame:
                ego_pose = ego_status.ego_pose
            else:
                # Should be global frame for TaCarla scenes
                ego_pose = ego_status.ego_pose

            global_ego_poses.append(np.array(ego_pose, dtype=np.float64))

            if len(global_ego_poses) > 1:
                prev = global_ego_poses[-2]
                curr = global_ego_poses[-1]
                distance = np.linalg.norm(curr[:2] - prev[:2])
                distances.append(distance)
                accumulated_distance += distance

            if accumulated_distance > max_dis:
                break

        # Convert to local coordinates relative to the current frame
        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[0]),
            np.array(global_ego_poses, dtype=np.float64),
        )

        # Interpolate at fixed distance intervals
        distances = np.cumsum(distances)
        target_distance = np.arange(1, num_pts + 1) * interval
        path = np.array([
            np.interp(target_distance, distances, local_ego_poses[:, 0]),
            np.interp(target_distance, distances, local_ego_poses[:, 1]),
            np.interp(target_distance, distances, local_ego_poses[:, 2]),
        ]).T

        # Wrap heading to [-pi, pi)
        path[:, 2] = (path[:, 2] + np.pi) % (2 * np.pi) - np.pi

        path_mask = np.ones(num_pts, dtype=np.float32)
        valid_points = min(num_pts, int(np.floor(accumulated_distance / interval)))
        path_mask[valid_points:] = 0

        return torch.tensor(path, dtype=torch.float32), torch.tensor(path_mask)


class SparseDriveAgentTaCarla(AbstractAgent):
    """SparseDrive agent for TaCarla (Bench2Drive) dataset."""

    def __init__(
        self,
        config: SparseDriveConfigTaCarla,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=3, interval_length=0.5),
    ):
        super().__init__(trajectory_sampling)
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._sparsedrive_model = SparseDriveModel(config)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._checkpoint_path is None:
            return
        if torch.cuda.is_available():
            state_dict = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """6 cameras, no lidar."""
        return SensorConfig(
            cam_f0=[0],
            cam_l0=[0],
            cam_l1=[0],
            cam_l2=[],
            cam_r0=[0],
            cam_r1=[0],
            cam_r2=[],
            cam_b0=[0],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TaCarlaTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [SparseDriveFeatureBuilder(config=self._config)]

    def forward(self, features, targets):
        return self._sparsedrive_model(features, targets)

    def compute_loss(self, features, targets, predictions):
        output, loss_dict = predictions
        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self._sparsedrive_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [CheckpointCallback()]
