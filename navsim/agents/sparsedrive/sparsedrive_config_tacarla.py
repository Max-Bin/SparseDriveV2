from dataclasses import dataclass
from typing import List

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .sparsedrive_config import SparseDriveConfig


@dataclass
class SparseDriveConfigTaCarla(SparseDriveConfig):
    """SparseDrive config for TaCarla (Bench2Drive) dataset."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=3, interval_length=0.5)

    # backbone: ResNet-50
    image_architecture: str = "resnet50"
    bkb_path: str = "ckpt/resnet50.bin"

    # vocabulary anchors (TaCarla-specific)
    path_anchor: str = "ckpt/kmeans_tacarla/path_1024.npy"
    velocity_anchor: str = "ckpt/kmeans_tacarla/velocity_256.npy"
    trajectory_anchor: str = "ckpt/kmeans_tacarla/trajectory_1024_256.npz"

    # path: 15 waypoints @ 1m = 15m horizon
    mode_path: int = 1024
    len_path: int = 15
    path_interval: float = 1.0

    # velocity: 6 timesteps @ 0.5s = 3s horizon
    mode_vel: int = 256
    len_vel_seq: int = 6
    vel_time_interval: float = 0.5

    # all 6 cameras
    cams: List[str] = ("cam_f0", "cam_l0", "cam_r0", "cam_b0", "cam_l1", "cam_r1")

    # TaCarla image params: 1600x900 -> 256x704
    H: int = 900
    W: int = 1600
    final_dim: List[float] = (256, 704)
    resize_lim: List[float] = (0.44, 0.44)  # 704/1600
    bot_pct_lim: List[float] = (0.0, 0.0)

    # no metric supervision (pure imitation learning)
    metric_loss_weight: float = 0.0
    dataset_version: str = "tacarla"
