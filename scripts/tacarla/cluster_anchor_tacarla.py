#!/usr/bin/env python3
"""K-means clustering for TaCarla trajectory vocabulary anchors.

Run AFTER caching the TaCarla dataset. Reads cached path/velocity targets
and generates anchor files for the TaCarla config.

Usage:
    python scripts/tacarla/cluster_anchor_tacarla.py --cache-path exp/tacarla_cache
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from navsim.planning.training.dataset import load_feature_target_from_pickle

K_PATH = 1024
K_VELOCITY = 256
DT = 0.5
CKPT_DIR = "ckpt/kmeans_tacarla"


def interp1d_extrap(x, xp, fp):
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    y = np.interp(x, xp, fp)
    m_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
    left_mask = x < xp[0]
    y[left_mask] = fp[0] + m_left * (x[left_mask] - xp[0])
    m_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    right_mask = x > xp[-1]
    y[right_mask] = fp[-1] + m_right * (x[right_mask] - xp[-1])
    return y


def interp_trajectory(path_cluster, velocity_cluster):
    num_velocity = velocity_cluster.shape[1]
    trajectory = np.zeros((K_PATH, K_VELOCITY, num_velocity, 3))
    trajectory_mask = np.ones((K_PATH, K_VELOCITY, num_velocity))

    for i in range(K_PATH):
        for j in range(K_VELOCITY):
            path = path_cluster[i]
            velocity = velocity_cluster[j]
            target_distance = np.cumsum(velocity * DT, axis=0)
            pad_path = np.concatenate([np.zeros((1, 3)), path], axis=0)
            distance = np.linalg.norm(pad_path[1:, :2] - pad_path[:-1, :2], axis=-1).cumsum(axis=0)
            distance = np.concatenate([np.zeros((1,)), distance], axis=0)
            interp_traj = np.array([
                interp1d_extrap(target_distance, distance, pad_path[:, 0]),
                interp1d_extrap(target_distance, distance, pad_path[:, 1]),
                interp1d_extrap(target_distance, distance, pad_path[:, 2]),
            ]).T
            interp_traj[:, 2] = (interp_traj[:, 2] + np.pi) % (2 * np.pi) - np.pi
            trajectory[i, j] = interp_traj
            max_dist = distance[-1]
            trajectory_mask[i, j, target_distance > max_dist] = 0.0

    return trajectory, trajectory_mask


def load_one(cache_path, log_name, token):
    data_path = os.path.join(cache_path, log_name, token, "sparsedrive_target.gz")
    data = load_feature_target_from_pickle(data_path)
    if data["path_mask"].all():
        path = np.array(data["path"], copy=True)
        path[:, 2] = (path[:, 2] + np.pi) % (2 * np.pi) - np.pi
    else:
        path = None
    velocity = data["velocity"]
    return path, velocity


def load_all_parallel(cache_path, max_workers=64):
    paths = []
    velocities = []
    tasks = []
    for log_name in os.listdir(cache_path):
        log_dir = os.path.join(cache_path, log_name)
        if not os.path.isdir(log_dir):
            continue
        for token in os.listdir(log_dir):
            token_dir = os.path.join(log_dir, token)
            if not os.path.isdir(token_dir):
                continue
            tasks.append((log_name, token))

    print(f"Loading {len(tasks)} cached scenes...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_one, cache_path, ln, tk) for ln, tk in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            path, velocity = fut.result()
            if path is not None:
                paths.append(path)
            velocities.append(velocity)

    return paths, velocities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default="exp/tacarla_cache")
    args = parser.parse_args()

    paths, velocities = load_all_parallel(args.cache_path)
    print(f"Total paths: {len(paths)}, velocities: {len(velocities)}")

    # Path clustering: (N, len_path, 3) -> flatten -> KMeans
    num_pts = paths[0].shape[0]
    paths_flatten = np.stack(paths).reshape(len(paths), -1)
    print(f"Running K-means on paths (K={K_PATH}, shape={paths_flatten.shape})...")
    path_cluster = KMeans(n_clusters=K_PATH, n_init=10, verbose=1).fit(paths_flatten).cluster_centers_
    path_cluster = path_cluster.reshape(K_PATH, num_pts, 3)
    path_cluster[:, :, 2] = (path_cluster[:, :, 2] + np.pi) % (2 * np.pi) - np.pi

    # Velocity clustering
    velocities = np.stack(velocities)
    print(f"Running K-means on velocities (K={K_VELOCITY}, shape={velocities.shape})...")
    velocity_cluster = KMeans(n_clusters=K_VELOCITY, n_init=10, verbose=1).fit(velocities).cluster_centers_

    # Compose trajectory vocabulary
    print("Composing trajectory vocabulary...")
    trajectory, trajectory_mask = interp_trajectory(path_cluster, velocity_cluster)

    # Save
    os.makedirs(CKPT_DIR, exist_ok=True)
    np.save(f"{CKPT_DIR}/path_{K_PATH}.npy", path_cluster)
    np.save(f"{CKPT_DIR}/velocity_{K_VELOCITY}.npy", velocity_cluster)
    np.savez(
        f"{CKPT_DIR}/trajectory_{K_PATH}_{K_VELOCITY}.npz",
        trajectory=trajectory,
        trajectory_mask=trajectory_mask,
    )
    print(f"Saved anchors to {CKPT_DIR}/")
    print(f"  path: ({K_PATH}, {num_pts}, 3)")
    print(f"  velocity: ({K_VELOCITY}, {velocities.shape[1]})")
    print(f"  trajectory: {trajectory.shape}")


if __name__ == "__main__":
    main()
