#!/usr/bin/env python3
"""
TaCarla dataset caching: parquet → cached features + targets.

NO image extraction needed. Cache stores camera parameters and image paths,
images are loaded on-the-fly during training in pipeline().

Usage:
    python scripts/tacarla/run_tacarla_caching.py \
        --label-dirs /path/to/TaCarla_labels/Town12_labels_hpc \
                     /path/to/TaCarla_labels/Town13_labels \
        --sensor-root /path/to/extracted_sensors \
        --camera-params /path/to/camera_params \
        --cache-path /path/to/tacarla_cache \
        [--cluster]    # also run K-means anchor clustering
        [--workers 16]
"""

import argparse
import gzip
import json
import os
import pickle
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

warnings.filterwarnings("ignore")

# ─── Camera setup ────────────────────────────────────────────────────────────

TACARLA_CAMS = ["front", "front_left", "front_right", "back", "back_left", "back_right"]
NAVSIM_CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0", "CAM_L1", "CAM_R1"]
CAM_MAP = dict(zip(TACARLA_CAMS, NAVSIM_CAMS))

CARLA_CMD = {
    1: np.array([1, 0, 0, 0], dtype=np.float32),  # LEFT
    2: np.array([0, 1, 0, 0], dtype=np.float32),  # RIGHT
    3: np.array([0, 0, 1, 0], dtype=np.float32),  # STRAIGHT
    4: np.array([0, 0, 0, 1], dtype=np.float32),  # FOLLOW_LANE
    5: np.array([1, 0, 0, 0], dtype=np.float32),  # CHANGE_LEFT
    6: np.array([0, 1, 0, 0], dtype=np.float32),  # CHANGE_RIGHT
}


def load_camera_params(path: Path) -> Dict[str, Any]:
    with open(path / "new_intrinsics_dict.json") as f:
        intr = json.load(f)
    with open(path / "new_extrinsics_dict.json") as f:
        extr = json.load(f)
    return {c: {"intrinsic": np.array(intr[c], dtype=np.float32),
                "extrinsic": np.array(extr[c], dtype=np.float32)} for c in TACARLA_CAMS}


# ─── Process one episode ─────────────────────────────────────────────────────

def process_episode(args):
    """Cache one episode. Returns (episode, scenes_cached, scenes_skipped)."""
    parquet_path, sensor_root, camera_params, cache_path, cfg = args
    episode = parquet_path.stem
    ep_cache = cache_path / episode

    subsample = cfg["subsample"]
    num_hist = cfg["num_history"]
    num_fut = cfg["num_future"]
    len_path = cfg["len_path"]
    path_interval = cfg["path_interval"]
    vel_dt = cfg["vel_time_interval"]
    num_frames = num_hist + num_fut

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        return (episode, 0, 0, f"parquet error: {e}")

    indices = list(range(0, len(df), subsample))
    if len(indices) < num_frames:
        return (episode, 0, 0, None)

    # Read only needed measurements
    meas = {i: df["measurements"].iloc[i] for i in indices}
    dt = subsample * 0.1

    # Build lightweight frame dicts
    frames = []
    for fi, ri in enumerate(indices):
        m = meas[ri]
        ego_mat = np.array(m["ego_matrix"].tolist(), dtype=np.float64)
        rot = R.from_matrix(ego_mat[:3, :3])
        q = rot.as_quat()  # x,y,z,w

        sp = float(m["speed"])
        th = float(m["theta"])
        vx, vy = sp * np.cos(th), sp * np.sin(th)
        if fi > 0:
            mp = meas[indices[fi - 1]]
            spp, thp = float(mp["speed"]), float(mp["theta"])
            ax, ay = (vx - spp * np.cos(thp)) / dt, (vy - spp * np.sin(thp)) / dt
        else:
            ax, ay = 0.0, 0.0

        cmd = CARLA_CMD.get(int(m.get("command", 4)), CARLA_CMD[4])

        cams = {}
        for tc, nv in CAM_MAP.items():
            raw = str(df[tc].iloc[ri]).strip()
            mat = re.search(r"(detection/.+)$", raw)
            rel = mat.group(1) if mat else raw
            cp = camera_params[tc]
            cams[nv] = {
                "data_path": str(sensor_root / episode / rel),
                "cam_intrinsic": cp["intrinsic"],
                "sensor2lidar_rotation": cp["extrinsic"][:3, :3].astype(np.float32),
                "sensor2lidar_translation": cp["extrinsic"][:3, 3].astype(np.float32),
                "distortion": np.zeros(5, dtype=np.float32),
            }

        frames.append({
            "token": f"{episode}_{ri}",
            "ego2global_translation": ego_mat[:3, 3].tolist(),
            "ego2global_rotation": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
            "ego_dynamic_state": [vx, vy, ax, ay],
            "driving_command": cmd,
            "cams": cams,
        })

    # Window + cache
    cached = 0
    skipped = 0
    for start in range(0, len(frames), num_frames):
        window = frames[start:start + num_frames]
        if len(window) < num_frames:
            break

        token = window[num_hist - 1]["token"]
        tp = ep_cache / token
        fp = tp / "sparsedrive_feature.gz"
        tgp = tp / "sparsedrive_target.gz"

        if fp.exists() and tgp.exists():
            skipped += 1
            continue

        tp.mkdir(parents=True, exist_ok=True)

        # ── Features ──
        sd = window[num_hist - 1]
        cam_keys = ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]
        cf = {}
        for k in cam_keys:
            uk = k.upper()
            if uk in sd["cams"]:
                ci = sd["cams"][uk]
                cf[k] = {"image_path": Path(ci["data_path"]),
                         "sensor2lidar_rotation": ci["sensor2lidar_rotation"],
                         "sensor2lidar_translation": ci["sensor2lidar_translation"],
                         "intrinsics": ci["cam_intrinsic"],
                         "distortion": ci["distortion"]}
            else:
                cf[k] = {}

        status = torch.cat([
            torch.tensor(sd["driving_command"], dtype=torch.float32),
            torch.tensor(sd["ego_dynamic_state"][:2], dtype=torch.float32),
            torch.tensor(sd["ego_dynamic_state"][2:], dtype=torch.float32),
        ])
        features = {"camera_feature": [cf], "status_feature": status}

        # ── Targets ──
        gposes = []
        for fd in window:
            t = fd["ego2global_translation"]
            quat = Quaternion(*fd["ego2global_rotation"])
            gposes.append(np.array([t[0], t[1], quat.yaw_pitch_roll[0]], dtype=np.float64))

        ref = gposes[num_hist - 1]
        lposes = convert_absolute_to_relative_se2_array(StateSE2(*ref), np.array(gposes, dtype=np.float64))

        traj = lposes[num_hist:num_hist + num_fut]
        trajectory = torch.tensor(traj, dtype=torch.float32)

        pad_t = np.concatenate([np.zeros((1, 2)), traj[:, :2]])
        velocity = torch.tensor(np.linalg.norm(pad_t[1:] - pad_t[:-1], axis=-1) / vel_dt, dtype=torch.float32)

        fut_poses = lposes[num_hist - 1:]
        d = [0.0] + [np.linalg.norm(fut_poses[j, :2] - fut_poses[j - 1, :2]) for j in range(1, len(fut_poses))]
        cum_d = np.cumsum(d)
        td = np.arange(1, len_path + 1) * path_interval
        path = np.array([np.interp(td, cum_d, fut_poses[:, i]) for i in range(3)]).T
        path[:, 2] = (path[:, 2] + np.pi) % (2 * np.pi) - np.pi
        path = torch.tensor(path, dtype=torch.float32)

        pmask = torch.ones(len_path, dtype=torch.float32)
        valid = min(len_path, int(np.floor(cum_d[-1] / path_interval)))
        pmask[valid:] = 0

        targets = {"trajectory": trajectory, "path": path, "path_mask": pmask, "velocity": velocity}

        with gzip.open(fp, "wb", compresslevel=1) as f:
            pickle.dump(features, f)
        with gzip.open(tgp, "wb", compresslevel=1) as f:
            pickle.dump(targets, f)
        cached += 1

    return (episode, cached, skipped, None)


# ─── Clustering ──────────────────────────────────────────────────────────────

def _load_one_target(gz_path):
    """Load one target .gz file. Returns (path_or_None, velocity)."""
    with gzip.open(gz_path, "rb") as fh:
        data = pickle.load(fh)
    vel = np.array(data["velocity"], dtype=np.float32)
    if data["path_mask"].all():
        p = np.array(data["path"], dtype=np.float32)
        p[:, 2] = (p[:, 2] + np.pi) % (2 * np.pi) - np.pi
        return p, vel
    return None, vel


def _gpu_kmeans(data: np.ndarray, k: int, n_iter: int = 100, n_init: int = 10):
    """K-means on GPU using PyTorch. Much faster than sklearn for large data."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(data).float().to(device)
    n = X.shape[0]
    best_centers = None
    best_inertia = float("inf")

    for attempt in range(n_init):
        # K-means++ init
        idx = [torch.randint(n, (1,)).item()]
        for _ in range(1, k):
            dists = torch.cdist(X, X[idx]).min(dim=1).values
            probs = dists ** 2
            probs /= probs.sum()
            idx.append(torch.multinomial(probs, 1).item())
        centers = X[idx].clone()

        for _ in range(n_iter):
            dists = torch.cdist(X, centers)
            labels = dists.argmin(dim=1)
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(k, device=device)
            new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, X.shape[1]), X)
            counts.scatter_add_(0, labels, torch.ones(n, device=device))
            mask = counts > 0
            new_centers[mask] /= counts[mask].unsqueeze(1)
            new_centers[~mask] = centers[~mask]
            if torch.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        inertia = torch.cdist(X, centers).min(dim=1).values.pow(2).sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers

        if (attempt + 1) % 3 == 0:
            print(f"    init {attempt+1}/{n_init}, inertia={best_inertia:.2f}")

    return best_centers.cpu().numpy()


def run_clustering(cache_path: Path):
    K_PATH, K_VEL, DT = 1024, 256, 0.5
    out = Path("ckpt/kmeans_tacarla")

    # ── Step 1: Parallel loading from cache ──
    # Use subprocess find for fast file discovery on network storage
    print("[Cluster] Scanning cache files (using find)...")
    import subprocess
    result = subprocess.run(
        ["find", str(cache_path), "-name", "sparsedrive_target.gz", "-type", "f"],
        capture_output=True, text=True, timeout=600,
    )
    gz_files = [Path(p) for p in result.stdout.strip().split("\n") if p]
    print(f"[Cluster] Found {len(gz_files)} cached targets, loading with {min(32, len(gz_files))} workers...")

    paths, vels = [], []
    with ProcessPoolExecutor(max_workers=32) as pool:
        futs = [pool.submit(_load_one_target, f) for f in gz_files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="[Cluster] load"):
            p, v = fut.result()
            if p is not None:
                paths.append(p)
            vels.append(v)

    print(f"[Cluster] Loaded {len(paths)} paths, {len(vels)} velocities")

    # ── Step 2: GPU K-means ──
    npts = paths[0].shape[0]
    pf = np.stack(paths).reshape(len(paths), -1)
    print(f"[Cluster] GPU K-means on paths ({K_PATH} clusters, {pf.shape})...")
    pc = _gpu_kmeans(pf, K_PATH, n_iter=100, n_init=10).reshape(K_PATH, npts, 3)
    pc[:, :, 2] = (pc[:, :, 2] + np.pi) % (2 * np.pi) - np.pi

    vs = np.stack(vels)
    print(f"[Cluster] GPU K-means on velocities ({K_VEL} clusters, {vs.shape})...")
    vc = _gpu_kmeans(vs, K_VEL, n_iter=100, n_init=10)
    nv = vc.shape[1]

    # ── Step 3: Compose trajectory vocabulary (vectorized) ──
    print(f"[Cluster] Composing trajectory vocab ({K_PATH}x{K_VEL}x{nv}x3)...")
    traj = np.zeros((K_PATH, K_VEL, nv, 3))
    tmask = np.ones((K_PATH, K_VEL, nv))
    for i in range(K_PATH):
        pad = np.vstack([np.zeros(3), pc[i]])
        d = np.r_[0, np.linalg.norm(pad[1:, :2] - pad[:-1, :2], axis=-1).cumsum()]
        for j in range(K_VEL):
            td = np.cumsum(vc[j] * DT)
            t = np.array([np.interp(td, d, pad[:, k]) for k in range(3)]).T
            t[:, 2] = (t[:, 2] + np.pi) % (2 * np.pi) - np.pi
            traj[i, j] = t
            tmask[i, j, td > d[-1]] = 0.0

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"path_{K_PATH}.npy", pc)
    np.save(out / f"velocity_{K_VEL}.npy", vc)
    np.savez(out / f"trajectory_{K_PATH}_{K_VEL}.npz", trajectory=traj, trajectory_mask=tmask)
    print(f"[Cluster] Saved to {out}/")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dirs", nargs="+", required=True)
    parser.add_argument("--sensor-root", required=True)
    parser.add_argument("--camera-params", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--subsample", type=int, default=5)
    args = parser.parse_args()

    cam_params = load_camera_params(Path(args.camera_params))
    sensor_root = Path(args.sensor_root)
    cache_path = Path(args.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    parquets = []
    for ld in args.label_dirs:
        parquets.extend(sorted(Path(ld).glob("*.parquet")))
    print(f"Total parquet files: {len(parquets)}")

    # Pre-filter: skip episodes already cached (one fast listdir, no per-file checks)
    cached_episodes = set()
    if cache_path.exists():
        cached_episodes = set(d.name for d in cache_path.iterdir() if d.is_dir())
    to_process = [p for p in parquets if p.stem not in cached_episodes]
    already_done = len(parquets) - len(to_process)
    print(f"Already cached: {already_done}, to process: {len(to_process)}")

    cfg = {"subsample": args.subsample, "num_history": 1, "num_future": 6,
           "len_path": 15, "path_interval": 1.0, "vel_time_interval": 0.5}

    tasks = [(p, sensor_root, cam_params, cache_path, cfg) for p in to_process]

    total_cached = 0
    total_skipped = 0
    total_errors = 0

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(process_episode, t): t[0].stem for t in tasks}
            with tqdm(total=len(tasks), desc="Caching episodes", unit="ep") as pbar:
                for fut in as_completed(futs):
                    ep, nc, ns, err = fut.result()
                    total_cached += nc
                    total_skipped += ns
                    if err:
                        total_errors += 1
                        tqdm.write(f"  ERR {ep}: {err}")
                    pbar.update(1)
                    pbar.set_postfix(cached=total_cached, skip=total_skipped, err=total_errors)

    print(f"\nDone: {total_cached} new scenes, {already_done} episodes already cached, {total_errors} errors")

    if args.cluster:
        print()
        run_clustering(cache_path)

    print("\nAll complete!")


if __name__ == "__main__":
    main()
