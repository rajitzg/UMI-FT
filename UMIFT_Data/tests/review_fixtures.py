"""Synthetic UMIFT episode zarrs for the episode review tool tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

UMIFT_DATA = Path(__file__).resolve().parents[1]
if str(UMIFT_DATA) not in sys.path:
    sys.path.insert(0, str(UMIFT_DATA))

META_LEN_KEYS = {
    "episode_rgb0_len": "rgb_0",
    "episode_ultrawide0_len": "ultrawide_0",
    "episode_depth0_len": "depth_0",
    "episode_robot0_len": "ts_pose_fb_0",
    "episode_gripper0_len": "gripper_0",
    "episode_wrench0_len": "wrench_left_0",
}


def _arc_pose7(t: np.ndarray) -> np.ndarray:
    """Smooth arc in x/z at constant y, slow yaw; pose7 [x,y,z,qw,qx,qy,qz]."""
    x = 0.2 * np.sin(0.2 * t)
    z = 0.1 * np.cos(0.2 * t)
    y = np.full_like(t, 0.05)
    yaw = 0.1 * t
    qw, qy = np.cos(yaw / 2), np.sin(yaw / 2)
    zeros = np.zeros_like(t)
    return np.stack([x, y, z, qw, zeros, qy, zeros], axis=1)


def write_synthetic_episode_zarr(
    path,
    n_eps: int = 2,
    n_rgb: int = 60,
    robot_hz: float = 200.0,
    rgb_hz: float = 15.0,
    wrench_hz: float = 28.0,
    start_pad_s: float = 0.0,
    scan_gaps=None,
    image_hw: int = 224,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(path), mode="w")
    root.attrs["start_pad_seconds"] = float(start_pad_s)
    data = root.create_group("data")
    meta = root.create_group("meta")
    cum = {k: [] for k in META_LEN_KEYS}
    for e in range(n_eps):
        n = n_rgb + e  # different lengths per episode
        duration = (n - 1) / rgb_hz
        t_rgb = np.arange(n) / rgb_hz
        t_rob = np.arange(0.0, duration + 1e-9, 1.0 / robot_hz)
        t_wr = np.arange(0.0, duration + 1e-9, 1.0 / wrench_hz)
        ep = data.create_group(f"episode_{e}")
        rng = np.random.default_rng(e)
        rgb = rng.integers(0, 255, size=(n, image_hw, image_hw, 3), dtype=np.uint8)
        ep.create_dataset("rgb_0", data=rgb, chunks=(208, 56, 56, 3))
        ep.create_dataset("ultrawide_0", data=np.zeros((n, 4, 4, 3), np.uint8))
        ep.create_dataset("depth_0", data=np.zeros((n, 4, 4, 3), np.float16))
        for k in ("rgb", "ultrawide", "depth"):
            ep.create_dataset(f"{k}_time_stamps_0", data=t_rgb[:, None])
        ep.create_dataset("map_to_uw_idx_0", data=np.arange(n)[:, None])
        ep.create_dataset("map_to_d_idx_0", data=np.arange(n)[:, None])
        pose = _arc_pose7(t_rob)
        ep.create_dataset("ts_pose_fb_0", data=pose)
        ep.create_dataset("ts_pose_command_0", data=pose)
        vt = pose.copy()
        vt[:, 1] += 0.01
        ep.create_dataset("ts_pose_virtual_target_0", data=vt)
        ep.create_dataset("stiffness_0", data=np.full(len(t_rob), 500.0))
        ep.create_dataset("robot_time_stamps_0", data=t_rob[:, None])
        ep.create_dataset("gripper_0", data=np.zeros((len(t_rob), 1)))
        ep.create_dataset("gripper_time_stamps_0", data=t_rob[:, None])
        wr = np.zeros((len(t_wr), 6))
        wr[:, 1] = 5.0 + np.sin(t_wr)
        for k in ("wrench_left_0", "wrench_left_coinft_0"):
            ep.create_dataset(k, data=wr)
        for k in ("wrench_right_0", "wrench_right_coinft_0"):
            ep.create_dataset(k, data=np.zeros_like(wr))
        ep.create_dataset("wrench_concat_0", data=np.concatenate([wr, 0 * wr], 1))
        ep.create_dataset("wrench_concat_coinft_0", data=np.concatenate([wr, 0 * wr], 1))
        for k in ("wrench_time_stamps_0", "wrench_time_stamps_left_0", "wrench_time_stamps_right_0"):
            ep.create_dataset(k, data=t_wr[:, None])
        if scan_gaps is not None:
            ep.create_dataset("scan_gap_intervals_0", data=np.asarray(scan_gaps, float).reshape(-1, 2))
        for mk, key in META_LEN_KEYS.items():
            prev = cum[mk][-1] if cum[mk] else 0
            cum[mk].append(prev + ep[key].shape[0])
    for mk, vals in cum.items():
        meta.create_dataset(mk, data=np.asarray(vals, np.int64))
    return path
