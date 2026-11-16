#!/usr/bin/env python3
"""
Convert unified T265 + wrist FT + D405 recordings into UMIFT-compatible zarr.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import zarr
from numcodecs import Blosc

UMIFT_DATA_ROOT = Path(__file__).resolve().parents[1]
if str(UMIFT_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(UMIFT_DATA_ROOT))

from umift.processing.pad_episode_start import pad_episode_start
from umift.processing.umift_image_prep import (
    resize_hwc_float,
    resize_hwc_uint8,
    resample_episode_to_hz,
    build_map_to_scan_idx,
)
from umift.utils.frame_conventions import (
    DEFAULT_R_FT_IN_TCP_M,
    FRAME_CONVENTION_VERSION,
    check_tcp_pose_consistency,
    conversion_info_json,
    ft_poses_to_tcp_poses,
    ft_wrench_to_tcp_wrench,
)


def _ensure_zarr_v2():
    version = getattr(zarr, "__version__", "unknown")
    if not hasattr(zarr, "DirectoryStore"):
        raise RuntimeError(
            "This converter is implemented for zarr v2 APIs (UMIFT-consistent). "
            f"Current zarr={version} does not expose DirectoryStore. "
            "Install zarr==2.18.3 and numcodecs==0.15.1."
        )


def _as_col(ts: np.ndarray) -> np.ndarray:
    return np.asarray(ts, dtype=np.float64).reshape(-1, 1)


def normalize_timestamps(streams: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Shift all stream timestamps by the smallest first sample."""
    first_samples = []
    for arr in streams.values():
        if arr.size == 0:
            continue
        first_samples.append(float(arr.reshape(-1)[0]))
    if not first_samples:
        return streams

    offset = min(first_samples)
    return {k: np.asarray(v, dtype=np.float64) - offset for k, v in streams.items()}


def _detect_frame_convention(root) -> str:
    """Read stamped convention from unified zarr attrs/manifest; default legacy."""
    attrs = getattr(root, "attrs", {}) or {}
    for key in ("frame_convention", "frame_convention_version"):
        if key in attrs:
            return str(attrs[key])
    try:
        manifest = attrs.get("manifest")
        if manifest:
            if isinstance(manifest, str):
                manifest = json.loads(manifest)
            if isinstance(manifest, dict) and "frame_convention" in manifest:
                return str(manifest["frame_convention"])
    except Exception:
        pass
    return "legacy"


def build_ts_pose_fb_legacy(
    odom: Dict[str, np.ndarray],
    sensor_offset_xyz_m: np.ndarray = np.array([0.0, 0.0, -0.02], dtype=np.float64),
) -> np.ndarray:
    """LEGACY: FT-frame / umift_tcp_v1 odom → TCP via ft_poses_to_tcp_poses."""
    positions = np.column_stack([odom["pos_x"], odom["pos_y"], odom["pos_z"]])
    orientations_wxyz = np.column_stack(
        [odom["orient_w"], odom["orient_x"], odom["orient_y"], odom["orient_z"]]
    )
    return ft_poses_to_tcp_poses(
        positions_ft=positions,
        orientations_wxyz=orientations_wxyz,
        sensor_offset_xyz_m=sensor_offset_xyz_m,
    )


def build_ts_pose_fb_v2(odom: Dict[str, np.ndarray]) -> np.ndarray:
    """umift_std_v2: odom already TCP in odom_std — identity copy to pose7."""
    return np.column_stack(
        [
            odom["pos_x"],
            odom["pos_y"],
            odom["pos_z"],
            odom["orient_w"],
            odom["orient_x"],
            odom["orient_y"],
            odom["orient_z"],
        ]
    ).astype(np.float64)


def resolve_tag_traj_path(unified_path: Path) -> Path | None:
    """<folder>/tag_traj/<prefix>_ee_master.npz for <folder>/zarr/<prefix>_unified.zarr."""
    unified_path = Path(unified_path)
    prefix = unified_path.name.replace("_unified.zarr", "").replace(".zarr", "")
    parent = unified_path.parent
    candidates = []
    if parent.name == "zarr":
        candidates.append(parent.parent / "tag_traj" / f"{prefix}_ee_master.npz")
    candidates.append(parent / "tag_traj" / f"{prefix}_ee_master.npz")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _dedupe_strictly_increasing_times(
    src_t: np.ndarray, pose7: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the last sample at each timestamp so SciPy Slerp accepts the series.

    Unified RGB times occasionally repeat (same stamp on consecutive frames);
    ee_tag_traj sidecars inherit that and otherwise crash combine.
    """
    src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
    pose7 = np.asarray(pose7, dtype=np.float64)
    if src_t.size == 0:
        return src_t, pose7
    # Last index of each unique time (stable for flat plateaus).
    _, first_rev = np.unique(src_t[::-1], return_index=True)
    keep = np.sort(src_t.size - 1 - first_rev)
    t_u = src_t[keep]
    # Guard against any remaining non-increase from float weirdness.
    if t_u.size >= 2 and not np.all(np.diff(t_u) > 0):
        for i in range(1, t_u.size):
            if t_u[i] <= t_u[i - 1]:
                t_u[i] = t_u[i - 1] + 1e-6
    return t_u, pose7[keep]


def _resample_pose7(
    src_t: np.ndarray,
    pose7: np.ndarray,
    query_t: np.ndarray,
) -> np.ndarray:
    """Linear pos + SLERP quat onto query times. pose7 is [x,y,z,qw,qx,qy,qz]."""
    from scipy.spatial.transform import Rotation, Slerp

    src_t, pose7 = _dedupe_strictly_increasing_times(src_t, pose7)
    query_t = np.asarray(query_t, dtype=np.float64).reshape(-1)
    out = np.zeros((len(query_t), 7), dtype=np.float64)
    if src_t.size == 0:
        raise ValueError("empty tag traj timestamps")
    if src_t.size == 1:
        out[:] = pose7[0]
        return out
    for j in range(3):
        out[:, j] = np.interp(query_t, src_t, pose7[:, j])
    # scipy Rotation uses xyzw
    quat_xyzw = np.column_stack([pose7[:, 4], pose7[:, 5], pose7[:, 6], pose7[:, 3]])
    rots = Rotation.from_quat(quat_xyzw)
    t0, t1 = float(src_t[0]), float(src_t[-1])
    t_clip = np.clip(query_t, t0, t1)
    slerp = Slerp(src_t, rots)
    q = slerp(t_clip).as_quat()  # xyzw
    out[:, 3] = q[:, 3]
    out[:, 4] = q[:, 0]
    out[:, 5] = q[:, 1]
    out[:, 6] = q[:, 2]
    return out


def load_tag_traj_pose7(sidecar: Path, query_t_unified: np.ndarray) -> np.ndarray:
    """Load already-TCP pose7 from ee_tag_traj sidecar; resample to query times."""
    data = np.load(sidecar)
    src_t = np.asarray(data["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
    pose7 = np.asarray(data["pose7"], dtype=np.float64)
    return _resample_pose7(src_t, pose7, query_t_unified)


def _apply_wrench_point_offset(
    wrench: np.ndarray, sensor_offset_xyz_m: np.ndarray
) -> np.ndarray:
    """
    Shift wrench from sensor point to target point using:
        tau_target = tau_sensor + r x F
    where r is sensor offset vector (target -> sensor) in meters.
    """
    r = np.asarray(sensor_offset_xyz_m, dtype=np.float64).reshape(3)
    if np.allclose(r, 0.0):
        return wrench
    adjusted = np.array(wrench, copy=True)
    force = adjusted[:, 0:3]
    adjusted[:, 3:6] = adjusted[:, 3:6] + np.cross(np.broadcast_to(r, force.shape), force)
    return adjusted


def build_wrench_keys(
    ft: Dict[str, np.ndarray],
    sensor_offset_xyz_m: np.ndarray = np.zeros(3),
    *,
    frame_convention: str = "legacy",
) -> Dict[str, np.ndarray]:
    """Build UMIFT wrench keys. v2: already TCP (identity). Legacy: FT→TCP remap."""
    wrench = np.column_stack(
        [
            ft["force_x"],
            ft["force_y"],
            ft["force_z"],
            ft["torque_x"],
            ft["torque_y"],
            ft["torque_z"],
        ]
    ).astype(np.float64)
    if frame_convention == FRAME_CONVENTION_VERSION or frame_convention == "umift_std_v2":
        left = wrench  # already TCP from FT publisher
    else:
        left = ft_wrench_to_tcp_wrench(wrench)
        left = _apply_wrench_point_offset(left, sensor_offset_xyz_m=sensor_offset_xyz_m)
    right = np.zeros_like(left)
    concat = np.hstack((left, right))

    return {
        "wrench_left_0": left,
        "wrench_right_0": right,
        "wrench_concat_0": concat,
        "wrench_left_coinft_0": left,
        "wrench_right_coinft_0": right,
        "wrench_concat_coinft_0": concat,
    }


def convert_depth_to_umift_depth(
    depth_mm: np.ndarray, depth_scale: float = 0.0001, depth_clip: float = 0.5
) -> np.ndarray:
    """
    Convert depth from uint16 mm to float16 meters and expand to 3 channels.
    """
    depth_m = np.asarray(depth_mm, dtype=np.float32) * depth_scale
    depth_m = np.clip(depth_m, 0.0, depth_clip).astype(np.float16)
    return np.repeat(depth_m[..., None], 3, axis=-1)


def _read_ts(group, rel_path: str) -> np.ndarray:
    arr = np.asarray(group[rel_path][:], dtype=np.float64).reshape(-1)
    return arr


def _read_frame_ts(camera_group) -> np.ndarray:
    if "timestamp_sec_relative" in camera_group:
        return _read_ts(camera_group, "timestamp_sec_relative")
    if "timestamp_sec" in camera_group:
        return _read_ts(camera_group, "timestamp_sec")
    if "timestamp_ms" in camera_group:
        return _read_ts(camera_group, "timestamp_ms") * 1e-3
    raise KeyError("camera stream must contain timestamp_sec_relative/sec/ms")


def _build_index_map(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    if len(target_ts) == 0:
        return np.zeros((len(source_ts), 1), dtype=np.int64)
    idx = np.searchsorted(target_ts, source_ts, side="left")
    idx = np.clip(idx, 0, len(target_ts) - 1)
    return idx.reshape(-1, 1).astype(np.int64)


def _open_unified_root(path: Path):
    # Works with zarr v3 environments where these unified datasets are produced.
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception as exc:
        raise RuntimeError(
            f"Unable to open unified zarr at {path}. "
            "This dataset format requires a zarr runtime that can read zarr v3 groups "
            "(for example: pip install 'zarr>=3')."
        ) from exc


def _read_episode_from_unified(
    path: Path,
    sensor_offset_xyz_m: np.ndarray = np.zeros(3),
    pose_source: str = "auto",
) -> tuple[Dict[str, np.ndarray], str, str]:
    """
    Returns (episode_dict, pose_source_used, frame_convention).

    pose_source: 'auto' | 't265' | 'tag'
      auto = tag sidecar if present else t265
    umift_std_v2: pose/wrench already TCP — identity copy (no remap).
    """
    root = _open_unified_root(path)
    frame_convention = _detect_frame_convention(root)
    is_v2 = frame_convention in (FRAME_CONVENTION_VERSION, "umift_std_v2")

    color_group = root["camera"]["color"]
    depth_group = root["camera"]["depth"]
    odom_group = root["sensors"]["t265_odom"]
    ft_group = root["sensors"]["ft_wrench"]

    rgb = np.asarray(color_group["frames"][:], dtype=np.uint8)
    # Schema placeholder only; filled with zeros in _postprocess_episode (not D405 color).
    ultrawide = np.zeros_like(rgb)
    depth = convert_depth_to_umift_depth(np.asarray(depth_group["frames"][:], dtype=np.uint16))

    rgb_ts = _read_frame_ts(color_group)
    uw_ts = np.copy(rgb_ts)
    depth_ts = _read_frame_ts(depth_group)
    robot_ts = _read_ts(odom_group, "timestamp_sec_relative")
    wrench_ts = _read_ts(ft_group, "timestamp_sec_relative")

    odom = {k: np.asarray(odom_group[k][:], dtype=np.float64) for k in [
        "pos_x",
        "pos_y",
        "pos_z",
        "orient_x",
        "orient_y",
        "orient_z",
        "orient_w",
    ]}

    sidecar = resolve_tag_traj_path(path)
    use_tag = pose_source == "tag" or (pose_source == "auto" and sidecar is not None)
    if pose_source == "tag" and sidecar is None:
        raise FileNotFoundError(f"--pose-source tag but no sidecar for {path}")
    if use_tag:
        if "timestamp_sec_unified" in odom_group:
            odom_unified = _read_ts(odom_group, "timestamp_sec_unified").reshape(-1)
        else:
            odom_unified = _read_ts(odom_group, "timestamp_sec_relative").reshape(-1)
            if "timestamp_sec_unified" in color_group:
                rgb_u = np.asarray(color_group["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
                odom_unified = odom_unified - odom_unified[0] + rgb_u[0]
        pose = load_tag_traj_pose7(sidecar, odom_unified)
        pose_source_used = "tag_fixture_tcp"
        print(f"[convert] Using tag traj sidecar {sidecar.name} for {path.name}")
    elif is_v2:
        pose = build_ts_pose_fb_v2(odom)
        pose_source_used = "t265_tcp_odom_std_v2"
        print(f"[convert] umift_std_v2 identity pose copy for {path.name}")
    else:
        pose = build_ts_pose_fb_legacy(odom, sensor_offset_xyz_m=sensor_offset_xyz_m)
        pose_source_used = "t265_odom_ft_to_tcp_legacy"
        print(f"[convert] LEGACY ft_poses_to_tcp_poses for {path.name}")

    consistency = check_tcp_pose_consistency(pose)
    print(f"[convert] TCP pose consistency ({path.name}): {consistency}")

    ft = {k: np.asarray(ft_group[k][:], dtype=np.float64) for k in [
        "force_x",
        "force_y",
        "force_z",
        "torque_x",
        "torque_y",
        "torque_z",
    ]}
    wrench_keys = build_wrench_keys(
        ft,
        sensor_offset_xyz_m=sensor_offset_xyz_m,
        frame_convention=frame_convention if is_v2 else "legacy",
    )

    ts_streams = {
        "rgb_time_stamps_0": rgb_ts,
        "ultrawide_time_stamps_0": uw_ts,
        "depth_time_stamps_0": depth_ts,
        "robot_time_stamps_0": robot_ts,
        "wrench_time_stamps_left_0": wrench_ts,
        "wrench_time_stamps_right_0": wrench_ts,
    }
    ts_norm = normalize_timestamps(ts_streams)
    # normalize_timestamps() shifts every stream by the same scalar (the smallest
    # first-sample time across `ts_streams`) but only returns the shifted arrays, not
    # the offset itself. Recompute it here with the identical formula (same input dict)
    # so scan_gap_intervals_0 below -- read in stage 1's own relative-time convention,
    # which is the same convention as `rgb_ts` *before* this shift -- can be rebased
    # onto the exact same zero point as ts_norm["rgb_time_stamps_0"].
    _first_samples = [
        float(np.asarray(v).reshape(-1)[0]) for v in ts_streams.values() if np.asarray(v).size > 0
    ]
    ts_norm_offset = min(_first_samples) if _first_samples else 0.0

    robot_len = pose.shape[0]
    gripper = np.zeros((robot_len, 1), dtype=np.float64)
    gripper_ts = _as_col(ts_norm["robot_time_stamps_0"])

    # Laser-scan heatmaps (optional; see convert_recording_to_zarr.py stage 1's
    # `scans` group). Absent for episodes with zero completed scans -- treat that as a
    # clean no-op (no keys added), matching stage 1's own "no group written" convention.
    scan_arrays: Dict[str, np.ndarray] = {}
    if "scans" in root:
        scans_group = root["scans"]
        heatmap = np.asarray(scans_group["scan_heatmap_0"][:], dtype=np.float32)  # (N, 80, 160)
        scan_gap_intervals_stage1 = np.asarray(
            scans_group["scan_gap_intervals_0"][:], dtype=np.float64
        )  # (N, 2), stage 1's relative-time convention (see _build_scan_gap_intervals)

        # scan_heatmap_0: add a channel dim here (replicate the single channel to 3) to
        # match the training schema's declared [3, 80, 160] (C,H,W) shape / the THWC
        # convention obs_rgb_depth_preprocess expects at read time (it does
        # `t, hi, wi, ci = this_scan_in.shape`). Stage 1's output is a general-purpose
        # (N, H, W) grid with no channel axis -- channel replication is a training-schema
        # concern, so it belongs here, not in stage 1. NaN is preserved (NOT nan_to_num'd
        # here); obs_rgb_depth_preprocess does its own nan_to_num after clip+normalize.
        scan_heatmap = np.repeat(heatmap[..., None], 3, axis=-1).astype(np.float32)  # (N, 80, 160, 3)

        # Rebase stage 1's scan_gap_intervals_0 onto ts_norm's zero point (the same
        # ts_norm_offset normalize_timestamps applied to rgb_time_stamps_0 above) so it's
        # directly comparable to ts_norm["rgb_time_stamps_0"] and to the map_to_scan_idx_0
        # we build next.
        scan_gap_intervals = (scan_gap_intervals_stage1 - ts_norm_offset).astype(np.float32)

        # map_to_scan_idx_0 is rebuilt fresh against THIS stage's own reference timeline
        # (ts_norm["rgb_time_stamps_0"]), not copied from stage 1: normalize_timestamps
        # always re-bases by ts_norm_offset (not guaranteed to be 0 -- e.g. if depth or
        # robot happens to have the smallest first sample, not rgb), so stage 1's
        # map_to_scan_idx_0 (built against stage 1's own reference-camera stream) is not
        # guaranteed valid against ts_norm's timeline even when lengths happen to match.
        map_to_scan_idx = build_map_to_scan_idx(ts_norm["rgb_time_stamps_0"], scan_gap_intervals)
        assert map_to_scan_idx.shape[0] == ts_norm["rgb_time_stamps_0"].shape[0], (
            "map_to_scan_idx_0 length must match rgb_time_stamps_0 length "
            f"({map_to_scan_idx.shape[0]} != {ts_norm['rgb_time_stamps_0'].shape[0]})"
        )

        scan_arrays = {
            "scan_heatmap_0": scan_heatmap,
            "map_to_scan_idx_0": map_to_scan_idx,
            "scan_gap_intervals_0": scan_gap_intervals,
        }

    return {
        "rgb_0": rgb,
        "ultrawide_0": ultrawide,
        "depth_0": depth,
        "map_to_uw_idx_0": _build_index_map(ts_norm["rgb_time_stamps_0"], ts_norm["ultrawide_time_stamps_0"]),
        "map_to_d_idx_0": _build_index_map(ts_norm["rgb_time_stamps_0"], ts_norm["depth_time_stamps_0"]),
        "ts_pose_fb_0": pose,
        "gripper_0": gripper,
        "rgb_time_stamps_0": _as_col(ts_norm["rgb_time_stamps_0"]),
        "ultrawide_time_stamps_0": _as_col(ts_norm["ultrawide_time_stamps_0"]),
        "depth_time_stamps_0": _as_col(ts_norm["depth_time_stamps_0"]),
        "robot_time_stamps_0": _as_col(ts_norm["robot_time_stamps_0"]),
        "gripper_time_stamps_0": gripper_ts,
        "wrench_time_stamps_left_0": _as_col(ts_norm["wrench_time_stamps_left_0"]),
        "wrench_time_stamps_right_0": _as_col(ts_norm["wrench_time_stamps_right_0"]),
        **wrench_keys,
        **scan_arrays,
    }, pose_source_used, frame_convention


def _episode_len(cumulative: List[int], this_len: int) -> None:
    cumulative.append((cumulative[-1] if cumulative else 0) + int(this_len))


def _postprocess_episode(
    ep_data: Dict[str, np.ndarray], resample_10hz: bool = False
) -> Dict[str, np.ndarray]:
    """Resize rgb/depth to 224; zero ultrawide; optionally resample episode to 10 Hz."""
    ep_data["rgb_0"] = resize_hwc_uint8(ep_data["rgb_0"], (224, 224))
    ep_data["ultrawide_0"] = np.zeros_like(ep_data["rgb_0"])
    ep_data["depth_0"] = resize_hwc_float(ep_data["depth_0"], (224, 224))
    if resample_10hz:
        ep_data = resample_episode_to_hz(ep_data, hz=10.0)
    return ep_data


def _input_unified_paths(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.name.endswith(".zarr"):
        return [input_path]
    if input_path.is_dir() and input_path.name.endswith(".zarr"):
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*_unified.zarr"))
    return []


def convert(
    input_path: Path,
    output_path: Path,
    sensor_offset_xyz_m: np.ndarray = np.zeros(3),
    pose_source: str = "auto",
    resample_10hz: bool = False,
    start_pad_seconds: float = 5.0,
    dagger_pattern: Optional[str] = None,
) -> None:
    _ensure_zarr_v2()

    unified_paths = _input_unified_paths(input_path)
    if not unified_paths:
        raise FileNotFoundError(f"No unified zarr found under: {input_path}")

    numeric_compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    image_compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    episode_rgb_len = []
    episode_uw_len = []
    episode_depth_len = []
    episode_robot_len = []
    episode_gripper_len = []
    episode_wrench_len = []
    pose_sources_used: List[str] = []
    conventions_used: List[str] = []

    for ep_idx, unified_path in enumerate(unified_paths):
        ep_data, pose_used, frame_conv = _read_episode_from_unified(
            unified_path,
            sensor_offset_xyz_m=sensor_offset_xyz_m,
            pose_source=pose_source,
        )
        ep_data = _postprocess_episode(ep_data, resample_10hz=resample_10hz)
        ep_data = pad_episode_start(ep_data, pad_duration_s=start_pad_seconds)
        pose_sources_used.append(pose_used)
        conventions_used.append(frame_conv)
        ep_group = data_group.create_group(f"episode_{ep_idx}")
        ep_group.attrs["start_pad_seconds"] = float(start_pad_seconds)
        # "dagger" if the unified zarr name contains --dagger-pattern (see UmiFTDataset dagger_frac)
        is_dagger = bool(dagger_pattern) and dagger_pattern in unified_path.name
        ep_group.attrs["source"] = "dagger" if is_dagger else "original"

        for key, value in ep_data.items():
            compressor = image_compressor if key in {"rgb_0", "ultrawide_0"} else numeric_compressor
            ep_group.create_dataset(
                name=key,
                data=value,
                shape=value.shape,
                chunks=True,
                dtype=value.dtype,
                compressor=compressor,
            )

        _episode_len(episode_rgb_len, ep_data["rgb_0"].shape[0])
        _episode_len(episode_uw_len, ep_data["ultrawide_0"].shape[0])
        _episode_len(episode_depth_len, ep_data["depth_0"].shape[0])
        _episode_len(episode_robot_len, ep_data["ts_pose_fb_0"].shape[0])
        _episode_len(episode_gripper_len, ep_data["gripper_0"].shape[0])
        _episode_len(episode_wrench_len, ep_data["wrench_left_0"].shape[0])

        print(f"[convert] episode_{ep_idx} <- {unified_path.name} (pose={pose_used})")

    meta_group.create_dataset(
        "episode_rgb0_len",
        data=np.asarray(episode_rgb_len, dtype=np.int64),
        compressor=numeric_compressor,
    )
    meta_group.create_dataset(
        "episode_ultrawide0_len",
        data=np.asarray(episode_uw_len, dtype=np.int64),
        compressor=numeric_compressor,
    )
    meta_group.create_dataset(
        "episode_depth0_len",
        data=np.asarray(episode_depth_len, dtype=np.int64),
        compressor=numeric_compressor,
    )
    meta_group.create_dataset(
        "episode_robot0_len",
        data=np.asarray(episode_robot_len, dtype=np.int64),
        compressor=numeric_compressor,
    )
    meta_group.create_dataset(
        "episode_gripper0_len",
        data=np.asarray(episode_gripper_len, dtype=np.int64),
        compressor=numeric_compressor,
    )
    meta_group.create_dataset(
        "episode_wrench0_len",
        data=np.asarray(episode_wrench_len, dtype=np.int64),
        compressor=numeric_compressor,
    )

    dominant_pose = (
        "tag_fixture_tcp"
        if pose_sources_used and all(p == "tag_fixture_tcp" for p in pose_sources_used)
        else (
            "mixed"
            if len(set(pose_sources_used)) > 1
            else (pose_sources_used[0] if pose_sources_used else "unknown")
        )
    )
    all_v2 = bool(conventions_used) and all(
        c in (FRAME_CONVENTION_VERSION, "umift_std_v2") for c in conventions_used
    )
    world_frame = (
        "fixture"
        if dominant_pose == "tag_fixture_tcp"
        else ("odom" if all_v2 else "+Y up (legacy)")
    )

    root.attrs["image_size_hw"] = [224, 224]
    root.attrs["resample_hz"] = 10.0 if resample_10hz else None
    root.attrs["start_pad_seconds"] = float(start_pad_seconds)
    root.attrs["conversion_info"] = conversion_info_json(
        r_ft_in_tcp_m=DEFAULT_R_FT_IN_TCP_M if all_v2 else sensor_offset_xyz_m,
        extra={
            "source_type": "unified_t265_wild",
            "pose_source": dominant_pose,
            "pose_sources_per_episode": pose_sources_used,
            "frame_conventions_per_episode": conventions_used,
            "world_frame_override": world_frame,
            "gripper_source": "synthetic_zero",
            "wrench_source": (
                "tcp_from_publisher_v2"
                if all_v2
                else "single_wrist_ft_mapped_to_left_tcp_frame_legacy"
            ),
            "sensor_offset_xyz_m": np.asarray(sensor_offset_xyz_m, dtype=np.float64).tolist(),
            "ultrawide_source": "zeros",
            "depth_source": "d405_depth_mm_to_m_float16_3ch_clipped_0p5",
        },
    )
    root.attrs["ultrawide_source"] = "zeros"

    print(f"[convert] wrote UMIFT-style dataset to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert unified T265 recordings to UMIFT zarr schema")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input unified zarr path or directory containing *_unified.zarr files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output episode-structured zarr path",
    )
    parser.add_argument(
        "--sensor_offset_z_m",
        type=float,
        default=0.0,
        help="Sensor offset along z (meters). Example: -0.03 for 3cm in -z.",
    )
    parser.add_argument(
        "--sensor_offset_xyz_m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Full sensor offset vector in meters. Overrides --sensor_offset_z_m.",
    )
    parser.add_argument(
        "--pose-source",
        choices=("auto", "t265", "tag"),
        default="auto",
        help="Pose for ts_pose_fb: auto prefers tag_traj sidecar when present.",
    )
    parser.add_argument(
        "--resample-10hz",
        action="store_true",
        help="Resample episode streams to a uniform 10 Hz grid before write.",
    )
    parser.add_argument(
        "--start-pad-seconds",
        type=float,
        default=5.0,
        help=(
            "Prepend this many seconds of repeated first-frame data to every "
            "episode, so training queries can be sampled from the true reset "
            "state instead of always requiring real backward history "
            "(see SequenceSamplerUmiFT's obs horizons). Should cover at least "
            "the largest configured obs horizon (currently ~3.2s). Set to 0 "
            "to disable."
        ),
    )
    parser.add_argument(
        "--dagger-pattern",
        type=str,
        default=None,
        help="Episodes whose unified zarr filename contains this substring get attrs['source']='dagger' (others 'original').",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.sensor_offset_xyz_m is not None:
        sensor_offset_xyz_m = np.asarray(args.sensor_offset_xyz_m, dtype=np.float64)
    else:
        sensor_offset_xyz_m = np.array([0.0, 0.0, args.sensor_offset_z_m], dtype=np.float64)

    convert(
        Path(args.input),
        Path(args.output),
        sensor_offset_xyz_m=sensor_offset_xyz_m,
        pose_source=args.pose_source,
        resample_10hz=args.resample_10hz,
        start_pad_seconds=args.start_pad_seconds,
        dagger_pattern=args.dagger_pattern,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
