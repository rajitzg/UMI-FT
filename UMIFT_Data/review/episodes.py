"""Dataset index, episode fingerprints and the per-episode binary payload.

Payload layout: uint32 LE header length | header JSON (UTF-8) | zero pad to a
4-byte boundary | float32 LE arrays concatenated in header["arrays"] order.
Times are raw episode seconds (the zarr timestamps), never re-zeroed.

Orientation is plotted as euler xyz (deg) of R_ref^T R(t), unwrapped. Absolute
euler angles of this tool sit near the xyz gimbal singularity (pitch ~ -78 deg),
so a fixed reference rotation shared by every episode keeps the curves readable
and still comparable across episodes.
"""

from __future__ import annotations

import json
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

POSE_JUMP_M = 0.005
SPEED_SMOOTH_SAMPLES = 25
MAX_POSE_SAMPLES = 20000


@dataclass(frozen=True)
class EpisodeRef:
    dataset: str
    index: int
    zarr_path: str
    source: str  # "original" | "dagger"

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.index}"


def dataset_display_name(zarr_path: str) -> str:
    name = Path(zarr_path).resolve().parent.name
    prefix = "stringer_v3.5_"
    return name[len(prefix):] if name.startswith(prefix) and len(name) > len(prefix) else name


def _episode_indices(data: zarr.Group) -> List[int]:
    return sorted(int(n.split("_", 1)[1]) for n in data.group_keys() if n.startswith("episode_"))


def _quat_wxyz_to_rot(q: np.ndarray) -> Rotation:
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]])


def fingerprint(group: zarr.Group) -> dict:
    t = group["robot_time_stamps_0"]
    return {
        "n_rgb": int(group["rgb_0"].shape[0]),
        "n_robot": int(group["ts_pose_fb_0"].shape[0]),
        "n_wrench": int(group["wrench_left_0"].shape[0]),
        "t_robot_first": round(float(np.ravel(t[0])[0]), 6),
        "t_robot_last": round(float(np.ravel(t[-1])[0]), 6),
    }


class DatasetIndex:
    """All episodes of the base and DAgger zarrs, opened read-only."""

    def __init__(self, base: Sequence[str], dagger: Sequence[str]):
        self._roots: Dict[str, zarr.Group] = {}
        self._groups: Dict[str, zarr.Group] = {}
        self._lock = threading.Lock()
        self.episodes: List[EpisodeRef] = []
        self.dataset_paths: Dict[str, str] = {}
        for paths, source in ((base, "original"), (dagger, "dagger")):
            for p in paths:
                p = str(Path(p).resolve())
                name = dataset_display_name(p)
                if name in self.dataset_paths:
                    raise ValueError(f"duplicate dataset name {name!r} ({p})")
                self.dataset_paths[name] = p
                root = zarr.open(p, mode="r")
                self._roots[name] = root
                for i in _episode_indices(root["data"]):
                    attr = root["data"][f"episode_{i}"].attrs.get("source")
                    if attr is not None and attr != source:
                        print(f"[review] WARNING {name}/{i}: episode attr source={attr!r} "
                              f"disagrees with CLI ({source!r}); using CLI")
                    self.episodes.append(EpisodeRef(name, i, p, source))
        self._by_key = {e.key: e for e in self.episodes}

    def get(self, key: str) -> EpisodeRef:
        return self._by_key[key]

    def root_attrs(self, ref: EpisodeRef) -> dict:
        return dict(self._roots[ref.dataset].attrs)

    def group(self, ref: EpisodeRef) -> zarr.Group:
        with self._lock:
            g = self._groups.get(ref.key)
            if g is None:
                g = self._roots[ref.dataset]["data"][f"episode_{ref.index}"]
                self._groups[ref.key] = g
            return g

    def summary(self, ref: EpisodeRef) -> dict:
        g = self.group(ref)
        t = g["rgb_time_stamps_0"]
        return {
            "key": ref.key,
            "dataset": ref.dataset,
            "index": ref.index,
            "source": ref.source,
            "duration_s": float(np.ravel(t[-1])[0] - np.ravel(t[0])[0]),
            "n_rgb": int(g["rgb_0"].shape[0]),
            "fingerprint": fingerprint(g),
        }

    def reference_rotation(self) -> np.ndarray:
        """Mean orientation of the first episode: one fixed R_ref for all plots."""
        g = self.group(self.episodes[0])
        q = g["ts_pose_fb_0"][::20, 3:7]
        return _quat_wxyz_to_rot(q).mean().as_matrix()


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) < n or n <= 1:
        return x
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, (pad, n - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def build_payload(
    group: zarr.Group,
    extra: Optional[dict] = None,
    root_attrs: Optional[dict] = None,
    r_ref: Optional[np.ndarray] = None,
) -> Tuple[dict, List[np.ndarray]]:
    named: List[Tuple[str, np.ndarray]] = []

    rgb_t = np.ravel(group["rgb_time_stamps_0"][:]).astype(np.float64)
    named.append(("rgb_t", rgb_t))

    pose_t = np.ravel(group["robot_time_stamps_0"][:]).astype(np.float64)
    pose = np.asarray(group["ts_pose_fb_0"][:], dtype=np.float64)
    vt = np.asarray(group["ts_pose_virtual_target_0"][:], dtype=np.float64)
    stiff = np.ravel(group["stiffness_0"][:]).astype(np.float64)

    step = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
    jumps = [float(pose_t[i + 1]) for i in np.flatnonzero(step > POSE_JUMP_M)]
    dt = np.diff(pose_t)
    dt[dt <= 0] = np.nan
    speed = np.concatenate([[0.0], np.nan_to_num(step / dt)])
    speed = _moving_average(speed, SPEED_SMOOTH_SAMPLES)

    if r_ref is None:
        r_ref_rot = _quat_wxyz_to_rot(pose[::20, 3:7]).mean()
    else:
        r_ref_rot = Rotation.from_matrix(np.asarray(r_ref, dtype=np.float64))
    rel = r_ref_rot.inv() * _quat_wxyz_to_rot(pose[:, 3:7])
    eul = np.degrees(np.unwrap(rel.as_euler("xyz"), axis=0))

    n = len(pose_t)
    sel = slice(None)
    if n > MAX_POSE_SAMPLES:
        k = int(np.ceil(n / MAX_POSE_SAMPLES))
        idx = np.arange(0, n, k)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        sel = idx
    named.append(("pose_t", pose_t[sel]))
    for j, nm in enumerate(["px", "py", "pz", "qw", "qx", "qy", "qz"]):
        named.append((nm, pose[sel, j]))
    for j, nm in enumerate(["rx", "ry", "rz"]):
        named.append((nm, eul[sel, j]))
    named.append(("speed", speed[sel]))
    named.append(("stiffness", stiff[sel] if len(stiff) == n else np.resize(stiff, n)[sel]))
    named.append(("vt_off", np.linalg.norm(vt[:, :3] - pose[:, :3], axis=1)[sel]))
    for j, nm in enumerate(["vtx", "vty", "vtz"]):
        named.append((nm, vt[sel, j]))

    wr_t = np.ravel(group["wrench_time_stamps_0"][:]).astype(np.float64)
    wr = np.asarray(group["wrench_left_0"][:], dtype=np.float64)
    named.append(("wrench_t", wr_t))
    for j, nm in enumerate(["fx", "fy", "fz", "tx", "ty", "tz"]):
        named.append((nm, wr[:, j]))
    named.append(("fmag", np.linalg.norm(wr[:, :3], axis=1)))

    gaps = []
    if "scan_gap_intervals_0" in group:
        gaps = np.asarray(group["scan_gap_intervals_0"][:], dtype=float).reshape(-1, 2).tolist()

    fps = float(1.0 / np.median(np.diff(rgb_t))) if len(rgb_t) > 1 else 15.0
    q_ref = r_ref_rot.as_quat()  # xyzw
    header = {
        "arrays": [{"name": nm, "length": int(len(a))} for nm, a in named],
        "duration_s": float(rgb_t[-1] - rgb_t[0]) if len(rgb_t) else 0.0,
        "start_pad_s": float((root_attrs or {}).get("start_pad_seconds", 0.0)),
        "markers": {"scan_gaps": gaps, "pose_jumps": jumps},
        "fps": fps,
        "orientation_ref_quat_wxyz": [float(q_ref[3]), float(q_ref[0]), float(q_ref[1]), float(q_ref[2])],
    }
    if extra:
        header.update(extra)
    return header, [np.asarray(a, dtype=np.float32) for _, a in named]


def encode_payload(header: dict, arrays: List[np.ndarray]) -> bytes:
    hdr = json.dumps(header).encode("utf-8")
    pad = (-(4 + len(hdr))) % 4
    parts = [struct.pack("<I", len(hdr)), hdr, b"\0" * pad]
    parts += [np.ascontiguousarray(a, dtype="<f4").tobytes() for a in arrays]
    return b"".join(parts)


def append_arrays(header: dict, arrays: List[np.ndarray], extra_arrays: Dict[str, np.ndarray]) -> None:
    """Append named arrays to a payload in place (used for the policy window)."""
    for nm, a in extra_arrays.items():
        header["arrays"].append({"name": nm, "length": int(len(a))})
        arrays.append(np.asarray(a, dtype=np.float32))
