"""Stringer (mold) mesh for the 3D overlay.

Reads the Fix B field (``fixb_field.npz``: ``vertices``/``faces`` in the fixture
frame, ``is_apron`` marks the synthetic flange apron), drops apron faces, and
decimates with open3d to ``max_tris`` (same approach as surface_frames'
``view_oct.py``). The result is cached as ``mold_<hash>.npz`` in the cache dir.

The fixture frame is the AprilTag tag0 frame, which ``ts_pose_fb_0`` also uses, so
the mesh lines up with demos as long as the mold has not moved relative to the
fixture tags since the scan it was built from (see the field's ``mold_id``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import numpy as np

DEFAULT_FIELD = "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/shared/fixb_field.npz"
MAX_TRIS = 20000


def load_mold(field_path, cache_dir, max_tris: int = MAX_TRIS) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(field_path).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{p}:{p.stat().st_mtime}:{max_tris}".encode()).hexdigest()[:16]
    cache = cache_dir / f"mold_{key}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["vertices"], d["faces"]

    d = np.load(p, allow_pickle=False)
    v = np.asarray(d["vertices"], dtype=np.float64)
    f = np.asarray(d["faces"], dtype=np.int64)
    if "is_apron" in d.files:
        f = f[~np.asarray(d["is_apron"], dtype=bool)[f].any(axis=1)]
    used = np.unique(f)
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    v, f = v[used], remap[f]
    if len(f) > max_tris:
        import open3d as o3d
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f.astype(np.int32)))
        m = m.simplify_quadric_decimation(target_number_of_triangles=max_tris)
        m.remove_unreferenced_vertices()
        v, f = np.asarray(m.vertices), np.asarray(m.triangles)
    v = v.astype(np.float32)
    f = f.astype(np.uint32)
    np.savez(cache, vertices=v, faces=f)
    return v, f
