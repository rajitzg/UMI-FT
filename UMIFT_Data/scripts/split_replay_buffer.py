#!/usr/bin/env python3
"""Split an acp_replay_buffer.zarr by episode range into a new zarr (episodes renumbered from 0).

Copies episode dirs, rebases the cumulative `meta/episode_*_len` arrays, slices the per-episode lists in
`conversion_info`, and writes `source_episodes.json` next to the output zarr for traceability.

  python split_replay_buffer.py SRC.zarr DST.zarr START END   # episodes [START, END)
"""
import json, shutil, sys
from pathlib import Path

import numpy as np
import zarr

src, dst, a, b = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
assert not dst.exists(), f"{dst} exists"
zs = zarr.open(str(src), "r")
n_src = len(zs["meta/episode_rgb0_len"])
assert 0 <= a < b <= n_src, (a, b, n_src)

dst.mkdir(parents=True)
for f in (".zgroup", ".zattrs"):
    if (src / f).exists():
        shutil.copy(src / f, dst / f)
(dst / "data").mkdir()
shutil.copy(src / "data/.zgroup", dst / "data/.zgroup")
for i in range(a, b):
    shutil.copytree(src / f"data/episode_{i}", dst / f"data/episode_{i - a}")

# meta: cumulative end indices -> rebase to the slice
shutil.copytree(src / "meta", dst / "meta")
zd = zarr.open(str(dst), "r+")
for k in zs["meta"]:
    c = zs["meta"][k][:]
    base = c[a - 1] if a > 0 else 0
    zd["meta"][k][: b - a] = c[a:b] - base
    zd["meta"][k].resize(b - a)

# conversion_info: slice per-episode lists
attrs = dict(zs.attrs)
ci = json.loads(attrs["conversion_info"])
for k, v in ci.items():
    if k.endswith("_per_episode") and isinstance(v, list) and len(v) == n_src:
        ci[k] = v[a:b]
attrs["conversion_info"] = json.dumps(ci)
attrs["source_split"] = {"src": str(src), "episodes": [a, b]}
zd.attrs.update(attrs)

json.dump({str(i - a): f"{src.name}:episode_{i}" for i in range(a, b)}, open(dst.parent / "source_episodes.json", "w"), indent=1)

# self-check: lengths match and last episode's data is bit-identical to the source
for k in zs["meta"]:
    assert zd["meta"][k][-1] == zs["meta"][k][b - 1] - (zs["meta"][k][a - 1] if a else 0)
assert np.array_equal(zd[f"data/episode_{b - a - 1}/ts_pose_fb_0"][:], zs[f"data/episode_{b - 1}/ts_pose_fb_0"][:])
print(f"OK: {dst} ({b - a} episodes from {src.name}[{a}:{b}])")
