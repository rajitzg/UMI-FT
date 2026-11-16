"""Cached per-episode MP4 (all-intra H.264), thumbnail and frame sprite.

The MP4 is encoded at a fixed VIDEO_FPS regardless of the recording rate, so
frame i sits at video time (i + 0.5) / VIDEO_FPS exactly; the client maps
frames to episode time with rgb_time_stamps_0, never by assuming a rate.
Every frame is a keyframe (-g 1) so seeking is instant.
"""

from __future__ import annotations

import math
import os
import subprocess
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import zarr
from PIL import Image

VIDEO_FPS = 15
SPRITE_TILE = 56
THUMB = 112
_CHUNK = 208

_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


def _lock(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks[str(path)]


def _frames(group: zarr.Group):
    rgb = group["rgb_0"]
    for s in range(0, rgb.shape[0], _CHUNK):
        yield np.asarray(rgb[s: s + _CHUNK])


def _atomic(path: Path, write) -> Path:
    with _lock(path):
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}_{threading.get_ident()}" + path.suffix)
        try:
            write(tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
    return path


def episode_video(group: zarr.Group, cache_dir: Path, cache_key: str) -> Path:
    out = Path(cache_dir) / f"{cache_key}.mp4"
    if out.exists():
        return out
    h, w = group["rgb_0"].shape[1:3]

    def write(tmp: Path):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(VIDEO_FPS), "-i", "-", "-c:v", "libx264", "-g", "1",
               "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-f", "mp4", str(tmp)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for block in _frames(group):
                proc.stdin.write(np.ascontiguousarray(block, dtype=np.uint8).tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()}")

    return _atomic(out, write)


def episode_thumb(group: zarr.Group, cache_dir: Path, cache_key: str) -> Path:
    out = Path(cache_dir) / f"{cache_key}_thumb.jpg"
    if out.exists():
        return out
    n = group["rgb_0"].shape[0]

    def write(tmp: Path):
        Image.fromarray(np.asarray(group["rgb_0"][n // 2])).resize((THUMB, THUMB)).save(tmp, "JPEG", quality=85)

    return _atomic(out, write)


def sprite_geometry(n: int) -> dict:
    return {"tile": SPRITE_TILE, "cols": int(math.ceil(math.sqrt(n))), "n": int(n)}


def episode_sprite(group: zarr.Group, cache_dir: Path, cache_key: str) -> Tuple[Path, dict]:
    out = Path(cache_dir) / f"{cache_key}_sprite.jpg"
    geo = sprite_geometry(group["rgb_0"].shape[0])
    if out.exists():
        return out, geo
    cols, t = geo["cols"], SPRITE_TILE
    rows = int(math.ceil(geo["n"] / cols))

    def write(tmp: Path):
        sheet = Image.new("RGB", (cols * t, rows * t))
        i = 0
        for block in _frames(group):
            for fr in block:
                sheet.paste(Image.fromarray(fr).resize((t, t), Image.BILINEAR), ((i % cols) * t, (i // cols) * t))
                i += 1
        sheet.save(tmp, "JPEG", quality=80)

    return _atomic(out, write), geo
