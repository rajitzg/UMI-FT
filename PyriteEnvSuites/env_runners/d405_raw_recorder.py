"""Per-rollout raw D405 recorder: IR pair + native depth (+ optional color), lossless.

Writes ``<out_dir>/{infrared_1,infrared_2,depth[,color]}.raw`` plus ``manifest.json`` in the same
layout the wand's live-ingest server uses (``Robotiq-portable-real2sim/python/live_ingest``), so the
offline FFS / defect tools (``worktrees/UMI-FT-ffs_depth/UMIFT_Data/ffs_depth``) can read robot
rollouts like wand episodes. The manifest has extra keys that wand readers ignore:

- ``device``: serial, depth scale, per-stream intrinsics, extrinsics (IR1->IR2 gives the stereo
  baseline), as reported by librealsense.
- ``clock_sync``: (host time.time(), ManipServer hardware time) pairs taken at start and stop, and
  every ~10 s while recording. Use them to map a frame's ``timestamp_sec`` (host clock) onto
  the robot pose timestamps (``ts_pose_fb``, ManipServer clock).
- per-frame ``rs_timestamp_ms`` / ``rs_frame_number``.
- ``info``: dropped frames, server episode folder, etc.

Frames are copied on the capture thread and written on a separate thread, so a slow disk drops
frames (counted in ``info.dropped_frames``) instead of stalling the D405 capture loop that
feeds the policy.

At 640x480 / 15 fps: IR pair + depth is about 18 MB/s (~8.8 GB for an 8 min episode).
Adding color brings it to about 32 MB/s.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

STREAM_FORMATS = {"infrared_1": "y8", "infrared_2": "y8", "depth": "z16", "color": "rgb8"}
MIN_FREE_GB = 10.0
CLOCK_SYNC_PERIOD_S = 10.0


class D405RawRecorder:
    def __init__(
        self,
        out_dir: Path,
        *,
        record_color: bool = False,
        device_info: Optional[dict] = None,
        server_clock_s: Optional[Callable[[], float]] = None,
        extra_info: Optional[dict] = None,
        max_queue: int = 90,
        manifest_every_n: int = 300,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.streams: List[str] = ["infrared_1", "infrared_2", "depth"] + (["color"] if record_color else [])
        self.device_info = device_info or {}
        self.server_clock_s = server_clock_s
        self.extra_info = dict(extra_info or {})
        self.manifest_every_n = int(manifest_every_n)
        self._q: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=int(max_queue))
        self._files: Dict[str, object] = {}
        self._frames: Dict[str, list] = {s: [] for s in self.streams}
        self._offsets: Dict[str, int] = {s: 0 for s in self.streams}
        self._shape: Dict[str, tuple] = {}
        self._clock_sync: List[dict] = []
        self._last_sync = 0.0
        self.dropped = 0
        self.missing: Dict[str, int] = {s: 0 for s in self.streams}
        self.n_frames = 0
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.enabled = False

    # ------------------------------------------------------------------ control (main thread)
    def start(self) -> bool:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(self.out_dir).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"[d405-raw] only {free_gb:.1f} GB free on {self.out_dir}; NOT recording")
            return False
        for s in self.streams:
            self._files[s] = open(self.out_dir / f"{s}.raw", "wb")
        self._sync_clock("start")
        self.enabled = True
        self._thread = threading.Thread(target=self._writer_loop, name="D405RawRec", daemon=True)
        self._thread.start()
        return True

    def maybe_sync_clock(self) -> None:
        """Cheap; call from the control loop. Records a clock pair every CLOCK_SYNC_PERIOD_S."""
        if self.enabled and time.time() - self._last_sync > CLOCK_SYNC_PERIOD_S:
            self._sync_clock("periodic")

    def stop(self) -> dict:
        if not self.enabled:
            return {"dir": str(self.out_dir), "frames": 0, "dropped": 0, "recorded": False}
        self.enabled = False
        self._sync_clock("stop")
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            self._thread = None
        for f in self._files.values():
            f.close()
        self._write_manifest()
        return {"dir": str(self.out_dir), "frames": self.n_frames, "dropped": self.dropped,
                "missing": dict(self.missing), "recorded": True}

    def _sync_clock(self, tag: str) -> None:
        if self.server_clock_s is None:
            return
        try:
            h0 = time.time()
            srv = float(self.server_clock_s())
            h1 = time.time()
        except Exception as exc:  # never break the rollout over bookkeeping
            print(f"[d405-raw] clock sync failed: {exc}")
            return
        with self._lock:
            self._clock_sync.append({"tag": tag, "host_s": 0.5 * (h0 + h1), "server_s": srv, "rtt_s": h1 - h0})
        self._last_sync = h1

    # ------------------------------------------------------------------ capture thread
    def push(self, frame: dict) -> None:
        """Called from the D405 capture thread with a dict built by D405VisionRingBuffer.

        Arrays may alias librealsense buffers, so copy them here before queueing.
        """
        if not self.enabled:
            return
        item = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in frame.items()}
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    # ------------------------------------------------------------------ writer thread
    def _writer_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            for s in self.streams:
                arr = item.get(s)
                if arr is None:
                    self.missing[s] += 1
                    continue
                buf = np.ascontiguousarray(arr).tobytes()
                self._files[s].write(buf)
                if s not in self._shape:
                    self._shape[s] = arr.shape[:2]
                self._frames[s].append({
                    "index": len(self._frames[s]),
                    "length": len(buf),
                    "offset": self._offsets[s],
                    "timestamp_sec": float(item["host_time_s"]),
                    "rs_timestamp_ms": item.get(f"{s}_rs_timestamp_ms"),
                    "rs_frame_number": item.get(f"{s}_rs_frame_number"),
                })
                self._offsets[s] += len(buf)
            self.n_frames += 1
            if self.manifest_every_n > 0 and self.n_frames % self.manifest_every_n == 0:
                for f in self._files.values():
                    f.flush()
                self._write_manifest()

    def _write_manifest(self) -> None:
        streams = {}
        for s in self.streams:
            h, w = self._shape.get(s, (None, None))
            streams[s] = {"format": STREAM_FORMATS[s], "frame_count": len(self._frames[s]), "width": w,
                          "height": h, "frames": list(self._frames[s])}
        with self._lock:
            sync = list(self._clock_sync)
        manifest = {
            "streams": streams,
            "device": self.device_info,
            "clock_sync": sync,
            "info": {**self.extra_info, "source": "umift_env_runner D405RawRecorder",
                     "dropped_frames": self.dropped, "missing_per_stream": dict(self.missing),
                     "timestamp_sec": "host time.time() when the frameset was received",
                     "complete": not self.enabled},
        }
        path = self.out_dir / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest))
        os.replace(tmp, path)
