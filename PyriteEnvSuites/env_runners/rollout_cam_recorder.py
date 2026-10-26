"""Per-rollout MP4 recorders: wrist (D405 feed) + center Orbbec (V4L)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEFAULT_ORBBEC_V4L = (
    "/dev/v4l/by-id/usb-Orbbec_Orbbec_Femto_Bolt_3D_Camera_CL8K14100WH-video-index0"
)


class _Mp4Writer:
    def __init__(self, path: Path, fps: float, size_wh: tuple[int, int]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._w = cv2.VideoWriter(str(self.path), fourcc, float(fps), size_wh)
        if not self._w.isOpened():
            raise RuntimeError(f"VideoWriter failed to open {self.path}")
        self.n = 0

    def write_bgr(self, frame_bgr: np.ndarray) -> None:
        self._w.write(frame_bgr)
        self.n += 1

    def release(self) -> None:
        self._w.release()


class RolloutCamRecorder:
    """Writes wrist_d405.mp4 + orbbec_center.mp4 under ``out_dir``."""

    def __init__(
        self,
        out_dir: Path,
        *,
        orbbec_v4l: str = DEFAULT_ORBBEC_V4L,
        wrist_fps: float = 15.0,
        orbbec_fps: float = 15.0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.orbbec_v4l = orbbec_v4l
        self.wrist_fps = float(wrist_fps)
        self.orbbec_fps = float(orbbec_fps)
        self._wrist: Optional[_Mp4Writer] = None
        self._orbbec: Optional[_Mp4Writer] = None
        self._orb_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        self._stop.clear()
        self._orb_thread = threading.Thread(
            target=self._orbbec_loop, name="OrbbecRolloutRec", daemon=True
        )
        self._orb_thread.start()

    def push_wrist_rgb(self, rgb: np.ndarray) -> None:
        """Accept HxWx3 RGB uint8 (D405 raw or policy-res)."""
        if rgb is None or rgb.size == 0:
            return
        bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        with self._lock:
            if self._wrist is None:
                self._wrist = _Mp4Writer(
                    self.out_dir / "wrist_d405.mp4", self.wrist_fps, (w, h)
                )
            self._wrist.write_bgr(bgr)

    def stop(self) -> dict:
        self._stop.set()
        if self._orb_thread is not None:
            self._orb_thread.join(timeout=3.0)
            self._orb_thread = None
        with self._lock:
            n_w = self._wrist.n if self._wrist else 0
            if self._wrist is not None:
                self._wrist.release()
                self._wrist = None
            n_o = self._orbbec.n if self._orbbec else 0
            if self._orbbec is not None:
                self._orbbec.release()
                self._orbbec = None
        return {
            "wrist_mp4": str(self.out_dir / "wrist_d405.mp4") if n_w else None,
            "orbbec_mp4": str(self.out_dir / "orbbec_center.mp4") if n_o else None,
            "wrist_frames": n_w,
            "orbbec_frames": n_o,
        }

    def _orbbec_loop(self) -> None:
        cap = cv2.VideoCapture(self.orbbec_v4l)
        if not cap.isOpened():
            print(f"[rollout-cam] Orbbec open failed: {self.orbbec_v4l}")
            return
        period = 1.0 / max(self.orbbec_fps, 1e-3)
        try:
            while not self._stop.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                h, w = frame.shape[:2]
                with self._lock:
                    if self._orbbec is None:
                        self._orbbec = _Mp4Writer(
                            self.out_dir / "orbbec_center.mp4",
                            self.orbbec_fps,
                            (w, h),
                        )
                    self._orbbec.write_bgr(frame)
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            cap.release()
