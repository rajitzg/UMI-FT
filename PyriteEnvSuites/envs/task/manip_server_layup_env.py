"""
ManipServerLayupEnv: UMIFT inference on D405 + UR5e wrist FT via ManipServerEnv.

RGB and depth come from a single pyrealsense2 pipeline (background capture thread).
Manip_server handles robot pose + UR wrist FT only (run_rgb_thread: false).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from PyriteUtility.computer_vision.computer_vision_utility import (
    get_image_transform,
    get_image_transform_with_border,
)
from PyriteEnvSuites.envs.task.manip_server_env import ManipServerEnv, input_res

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - optional at import time for unit tests
    rs = None

# Live "scan_heatmap_0" observation slot, filled synchronously by
# run_layup_scan_periodic() (envs/task/run_layup_scan_periodic.py) via
# set_latest_scan_heatmap() -- called before RTDE control resumes, so the
# policy never reads a stale heatmap after a scan -- and read by
# get_observation_from_buffer() on every horizon. Wired up in
# umift_env_runner.main()'s per-episode loop (periodic ~30s pause/scan/
# resume, separate from the post-episode-only run_layup_scan_eval path).
#
# Internal storage shape: (3, 80, 160) float32 CHW, values in mm (NaN -> 0.0).
# Matches shape_meta.raw/obs.scan_heatmap_0 in umift_zucchini.yaml:
# C=3 (single NN-distance channel replicated 3x, like depth_0),
# H=80, W=160 (compare_pointcloud_distance.py save_small_ml_grid convention).
# Initialized to zeros until the first scan completes.
#
# Emission shape (get_observation_from_buffer): (1, H, W, C) THWC so
# MPC.sample_sparse_obs (horizon=1) and obs_rgb_depth_preprocess's
# scan_heatmap branch (expects THWC, converts to TCHW) see the same layout
# as rgb_0/depth_0. Internal slot stays CHW for set_latest_scan_heatmap.
SCAN_HEATMAP_SHAPE: Tuple[int, int, int] = (3, 80, 160)


def convert_depth_frame_to_umift(
    depth_raw: np.ndarray,
    depth_scale: float,
    depth_clip: float,
    output_hw: Tuple[int, int] = input_res,
) -> np.ndarray:
    """Convert uint16 depth frame to clipped float16 meters, 3 channels, resized."""
    depth_m = np.asarray(depth_raw, dtype=np.float32) * float(depth_scale)
    depth_m = np.clip(depth_m, 0.0, depth_clip)
    if depth_m.shape[:2] != output_hw:
        depth_m = cv2.resize(
            depth_m,
            (output_hw[1], output_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    depth_ch = depth_m.astype(np.float16)
    return np.repeat(depth_ch[..., None], 3, axis=-1)


def convert_color_frame_to_policy(
    color_rgb: np.ndarray,
    raw_to_input_transform: Callable[[np.ndarray], np.ndarray],
    input_to_policy_transform: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Resize RealSense RGB (640x480) to input_res then policy output resolution."""
    color_rgb = np.asarray(color_rgb, dtype=np.uint8)
    if color_rgb.ndim != 3 or color_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB frame, got shape {color_rgb.shape}")
    at_input = raw_to_input_transform(color_rgb)
    return input_to_policy_transform(at_input)


def _frame_timestamp_s(frame) -> float:
    """RealSense frame timestamp in seconds (fallback to wall clock)."""
    try:
        return float(frame.get_timestamp()) / 1000.0
    except Exception:
        return time.time()


def _stack_last_k(
    frames: List[np.ndarray],
    timestamps: np.ndarray,
    k: int,
    empty_shape: Tuple[int, ...],
    empty_dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    if not frames:
        return np.zeros((k, *empty_shape), dtype=empty_dtype), np.zeros((k,), dtype=np.float64)
    while len(frames) < k:
        frames.insert(0, frames[0])
        timestamps = np.concatenate([[timestamps[0]], timestamps])
    return np.stack(frames[-k:], axis=0), timestamps[-k:]


class D405VisionRingBuffer:
    """Background D405 color+depth capture via a single pyrealsense2 pipeline."""

    def __init__(
        self,
        buffer_size: int,
        depth_clip: float,
        raw_to_input_transform: Callable[[np.ndarray], np.ndarray],
        input_to_policy_transform: Callable[[np.ndarray], np.ndarray],
        output_hw: Tuple[int, int] = input_res,
        policy_rgb_hw: Tuple[int, int] = input_res,
        fps: int = 15,
        serial: Optional[str] = None,
        capture_color: bool = True,
        capture_depth: bool = True,
    ) -> None:
        if rs is None:
            raise ImportError(
                "pyrealsense2 is required for ManipServerLayupEnv vision streaming."
            )
        self.buffer_size = max(int(buffer_size), 1)
        self.depth_clip = float(depth_clip)
        self.raw_to_input_transform = raw_to_input_transform
        self.input_to_policy_transform = input_to_policy_transform
        self.output_hw = output_hw
        self.policy_rgb_hw = policy_rgb_hw
        self.fps = fps
        self.serial = serial
        self.capture_color = capture_color
        self.capture_depth = capture_depth
        self._lock = threading.Lock()
        self._rgb_frames: Deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self._rgb_timestamps_s: Deque[float] = deque(maxlen=self.buffer_size)
        self._depth_frames: Deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self._depth_timestamps_s: Deque[float] = deque(maxlen=self.buffer_size)
        self._pipeline: Optional[rs.pipeline] = None
        self._depth_scale = 0.001
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Optional sink for raw 640x480 RGB (rollout MP4); must be cheap/non-blocking.
        self._raw_rgb_sink: Optional[Callable[[np.ndarray], Any]] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        if self.capture_color:
            config.enable_stream(
                rs.stream.color, 640, 480, rs.format.rgb8, self.fps
            )
        if self.capture_depth:
            config.enable_stream(
                rs.stream.depth, 640, 480, rs.format.z16, self.fps
            )
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        device = profile.get_device()
        opened_serial = device.get_info(rs.camera_info.serial_number)
        requested = self.serial or "(first connected device)"
        print(
            f"[D405VisionRingBuffer] Opened RealSense serial {opened_serial} "
            f"(requested: {requested})"
        )
        if self.capture_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="D405VisionCapture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def _capture_loop(self) -> None:
        assert self._pipeline is not None
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=500)
                ts_s = time.time()

                if self.capture_color:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_raw = np.asarray(color_frame.get_data())
                        sink = self._raw_rgb_sink
                        if sink is not None:
                            try:
                                sink(color_raw)
                            except Exception:
                                pass
                        color_policy = convert_color_frame_to_policy(
                            color_raw,
                            self.raw_to_input_transform,
                            self.input_to_policy_transform,
                        )
                        ts_s = _frame_timestamp_s(color_frame)
                        with self._lock:
                            self._rgb_frames.append(color_policy)
                            self._rgb_timestamps_s.append(ts_s)

                if self.capture_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_raw = np.asarray(depth_frame.get_data())
                        depth_umift = convert_depth_frame_to_umift(
                            depth_raw,
                            depth_scale=self._depth_scale,
                            depth_clip=self.depth_clip,
                            output_hw=self.output_hw,
                        )
                        ts_s = _frame_timestamp_s(depth_frame)
                        with self._lock:
                            self._depth_frames.append(depth_umift)
                            self._depth_timestamps_s.append(ts_s)
            except RuntimeError:
                if self._stop.is_set():
                    break
                continue

    def get_last_k_rgb(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = max(int(k), 1)
        with self._lock:
            if not self._rgb_frames:
                empty = np.zeros(
                    (k, self.policy_rgb_hw[0], self.policy_rgb_hw[1], 3),
                    dtype=np.uint8,
                )
                return empty, np.zeros((k,), dtype=np.float64)
            frames = list(self._rgb_frames)[-k:]
            timestamps = np.asarray(list(self._rgb_timestamps_s)[-k:], dtype=np.float64)
        return _stack_last_k(
            frames,
            timestamps,
            k,
            (self.policy_rgb_hw[0], self.policy_rgb_hw[1], 3),
            np.dtype(np.uint8),
        )

    def get_last_k_depth(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = max(int(k), 1)
        with self._lock:
            if not self._depth_frames:
                empty = np.zeros(
                    (k, self.output_hw[0], self.output_hw[1], 3), dtype=np.float16
                )
                return empty, np.zeros((k,), dtype=np.float64)
            frames = list(self._depth_frames)[-k:]
            timestamps = np.asarray(
                list(self._depth_timestamps_s)[-k:], dtype=np.float64
            )
        return _stack_last_k(
            frames,
            timestamps,
            k,
            (self.output_hw[0], self.output_hw[1], 3),
            np.dtype(np.float16),
        )


# Backward-compatible alias for tests
D405DepthRingBuffer = D405VisionRingBuffer


class ManipServerLayupEnv(ManipServerEnv):
    """
    UMIFT inference env for wall-mount layup: UR wrist FT + D405 rgb/depth.

    Robot control via manip_server; vision via pyrealsense2 background thread.
    """

    def __init__(
        self,
        camera_res_hw: List[int],
        hardware_config_path: str,
        query_sizes: Dict[str, Dict],
        compliant_dimensionality: int,
        depth_clip: float = 0.5,
        vision_key: Optional[List[str]] = None,
        realsense_serial: Optional[str] = None,
    ) -> None:
        self._realsense_serial = realsense_serial
        self.vision_key = vision_key or []
        self._use_rgb = "rgb_0" in self.vision_key
        self._use_depth = "depth_0" in self.vision_key
        self._use_pyrealsense_vision = self._use_rgb or self._use_depth

        super().__init__(
            camera_res_hw=camera_res_hw,
            hardware_config_path=hardware_config_path,
            query_sizes=query_sizes,
            compliant_dimensionality=compliant_dimensionality,
        )
        self.depth_clip = float(depth_clip)

        depth_query = query_sizes["sparse"].get("depth")
        rgb_query = query_sizes["sparse"]["rgb"]
        buffer_size = max(
            int(depth_query) if depth_query else 0,
            int(rgb_query),
            1,
        )

        self._vision_buffer: Optional[D405VisionRingBuffer] = None
        if self._use_pyrealsense_vision:
            # Training frames were letterboxed (pad 640x480 to square, then INTER_AREA
            # to 224; UMIFT_Data/umift/processing/umift_image_prep.py). Do the same
            # here -- get_image_transform (resize + center crop) narrows the FoV and
            # enlarges the tool vs. what the policy was trained on.
            raw_to_input = get_image_transform_with_border(
                input_res=(640, 480),
                output_res=input_res,
                bgr_to_rgb=False,
            )
            policy_hw = (camera_res_hw[0], camera_res_hw[1])
            self._vision_buffer = D405VisionRingBuffer(
                buffer_size=buffer_size,
                depth_clip=self.depth_clip,
                raw_to_input_transform=raw_to_input,
                input_to_policy_transform=self.image_transform,
                output_hw=input_res,
                policy_rgb_hw=policy_hw,
                serial=self._realsense_serial,
                capture_color=self._use_rgb,
                capture_depth=self._use_depth,
            )
            self._vision_buffer.start()

        # Live scan_heatmap_0 slot (see SCAN_HEATMAP_SHAPE comment above).
        # Written synchronously by run_layup_scan_periodic() (via
        # set_latest_scan_heatmap, called before it resumes RTDE control);
        # read from the main obs-assembly path (get_observation_from_buffer).
        # A plain Lock is enough since this is just a small array swap, not
        # a hot loop.
        self._scan_heatmap_lock = threading.Lock()
        self._latest_scan_heatmap = np.zeros(SCAN_HEATMAP_SHAPE, dtype=np.float32)

    def set_latest_scan_heatmap(self, heatmap: np.ndarray) -> None:
        """Thread-safe setter for the live scan_heatmap_0 obs slot.

        Called synchronously from run_layup_scan_periodic()
        (envs/task/run_layup_scan_periodic.py), before it resumes RTDE
        control, so the policy never reads a stale heatmap after a scan;
        read from the main obs-assembly thread inside
        get_observation_from_buffer().

        Accepts either a single (H, W) grid (as produced directly by
        compare_pointcloud_distance.py's <stem>_small.npy) or an
        already-channel-replicated (1, H, W) / (C, H, W) array; always
        normalizes to SCAN_HEATMAP_SHAPE before storing. NaNs
        (invalid/unscanned cells) are replaced with 0.0 -- "no measured
        deviation" -- rather than left as NaN, since the live inference
        path (raw_to_obs_inference / obs_rgb_depth_preprocess) does not do
        the NaN-aware index resolution the offline training pipeline does
        for scan_heatmap_0 (see umift_type_conversions.raw_to_obs).
        """
        arr = np.asarray(heatmap, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(
                "set_latest_scan_heatmap expects a (H,W) or (C,H,W) array, "
                f"got shape {arr.shape}"
            )
        if arr.shape[0] == 1 and SCAN_HEATMAP_SHAPE[0] > 1:
            arr = np.repeat(arr, SCAN_HEATMAP_SHAPE[0], axis=0)
        if arr.shape != SCAN_HEATMAP_SHAPE:
            raise ValueError(
                f"set_latest_scan_heatmap shape {arr.shape} does not match "
                f"SCAN_HEATMAP_SHAPE {SCAN_HEATMAP_SHAPE}"
            )
        arr = np.nan_to_num(arr, nan=0.0)
        with self._scan_heatmap_lock:
            self._latest_scan_heatmap = arr

    def get_latest_scan_heatmap(self) -> np.ndarray:
        """Thread-safe getter; returns a copy so callers can't mutate the
        stored array out from under a concurrent set_latest_scan_heatmap."""
        with self._scan_heatmap_lock:
            return self._latest_scan_heatmap.copy()

    def cleanup(self) -> None:
        if self._vision_buffer is not None:
            self._vision_buffer.stop()
        super().cleanup()

    def get_sparse_observation_from_buffer(self):
        """Pose/wrench from manip_server; skip V4L RGB when using pyrealsense vision."""
        if not self._use_pyrealsense_vision:
            return super().get_sparse_observation_from_buffer()

        for id in self.id_list:
            self.sparse_ts_pose_fb[id] = self.server.get_pose(
                self.query_sizes["sparse"]["ts_pose_fb"], id
            ).transpose()
            self.sparse_ts_pose_fb_timestamp_s[id] = (
                self.server.get_pose_timestamps_ms(id) / 1000
            )

            self.sparse_wrench[id] = self.server.get_wrench(
                self.query_sizes["sparse"]["wrench"], id
            ).transpose()
            self.sparse_wrench_timestamp_s[id] = (
                self.server.get_wrench_timestamps_ms(id) / 1000
            )

            self.sparse_wrench_filtered[id] = self.server.get_wrench_filtered(
                self.query_sizes["sparse"]["wrench"], id
            ).transpose()
            self.sparse_wrench_filtered_timestamp_s[id] = (
                self.server.get_wrench_filtered_timestamps_ms(id) / 1000
            )

        for id in self.id_list:
            self.sparse_robot_wrench[id] = self.server.get_robot_wrench(
                500, id
            ).transpose()
            self.sparse_robot_wrench_timestamp_s[id] = (
                self.server.get_robot_wrench_timestamps_ms(id) / 1000
            )

        if self.server.has_eoat():
            for id in self.id_list:
                self.sparse_robot_eoat_pos[id] = self.server.get_eoat(
                    self.query_sizes["sparse"]["eoat"], id
                ).transpose()

        results = {}
        for id in self.id_list:
            results[f"ts_pose_fb_{id}"] = self.sparse_ts_pose_fb[id]
            results[f"robot_wrench_{id}"] = self.sparse_robot_wrench[id]
            results[f"robot_wrench_time_stamps_{id}"] = (
                self.sparse_robot_wrench_timestamp_s[id]
            )
            results[f"robot_time_stamps_{id}"] = self.sparse_ts_pose_fb_timestamp_s[id]
            results[f"wrench_{id}"] = self.sparse_wrench[id]
            results[f"wrench_time_stamps_{id}"] = self.sparse_wrench_timestamp_s[id]

            if self.server.has_eoat():
                results[f"eoat_pos_{id}"] = self.sparse_robot_eoat_pos[id]

        return results

    def schedule_controls(
        self,
        pose7_cmd: np.ndarray,
        eoat_cmd: np.ndarray = None,
        timestamps: np.ndarray = None,
        stiffness_matrices_6x6: np.ndarray = None,
    ) -> bool:
        """UMIFT runner signature (eoat before timestamps in kwargs)."""
        return super().schedule_controls(
            pose7_cmd=pose7_cmd,
            timestamps=timestamps,
            stiffness_matrices_6x6=stiffness_matrices_6x6,
            eoat_cmd=eoat_cmd,
        )

    def start_saving_data_for_a_new_episode(self, episode_name: str = "") -> None:
        self.server.start_listening_key_events()
        self.server.start_saving_data_for_a_new_episode(episode_name)

    def stop_saving_data(self) -> None:
        self.server.stop_saving_data()
        self.server.stop_listening_key_events()

    def set_wrist_raw_rgb_sink(
        self, sink: Optional[Callable[[np.ndarray], Any]]
    ) -> None:
        """Hook D405 raw RGB frames (HxWx3 RGB) for rollout video recording."""
        if self._vision_buffer is not None:
            self._vision_buffer._raw_rgb_sink = sink

    def get_episode_folder(self) -> str:
        return self.server.get_episode_folder()

    def get_observation_from_buffer(self) -> Dict[str, np.ndarray]:
        parent_obs = self.get_sparse_observation_from_buffer()
        results: Dict[str, np.ndarray] = {}

        for robot_id in self.id_list:
            if self._use_rgb and self._vision_buffer is not None:
                rgb_k = self.query_sizes["sparse"]["rgb"]
                rgb_frames, rgb_ts = self._vision_buffer.get_last_k_rgb(rgb_k)
                results[f"rgb_{robot_id}"] = rgb_frames
                results[f"rgb_time_stamps_{robot_id}"] = rgb_ts
            elif f"rgb_{robot_id}" in parent_obs:
                results[f"rgb_{robot_id}"] = parent_obs[f"rgb_{robot_id}"]
                results[f"rgb_time_stamps_{robot_id}"] = parent_obs[
                    f"rgb_time_stamps_{robot_id}"
                ]

            results[f"ts_pose_fb_{robot_id}"] = parent_obs[f"ts_pose_fb_{robot_id}"]
            results[f"robot_time_stamps_{robot_id}"] = parent_obs[
                f"robot_time_stamps_{robot_id}"
            ]

            wrench_key = f"wrench_{robot_id}"
            if wrench_key in parent_obs:
                wrench = np.asarray(parent_obs[wrench_key], dtype=np.float64)
                if wrench.ndim == 1:
                    wrench = wrench.reshape(1, -1)
                n = wrench.shape[0]
                if wrench.shape[1] == 6:
                    results[f"wrench_left_{robot_id}"] = wrench[:, :6]
                    results[f"wrench_right_{robot_id}"] = np.zeros(
                        (n, 6), dtype=np.float64
                    )
                elif wrench.shape[1] >= 12:
                    results[f"wrench_left_{robot_id}"] = wrench[:, :6]
                    results[f"wrench_right_{robot_id}"] = wrench[:, 6:12]
                else:
                    raise ValueError(
                        f"Unexpected wrench width {wrench.shape[1]}; "
                        "expected 6 (UR FT) or 12 (dual CoinFT)."
                    )
                results[f"wrench_time_stamps_{robot_id}"] = parent_obs[
                    f"wrench_time_stamps_{robot_id}"
                ]

            if self._use_depth and self._vision_buffer is not None:
                depth_k = self.query_sizes["sparse"]["depth"]
                depth_frames, depth_ts = self._vision_buffer.get_last_k_depth(depth_k)
                results[f"depth_{robot_id}"] = depth_frames
                results[f"depth_time_stamps_{robot_id}"] = depth_ts

        # scan_heatmap_0: not per-robot-indexed in shape_meta (unlike
        # rgb_{id}/depth_{id}) -- matches how mic_0 is handled elsewhere.
        # Populated synchronously by run_layup_scan_periodic() via
        # set_latest_scan_heatmap(); zeros until the first scan completes.
        # Emit THWC (T=1) for sample_sparse_obs / preprocess.
        chw = self.get_latest_scan_heatmap()  # (C, H, W)
        results["scan_heatmap_0"] = np.moveaxis(chw, 0, -1)[None, ...]  # (1, H, W, C)

        return results
