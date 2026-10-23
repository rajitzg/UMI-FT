import json
import os
import re
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

# from PyriteConfig.tasks.common.common_type_conversions import raw_to_obs
from PyriteConfig.tasks.umift.umift_type_conversions import raw_to_obs_inference
from PyriteConfig.tasks.umift.workcell_frames import (
    load_workcell_extrinsics,
    scan_crop_cfg,
    shape_meta_needs_fixture_frame,
)
from PyriteUtility.common import GracefulKiller
from PyriteUtility.planning_control.mpc import ModelPredictiveController
from PyriteUtility.planning_control.policy_clock import PolicyElapsedClock
from PyriteUtility.planning_control.scan_encoder_monitor import (
    ScanEncoderMonitor,
    render_scan_panel,
    stack_for_display,
)
from PyriteUtility.planning_control.temporal_ensemble import PoseChunkEnsembler
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
from PyriteUtility.pytorch_utils.model_io import load_policy
from PyriteUtility.spatial_math import spatial_utilities as su

from PyriteEnvSuites.envs.task.manip_server_layup_env import ManipServerLayupEnv
# from PyriteEnvSuites.envs.task.manip_server_umift_env import ManipServerUMIFTEnv
from PyriteEnvSuites.envs.task.run_layup_scan_periodic import run_layup_scan_periodic

from PyriteEnvSuites.utils.env_utils import (
    pose9g1_to_traj,
    pose9pose9s1a2_to_traj,
    ts_to_js_traj,
)


def flush_matplotlib_figure(fig, pause_s: float = 0.1) -> None:
    """Force a Matplotlib GUI backend to repaint before blocking on stdin."""
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(pause_s)


def _parse_tagged_line(stdout: str, tag: str) -> Optional[str]:
    """Return value from a ``TAG=value`` line in subprocess stdout."""
    match = re.search(rf"^{re.escape(tag)}=(.+)$", stdout, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _pose7_with_y_offset(pose7: np.ndarray, dy_m: float) -> np.ndarray:
    out = np.array(pose7, dtype=np.float64, copy=True)
    out[1] += dy_m
    return out


def _stitch_pose7_path(
    poses: List[np.ndarray],
    durations_s: List[float],
    n_per_seg: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate pose7 waypoints; drop duplicate joints between segments."""
    chunks_w: List[np.ndarray] = []
    chunks_t: List[np.ndarray] = []
    t_cursor = 0.0
    for i, (dur, n) in enumerate(zip(durations_s, n_per_seg)):
        alpha = np.linspace(0.0, 1.0, n)
        waypoints = su.pose7_interp(poses[i], poses[i + 1], alpha)
        times = np.linspace(t_cursor, t_cursor + dur, n)
        if i > 0:
            waypoints = waypoints[1:]
            times = times[1:]
        chunks_w.append(waypoints)
        chunks_t.append(times)
        t_cursor += dur
    return np.concatenate(chunks_w, axis=0), np.concatenate(chunks_t, axis=0)


def home_to_pose7(
    env,
    id_list: list,
    ts_pose_initial: list,
    duration_s: float = 5.0,
    n_samples: int = 100,
    wait: bool = True,
    lift_m: float = 0.10,
) -> None:
    """Interpolate from current TCP pose(s) back to episode-start pose7.

    Default path first lifts ``lift_m`` in -Y (base frame, same convention as
    scan park-clear), travels at that height, then descends onto start.
    """
    obs_raw = env.get_observation_from_buffer()
    if lift_m > 0.0:
        n_lift = max(2, int(round(n_samples * 0.2)))
        n_descend = n_lift
        n_transit = max(2, n_samples - n_lift - n_descend)
        durations_s = [
            0.2 * duration_s,
            0.6 * duration_s,
            0.2 * duration_s,
        ]
        n_per_seg = [n_lift, n_transit, n_descend]
        waypoints_by_id: Dict[int, np.ndarray] = {}
        timestamps: Optional[np.ndarray] = None
        for rid in id_list:
            current = np.asarray(obs_raw[f"ts_pose_fb_{rid}"][-1], dtype=np.float64)
            goal = np.asarray(ts_pose_initial[rid], dtype=np.float64)
            current_up = _pose7_with_y_offset(current, -lift_m)
            goal_up = _pose7_with_y_offset(goal, -lift_m)
            waypoints, timestamps = _stitch_pose7_path(
                [current, current_up, goal_up, goal],
                durations_s,
                n_per_seg,
            )
            waypoints_by_id[rid] = waypoints
        assert timestamps is not None
        homing_ts_targets = np.zeros([7 * len(id_list), timestamps.shape[0]])
        for rid in id_list:
            homing_ts_targets[rid * 7 : rid * 7 + 7, :] = waypoints_by_id[rid].T
    else:
        sample_indices = np.linspace(0, 1, n_samples)
        timestamps = sample_indices * duration_s
        homing_ts_targets = np.zeros([7 * len(id_list), n_samples])
        for rid in id_list:
            ts_pose_fb = obs_raw[f"ts_pose_fb_{rid}"][-1]
            pose7_waypoints = su.pose7_interp(
                ts_pose_fb, ts_pose_initial[rid], sample_indices
            )
            homing_ts_targets[0 + rid * 7 : 7 + rid * 7, :] = pose7_waypoints.T
    time_now_s = env.current_hardware_time_s
    env.schedule_controls(
        pose7_cmd=homing_ts_targets,
        timestamps=(timestamps + time_now_s) * 1000,
    )
    if wait:
        time.sleep(duration_s + 0.5)


def read_robot_joints_via_rtde(
    keyence_python: Path,
    robot_ip: str,
) -> List[float]:
    """Read actual Q via RTDE receive (coexists with ManipServer control)."""
    code = (
        "from rtde_receive import RTDEReceiveInterface\n"
        f"r = RTDEReceiveInterface(hostname={robot_ip!r})\n"
        "print(','.join(map(str, r.getActualQ())))\n"
        "r.disconnect()\n"
    )
    proc = subprocess.run(
        [str(keyence_python), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to read joints from {robot_ip}: {proc.stderr.strip()}"
        )
    joints = [float(x) for x in proc.stdout.strip().split(",") if x.strip()]
    if len(joints) != 6:
        raise RuntimeError(f"Expected 6 joints, got {joints!r} from {robot_ip}")
    return joints


def run_compare_pointcloud(
    *,
    keyence_python: Path,
    keyence_ws: Path,
    compare_script: Path,
    episode_id: int,
    high_ply: Path,
    low_ply: Path,
    png_path: Path,
    score_json: Path,
    session_dir: Path,
    zones_config: Optional[Path] = None,
) -> Dict[str, Any]:
    """Blocking compare; safe to run in a background thread."""
    print(
        f"[layup-eval] [bg] Episode {episode_id}: comparing "
        f"{high_ply.name} vs {low_ply.name}..."
    )
    cmd = [
        str(keyence_python),
        str(compare_script),
        "--high",
        str(high_ply),
        "--low",
        str(low_ply),
        "--no-show",
        "--png",
        str(png_path),
        "--score-json",
        str(score_json),
    ]
    if zones_config is not None:
        cmd.extend(["--zones-config", str(zones_config)])
    cmp_proc = subprocess.run(
        cmd,
        cwd=str(keyence_ws),
        capture_output=True,
        text=True,
        check=False,
    )
    if cmp_proc.stdout:
        print(cmp_proc.stdout, flush=True)
    if cmp_proc.stderr:
        print(cmp_proc.stderr, flush=True)
    if cmp_proc.returncode != 0:
        raise RuntimeError(
            "compare_pointcloud_distance failed with code "
            f"{cmp_proc.returncode}"
        )
    zones: Optional[Dict[str, Any]] = None
    zones_axis: Optional[str] = None
    if score_json.is_file():
        score = json.loads(score_json.read_text(encoding="utf-8"))
        mean_mm = float(score["mean_mm"])
        heatmap = str(score.get("png", png_path))
        raw_zones = score.get("zones")
        if isinstance(raw_zones, dict):
            zones = raw_zones
            zones_axis = score.get("eval_zones_axis")
    else:
        mean_s = _parse_tagged_line(cmp_proc.stdout, "EVAL_MEAN_MM")
        if mean_s is None:
            raise RuntimeError(
                "Could not parse EVAL_MEAN_MM from compare output"
            )
        mean_mm = float(mean_s)
        heatmap = str(png_path)
    result: Dict[str, Any] = {
        "episode_id": int(episode_id),
        "mean_mm": float(mean_mm),
        "heatmap_png": str(heatmap),
        "session_dir": str(session_dir),
        "high_ply": str(high_ply),
        "low_ply": str(low_ply),
    }
    if zones is not None:
        result["zones"] = zones
        if zones_axis is not None:
            result["eval_zones_axis"] = zones_axis
    print(
        f"[layup-eval] [bg] Episode {episode_id} mean distance: "
        f"{result['mean_mm']:.3f} mm"
    )
    if zones:
        zone_bits = []
        for name, z in zones.items():
            mean_z = z.get("mean_mm")
            count = int(z.get("count", 0))
            if mean_z is None or count <= 0:
                zone_bits.append(f"{name}=n/a")
            else:
                zone_bits.append(f"{name}={float(mean_z):.3f}mm")
        print(
            f"[layup-eval] [bg] Episode {episode_id} zones: "
            + ", ".join(zone_bits)
        )
    print(f"[layup-eval] [bg] Heatmap: {result['heatmap_png']}")
    return result


def run_layup_scan_eval(
    *,
    env,
    episode_id: int,
    control_para: dict,
    right_return_joints: List[float],
    compare_executor: ThreadPoolExecutor,
) -> Future:
    """Pause RTDE, scan + restore right, resume RTDE, compare in background.

    Returns a Future that yields the score dict when compare finishes.
    """
    keyence_ws = Path(control_para["keyence_ws"])
    keyence_python = Path(control_para["keyence_python"])
    low_ply = Path(control_para["layup_eval_low_ply"])
    if not low_ply.is_file():
        raise FileNotFoundError(
            f"layup_eval_low_ply not found: {low_ply}. "
            "Set control_para['layup_eval_low_ply'] to the reference PLY."
        )
    if not keyence_python.is_file():
        raise FileNotFoundError(f"keyence_python not found: {keyence_python}")
    if len(right_return_joints) != 6:
        raise ValueError(
            f"right_return_joints must be length 6, got {len(right_return_joints)}"
        )

    session_name = f"inference_ep{episode_id}_{int(time.time())}"
    scan_script = keyence_ws / "scripts" / "run_layup_scan.py"
    compare_script = keyence_ws / "utils" / "compare_pointcloud_distance.py"
    zones_config = keyence_ws / "config" / "layup_scan.yaml"
    joints_csv = ",".join(f"{q:.9g}" for q in right_return_joints)

    print(f"[layup-eval] Pausing RTDE control for episode {episode_id}...")
    if not env.pause_rtde_control():
        raise RuntimeError("pause_rtde_control failed")

    session_dir: Optional[Path] = None
    high_ply: Optional[Path] = None
    scan_error: Optional[BaseException] = None
    try:
        try:
            print(
                f"[layup-eval] Running layup scan ({session_name}); "
                "right arm returns via RTDE moveJ after scan..."
            )
            # Use --flag=value so joint values starting with '-' are not
            # parsed as new argparse options.
            scan_proc = subprocess.run(
                [
                    str(keyence_python),
                    str(scan_script),
                    "--session-name",
                    session_name,
                    f"--right-return-joints={joints_csv}",
                ],
                cwd=str(keyence_ws),
                capture_output=True,
                text=True,
                check=False,
            )
            print(scan_proc.stdout)
            if scan_proc.stderr:
                print(scan_proc.stderr, flush=True)
            if scan_proc.returncode != 0:
                raise RuntimeError(
                    f"run_layup_scan failed with code {scan_proc.returncode}"
                )
            session_dir_s = _parse_tagged_line(scan_proc.stdout, "LAYUP_SESSION_DIR")
            if not session_dir_s:
                raise RuntimeError(
                    "Could not parse LAYUP_SESSION_DIR from scan output"
                )
            session_dir = Path(session_dir_s)
            high_ply = session_dir / "merged.ply"
            if not high_ply.is_file():
                raise FileNotFoundError(
                    f"Expected merged cloud missing: {high_ply}"
                )
        except BaseException as exc:
            scan_error = exc
    finally:
        print("[layup-eval] Resuming RTDE control...")
        if not env.resume_rtde_control():
            raise RuntimeError("resume_rtde_control failed") from scan_error

    if scan_error is not None:
        raise scan_error
    assert session_dir is not None and high_ply is not None

    png_path = session_dir / "distance_heatmap.png"
    score_json = session_dir / "eval_score.json"
    print(
        f"[layup-eval] Scan done for episode {episode_id}; "
        "starting compare in background so the next episode can begin."
    )
    return compare_executor.submit(
        run_compare_pointcloud,
        keyence_python=keyence_python,
        keyence_ws=keyence_ws,
        compare_script=compare_script,
        episode_id=int(episode_id),
        high_ply=high_ply,
        low_ply=low_ply,
        png_path=png_path,
        score_json=score_json,
        session_dir=session_dir,
        zones_config=zones_config if zones_config.is_file() else None,
    )


def collect_layup_eval_futures(
    futures: List[Future],
    *,
    wait: bool,
) -> List[Dict[str, Any]]:
    """Collect finished (or all, if wait) compare futures into result dicts."""
    results: List[Dict[str, Any]] = []
    still_pending: List[Future] = []
    for fut in futures:
        if not wait and not fut.done():
            still_pending.append(fut)
            continue
        try:
            results.append(fut.result())
        except Exception as exc:
            print(f"[layup-eval] Background compare FAILED: {exc}")
    futures[:] = still_pending
    results.sort(key=lambda r: r["episode_id"])
    return results


def print_layup_eval_summary(results: List[Dict[str, Any]]) -> None:
    print("\n========== Layup scan evaluation summary ==========")
    if not results:
        print("(no evaluations run)")
        print("===================================================\n")
        return
    zone_names: List[str] = []
    for r in results:
        for name in (r.get("zones") or {}):
            if name not in zone_names:
                zone_names.append(name)
    zone_hdr = "".join(f"  {name:>10}" for name in zone_names)
    print(f"{'Episode':>8}  {'mean_mm':>10}{zone_hdr}  heatmap")
    for r in results:
        zone_cols = ""
        zones = r.get("zones") or {}
        for name in zone_names:
            z = zones.get(name) or {}
            mean_z = z.get("mean_mm")
            count = int(z.get("count", 0))
            if mean_z is None or count <= 0:
                zone_cols += f"  {'n/a':>10}"
            else:
                zone_cols += f"  {float(mean_z):>10.3f}"
        print(
            f"{r['episode_id']:>8}  {r['mean_mm']:>10.3f}{zone_cols}  "
            f"{r['heatmap_png']}"
        )
    print("===================================================\n")


def layup_eval_json_path(ckpt_path: Path) -> Path:
    """JSON path adjacent to the checkpoint: ``<stem>_layup_eval.json``."""
    return ckpt_path.with_name(f"{ckpt_path.stem}_layup_eval.json")


def save_layup_eval_results(
    ckpt_path: Path,
    results: List[Dict[str, Any]],
) -> Optional[Path]:
    """Append layup-eval results to JSON next to the checkpoint.

    Re-running on the same ckpt merges into any existing file rather than
    overwriting prior episode scores.
    """
    if not results:
        return None
    out_path = layup_eval_json_path(ckpt_path)
    existing: List[Dict[str, Any]] = []
    if out_path.is_file():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            prior_eps = prior.get("episodes")
            if isinstance(prior_eps, list):
                existing = prior_eps
            else:
                print(
                    f"[layup-eval] WARNING: {out_path} has no episodes list; "
                    "starting fresh merge from new results only."
                )
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[layup-eval] WARNING: could not read {out_path} ({exc}); "
                "overwriting with this run's results."
            )
    merged = existing + list(results)
    payload = {
        "ckpt_path": str(ckpt_path),
        "num_episodes": len(merged),
        "episodes": merged,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[layup-eval] Wrote results: {out_path} "
        f"(appended {len(results)}, total {len(merged)})"
    )
    return out_path


def _stiffness_6x6n_to_per_step(stiffness_6x6n: np.ndarray) -> np.ndarray:
    """Convert (6, 6*N) packed stiffness to (N, 6, 6)."""
    n = stiffness_6x6n.shape[1] // 6
    return stiffness_6x6n.reshape(6, n, 6).transpose(1, 0, 2).copy()


def _stiffness_per_step_to_6x6n(stiffness_per_step: np.ndarray) -> np.ndarray:
    """Convert (N, 6, 6) stiffness back to (6, 6*N)."""
    n = stiffness_per_step.shape[0]
    return stiffness_per_step.transpose(1, 0, 2).reshape(6, 6 * n).copy()


def make_action_chunk_ensemblers(
    id_list: list,
    decay_m: float = 0.0,
) -> dict:
    """One PoseChunkEnsembler per robot for live MPC temporal ensembling."""
    return {
        rid: PoseChunkEnsembler(
            pose_keys=("pose7", "nominal_pose7"),
            linear_keys=("stiffness", "eoat"),
            weight_mode="exp",
            decay_m=decay_m,
        )
        for rid in id_list
    }


if "PYRITE_CHECKPOINT_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CHECKPOINT_FOLDERS")
if "PYRITE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
    raise ValueError(
        "Please set the environment variable PYRITE_HARDWARE_CONFIG_FOLDERS"
    )

if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CONTROL_LOG_FOLDERS")

checkpoint_folder_path = os.environ.get("PYRITE_CHECKPOINT_FOLDERS")
hardware_config_folder_path = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS")
control_log_folder_path = os.environ.get("PYRITE_CONTROL_LOG_FOLDERS")


def resolve_realsense_serial(hardware_config_path: str) -> Optional[str]:
    """Resolve D405 serial for pyrealsense2 (env var overrides yaml)."""
    serial = os.environ.get("UMIFT_REALSENSE_SERIAL")
    if serial:
        return serial
    with open(hardware_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("realsense_serial")


def log_raw_vision_to_obs_inference(obs_inference, obs_raw: dict, robot_id: int) -> None:
    """Persist all buffered RGB/depth frames and timestamps for one horizon."""
    rgb_key = f"rgb_{robot_id}"
    ultrawide_key = f"ultrawide_{robot_id}"
    depth_key = f"depth_{robot_id}"

    image_key = rgb_key if rgb_key in obs_raw else ultrawide_key if ultrawide_key in obs_raw else None
    if image_key is not None:
        frames = np.asarray(obs_raw[image_key])
        if frames.ndim >= 3 and frames.shape[0] > 0:
            obs_inference.create_dataset(image_key, data=frames)
            ts_key = (
                f"rgb_time_stamps_{robot_id}"
                if image_key == rgb_key
                else f"ultrawide_time_stamps_{robot_id}"
            )
            if ts_key in obs_raw:
                obs_inference.create_dataset(
                    ts_key,
                    data=np.asarray(obs_raw[ts_key], dtype=np.float64).reshape(-1),
                )

    if depth_key in obs_raw:
        depth_frames = np.asarray(obs_raw[depth_key])
        if depth_frames.ndim >= 3 and depth_frames.shape[0] > 0:
            obs_inference.create_dataset(depth_key, data=depth_frames)
            depth_ts_key = f"depth_time_stamps_{robot_id}"
            if depth_ts_key in obs_raw:
                obs_inference.create_dataset(
                    depth_ts_key,
                    data=np.asarray(obs_raw[depth_ts_key], dtype=np.float64).reshape(-1),
                )


def log_horizon(
    log,
    horizon_count: int,
    id_list: list,
    exe_horizon: int,
    outputs_ts_targets,
    outputs_ts_nominal_targets,
    outputs_ts_stiffnesses,
    timestamps,
    action_start_time_s: float,
    obs_raw: dict,
    env,
    controller: ModelPredictiveController,
    obs_task: dict,
    shape_meta: dict,
    log_policy_vision: bool,
    scan_encoder_stats: Optional[dict] = None,
) -> None:
    """Write one MPC horizon to the episode control log (executed slice only)."""
    horizon_log = log.create_group(f"horizon_{horizon_count}")
    exe = min(exe_horizon, outputs_ts_targets[0].shape[0])

    meta = horizon_log.create_group("meta")
    meta.attrs["hardware_time_s"] = float(env.current_hardware_time_s)
    meta.attrs["horizon_index"] = int(horizon_count)
    meta.attrs["exe_horizon"] = int(exe)

    action_log = horizon_log.create_group("action")
    for robot_id in id_list:
        action_log.create_dataset(
            f"ts_virtual_targets_{robot_id}",
            data=outputs_ts_targets[robot_id][:exe],
        )
        action_log.create_dataset(
            f"ts_nominal_targets_{robot_id}",
            data=outputs_ts_nominal_targets[robot_id][:exe],
        )
        action_log.create_dataset(
            f"ts_stiffnesses_{robot_id}",
            data=outputs_ts_stiffnesses[robot_id][:, : 6 * exe],
        )
    action_log.create_dataset(
        "timestamps_s",
        data=timestamps[:exe] + action_start_time_s,
    )

    obs_policy_group = horizon_log.create_group("obs_policy")
    obs_sampled = controller.sample_sparse_obs(obs_task["obs"])
    obs_policy, _ = controller.preprocess_policy_obs(
        obs_sampled, ignore_rgb=not log_policy_vision
    )
    vision_keys = {"rgb_0", "ultrawide_0", "depth_0"}
    for key in shape_meta["sample"]["obs"]["sparse"]:
        if key not in obs_policy:
            continue
        if not log_policy_vision and key in vision_keys:
            continue
        obs_policy_group.create_dataset(key, data=obs_policy[key])

    if scan_encoder_stats is not None:
        scan_enc_group = horizon_log.create_group("scan_encoder")
        scan_enc_group.create_dataset(
            "feature", data=np.asarray(scan_encoder_stats["feature"], dtype=np.float32)
        )
        scan_enc_group.attrs["status"] = str(scan_encoder_stats["status"])
        for k, v in ScanEncoderMonitor.log_scalars(scan_encoder_stats).items():
            scan_enc_group.attrs[k] = v

    obs_inference = horizon_log.create_group("obs_inference")
    obs_inference.attrs["hardware_time_s"] = float(env.current_hardware_time_s)
    for robot_id in id_list:
        obs_inference.create_dataset(
            f"ts_pose_fb_{robot_id}",
            data=obs_raw[f"ts_pose_fb_{robot_id}"][-1],
        )
        if log_policy_vision:
            log_raw_vision_to_obs_inference(obs_inference, obs_raw, robot_id)


def main():
    control_para = {
        # Training policy rate = 10 Hz; iPhone/GoPro legacy path uses 0.0167 (~60 Hz).
        "raw_time_step_s": 1.0 / 15.0,
        "slow_down_factor": 2,  # slow down policy execution by this factor
        # Must be <= the loaded checkpoint's sparse_action_horizon (epoch=0170 is 12;
        # new 10 Hz yaml uses 32). With ensemble on, smaller => more overlap.
        "sparse_execution_horizon": 16, # first step is at t=0, so this should be >1
        # Zero-added-latency temporal ensembling of overlapping action chunks.
        # Lower sparse_execution_horizon with this on increases overlap/smoothing.
        "enable_ensemble_chunking": True,
        "ensemble_decay_m": 0.01,  # 0 = uniform; >0 trusts near-term steps more (exp)
        # Per-episode wall-clock cutoff (hardware time). Episode ends and
        # returns to the post-episode menu when exceeded. Default: 4 minutes.
        "max_duration_s": 8 * 60,
        "gripper_action_scale": 1000,
        "use_audio": False,
        "gripper_lookahead_steps": 0,  # how many steps to look ahead for gripper action
        "set_wrench_fb_to_zero": False,
        "set_depth_to_zero": True,  # ablation: feed all-zero depth to policy
        "test_nominal_target": False,  # False to follow virtual target
        "test_nominal_target_stiffness": -1,  # -1, for a 'no force case'. 1500, for a quick test without force.
        "fix_orientation": False,
        "enable_control_log": True,
        "log_policy_vision": True,
        "pausing_mode": False,  # matplotlib debug only; independent of logging
        "device": "cuda",
        # Post-episode Keyence layup scan eval (menu option "e")
        "keyence_ws": "/home/boeing/keyence_scan_ws",
        "keyence_python": "/home/boeing/keyence_scan_ws/.venv/bin/python",
        # Reference / "low" surface PLY for compare_pointcloud_distance --low
        "layup_eval_low_ply": "/mnt/data/boeing/scans/scan_20261011_161311/merged.ply",
        # Right-arm IP for capturing episode-start joints (RTDE receive).
        "layup_eval_right_ip": "192.168.1.2",
        # Periodic mid-episode Keyence layup scan interrupts
        # (run_layup_scan_periodic). Distinct from the post-episode-only "e"
        # menu option (run_layup_scan_eval) above: this pauses/resumes the
        # LIVE control loop itself, roughly every periodic_scan_interval_s
        # seconds during an episode. Each call cold-starts run_layup_scan.py
        # (no persistent daemon) and resumes control only after the new
        # scan_heatmap_0 has been loaded, since the policy conditions on it.
        "periodic_scan_enabled": True,
        "periodic_scan_interval_s": 60.0,
        "periodic_scan_timeout_s": 90.0,
        "periodic_scan_compare_timeout_s": 30.0,
        # Show the heatmap / EE-centered crop the policy actually received next to the
        # rgb/depth popup (updated once per policy query).
        "show_scan_panel": True,
        # Hook the scan CNN and print/log per-query statistics to spot a collapsed
        # encoder (see PyriteUtility/planning_control/scan_encoder_monitor.py).
        "monitor_scan_encoder": True,
    }
    pipeline_para = {
        # Layup: D405 rgb (manip_server) + depth (pyrealsense2) + UR5e wrist FT.
        "use_layup_env": True,
        "iphone_params": {
            "socket_ip": "192.168.2.18",
            "ports": [5555],
        },
        "ckpt_path": "/stringer_scanner_v2/flatten_crop/epoch=0030-train_loss=0.020.ckpt",
        "hardware_config_path": hardware_config_folder_path
        + "/ur5e_wrist_ft.yaml",
        "audio_device_id": [5],  # only needed for audio input
        # D405 rgb/depth ~15 Hz live vs 10 Hz training → 15/10=1.5; //= floor → 1.
        "depth_downsample_correction": 1,
        # Live wrist FT via UR RTDE at 500 Hz (ur5e_wrist_ft.yaml: rtde_frequency=500,
        # run_wrench_thread=false); scale ds to match 10 Hz training wall-clock.
        "robot_downsample_correction": 500.0 / 200.0,
        "wrench_downsample_correction": 500.0 / 28.0,
    }
    if not pipeline_para["use_layup_env"]:
        control_para["raw_time_step_s"] = 0.0167
        pipeline_para["hardware_config_path"] = (
            hardware_config_folder_path + "/right_arm_coinft.yaml"
        )
        pipeline_para["depth_downsample_correction"] = 2
        pipeline_para["robot_downsample_correction"] = 1
        pipeline_para["wrench_downsample_correction"] = 1
    verbose = 1
    episode_id = 0
    layup_eval_results: List[Dict[str, Any]] = []
    layup_eval_futures: List[Future] = []
    compare_executor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="layup-compare"
    )

    def get_real_obs_resolution(shape_meta: dict) -> Tuple[int, int]:
        out_res = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            shape = attr.get("shape")
            # scan_heatmap_* is typed rgb for the data pipeline but has a
            # different (H, W) than rgb_0/depth_0; exclude it here the same
            # way timm_obs_encoder_with_force._is_scan_heatmap_key does.
            if type == "rgb" and "scan_heatmap" not in key:
                co, ho, wo = shape
                if out_res is None:
                    out_res = (wo, ho)
                assert out_res == (wo, ho)
        return out_res

    def printOrNot(verbose, *args):
        if verbose >= 0:
            print(f"[Episode {episode_id}] ", *args)

    vbs_h1 = verbose + 1  # verbosity for header
    vbs_h2 = verbose  # verbosity for sub-header
    # vbs_p = verbose - 1  # verbosity for paragraph

    # load policy
    ckpt_full_path = Path(
        checkpoint_folder_path + pipeline_para["ckpt_path"]
    )
    print("Loading policy: ", ckpt_full_path)
    device = torch.device(control_para["device"])
    policy, shape_meta = load_policy(str(ckpt_full_path), device)

    # image size
    (image_width, image_height) = get_real_obs_resolution(shape_meta)

    vision_key = []
    has_rgb = "rgb_0" in shape_meta["sample"]["obs"]["sparse"]
    has_ultrawide_rgb = "ultrawide_0" in shape_meta["sample"]["obs"]["sparse"]
    has_depth = "depth_0" in shape_meta["sample"]["obs"]["sparse"]

    if has_rgb:
        vision_key.append("rgb_0")
    if has_ultrawide_rgb:
        vision_key.append("ultrawide_0")
    if has_depth:
        vision_key.append("depth_0")

    # Validate vision modality constraints
    if has_rgb == 0 and has_ultrawide_rgb == 0:
        raise ValueError(
            "Either 'rgb_0' or 'ultrawide_0' must be present in shape_meta."
        )
    if len(vision_key) > 2:
        raise ValueError("More than two vision modalities present in shape_meta.")

    # Drop unused camera streams from shape_meta.raw (e.g. ultrawide when only
    # rgb_0 is active). Do NOT drop scan_heatmap_*: it is typed rgb for the data
    # pipeline but is not a camera stream, and MPC still samples it from
    # shape_meta.sample.obs.sparse via raw_to_obs_inference.
    from omegaconf import OmegaConf

    OmegaConf.set_struct(shape_meta["raw"], False)
    valid_camera_keys = set(vision_key)
    raw_vision_keys = [
        k for k, v in shape_meta["raw"].items() if v.get("type") == "rgb"
    ]
    for key in raw_vision_keys:
        if "scan_heatmap" in key:
            continue
        if key not in valid_camera_keys:
            print(f"Removing unused camera key from raw: {key}")
            del shape_meta["raw"][key]
    OmegaConf.set_struct(shape_meta["raw"], True)

    print(f"Vision Keys for id:0 in this shape meta: {vision_key}")

    # Both rgb and ultrawide stream at 60Hz, so we'll just use either one. rgb, if it exists.
    rgb_key = next(k for k in vision_key if k in ["rgb_0", "ultrawide_0"])

    # ultrawide also streams at 60Hz so no mapping is needed during inference. Just use the same downsampling and horizon.
    rgb_query_size = (
        shape_meta["sample"]["obs"]["sparse"][rgb_key]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"][rgb_key]["down_sample_steps"] + 1

    # Depth: live D405 ~15 Hz; shape_meta ds/h are for 10 Hz training. Floor-correction via //=.
    # Correction for depth downsampling
    if "depth_0" in shape_meta["sample"]["obs"]["sparse"]:
        shape_meta["sample"]["obs"]["sparse"]["depth_0"]["down_sample_steps"] //= (
            pipeline_para["depth_downsample_correction"]
        )

        depth_query_size = (
            shape_meta["sample"]["obs"]["sparse"]["depth_0"]["horizon"] - 1
        ) * shape_meta["sample"]["obs"]["sparse"]["depth_0"]["down_sample_steps"] + 1

        depth_clip = shape_meta["obs"]["depth_0"]["clip"]
    else:
        depth_query_size = None

    # Layup: scale down_sample_steps so MPC striding on ~500 Hz RTDE (pose + wrist FT)
    # buffers matches 10 Hz training wall-clock (ds=1 on 10 Hz zarr).
    if pipeline_para["use_layup_env"]:
        OmegaConf.set_struct(shape_meta["sample"]["obs"]["sparse"], False)
        robot_corr = pipeline_para["robot_downsample_correction"]
        for robot_key in (
            "robot0_eef_pos",
            "robot0_eef_rot_axis_angle",
            "robot0_abs_eef_pos",
            "robot0_abs_eef_rot_axis_angle",
        ):
            if robot_key in shape_meta["sample"]["obs"]["sparse"]:
                ds = int(
                    shape_meta["sample"]["obs"]["sparse"][robot_key][
                        "down_sample_steps"
                    ]
                )
                new_ds = max(1, round(ds * robot_corr))
                shape_meta["sample"]["obs"]["sparse"][robot_key][
                    "down_sample_steps"
                ] = new_ds
                print(
                    f"[layup] {robot_key} down_sample_steps: {ds} -> {new_ds} "
                    f"(x{robot_corr:.2f})"
                )
        wrench_corr = pipeline_para["wrench_downsample_correction"]
        for wrench_key in ("robot0_eef_wrench_left", "robot0_eef_wrench_right"):
            if wrench_key in shape_meta["sample"]["obs"]["sparse"]:
                ds = int(
                    shape_meta["sample"]["obs"]["sparse"][wrench_key][
                        "down_sample_steps"
                    ]
                )
                new_ds = max(1, round(ds * wrench_corr))
                shape_meta["sample"]["obs"]["sparse"][wrench_key][
                    "down_sample_steps"
                ] = new_ds
                print(
                    f"[layup] {wrench_key} down_sample_steps: {ds} -> {new_ds} "
                    f"(x{wrench_corr:.2f})"
                )

    ts_pose_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["down_sample_steps"] + 1

    if "robot0_eef_wrench_left" in shape_meta["sample"]["obs"]["sparse"]:
        wrench_query_size = (
            shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench_left"]["horizon"]
            - 1
        ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench_left"][
            "down_sample_steps"
        ] + 1
    else:
        wrench_query_size = 0

    if "robot0_gripper" in shape_meta["sample"]["obs"]["sparse"]:
        eoat_query_size = (
            shape_meta["sample"]["obs"]["sparse"]["robot0_gripper"]["horizon"] - 1
        ) * shape_meta["sample"]["obs"]["sparse"]["robot0_gripper"][
            "down_sample_steps"
        ] + 1
    else:
        eoat_query_size = 0

    query_sizes_sparse = {
        "rgb": rgb_query_size,
        "depth": depth_query_size,
        "ts_pose_fb": ts_pose_query_size,
        "wrench": wrench_query_size,
        "eoat": eoat_query_size,
    }

    if "mic_0" in shape_meta["sample"]["obs"]["sparse"]:
        mic_query_size = (
            shape_meta["sample"]["obs"]["sparse"]["mic_0"]["horizon"] - 1
        ) * shape_meta["sample"]["obs"]["sparse"]["mic_0"]["down_sample_steps"] + 1
        query_sizes_sparse["mic"] = mic_query_size

    query_sizes = {
        "sparse": query_sizes_sparse,
    }

    if (
        control_para["test_nominal_target"]
        and control_para["test_nominal_target_stiffness"] < 0
    ):
        n_af = 0
    else:
        n_af = 3

    audio_params = None
    if control_para["use_audio"]:
        audio_params = {
            "audio_device_id": pipeline_para["audio_device_id"],
        }

    if pipeline_para["use_layup_env"]:
        realsense_serial = resolve_realsense_serial(
            pipeline_para["hardware_config_path"]
        )
        print(
            f"[layup] pyrealsense2 D405 serial: "
            f"{realsense_serial or '(first connected device)'}"
        )
        env = ManipServerLayupEnv(
            camera_res_hw=(image_height, image_width),
            hardware_config_path=pipeline_para["hardware_config_path"],
            query_sizes=query_sizes,
            compliant_dimensionality=n_af,
            depth_clip=depth_clip if depth_query_size is not None else 0.5,
            vision_key=vision_key,
            realsense_serial=realsense_serial,
        )
    else:
        env = ManipServerUMIFTEnv(
            camera_res_hw=(image_height, image_width),
            hardware_config_path=pipeline_para["hardware_config_path"],
            query_sizes=query_sizes,
            audio_params=audio_params,
            compliant_dimensionality=n_af,
            iphone_params=pipeline_para["iphone_params"],
            vision_key=vision_key,
        )

    env.reset()

    # Periodic mid-episode scan interrupts (see
    # control_para["periodic_scan_*"] above). run_layup_scan_periodic() is
    # called directly from the main loop below -- no persistent
    # daemon/controller object to construct or tear down per episode.
    # A checkpoint trained with task=umift_ss_v2_no_scan has no scan_heatmap
    # input; do not pause the robot for scans that policy will never see.
    policy_uses_scan = any(
        "scan_heatmap" in key for key in shape_meta["sample"]["obs"]["sparse"]
    )
    periodic_scan_enabled = (
        bool(control_para.get("periodic_scan_enabled", False)) and policy_uses_scan
    )
    # Policies with an elapsed_time_0 input were trained on demo-time seconds since the
    # policy start. The clock rescales wall time by slow_down_factor and adds scan
    # pauses back unscaled (see PolicyElapsedClock).
    has_elapsed_time = "elapsed_time_0" in shape_meta["sample"]["obs"]["sparse"]
    elapsed_clock = PolicyElapsedClock(control_para["slow_down_factor"])
    if has_elapsed_time:
        assert (
            shape_meta["sample"]["obs"]["sparse"]["elapsed_time_0"]["horizon"] == 1
        ), "elapsed_time_0 must have horizon 1"
        print(
            "[elapsed-time] policy takes elapsed_time_0; demo-time scale = 1/"
            f"{control_para['slow_down_factor']}"
        )
    if control_para.get("periodic_scan_enabled", False) and not policy_uses_scan:
        print(
            "[layup] periodic scans off: this checkpoint has no scan_heatmap input"
        )
    next_scan_id = 0

    p_timestep_s = control_para["raw_time_step_s"]

    sparse_action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
        "down_sample_steps"
    ]
    sparse_action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
    # sparse_action_horizon_s = (
    #     sparse_action_horizon * sparse_action_down_sample_steps * p_timestep_s
    # )
    sparse_execution_horizon = (
        sparse_action_down_sample_steps * control_para["sparse_execution_horizon"]
    )
    sparse_action_timesteps_s = (
        np.arange(0, sparse_action_horizon)
        * sparse_action_down_sample_steps
        * p_timestep_s
        * control_para["slow_down_factor"]
    )

    action_type = None
    id_list = [0]
    if shape_meta["action"]["shape"][0] == 9:
        action_type = "pose9"
        id_list = [0]
        action_to_trajectory = ts_to_js_traj
    elif shape_meta["action"]["shape"][0] == 10:
        action_type = "pose9g1"
        id_list = [0]
        action_to_trajectory = pose9g1_to_traj
    elif shape_meta["action"]["shape"][0] == 21:
        action_type = "pose9pose9s1a2"
        id_list = [0]
        action_to_trajectory = pose9pose9s1a2_to_traj
    elif shape_meta["action"]["shape"][0] == 42:
        action_type = "pose9pose9s1a2"
        id_list = [0, 1]
        action_to_trajectory = pose9pose9s1a2_to_traj
    else:
        raise RuntimeError("unsupported")

    printOrNot(vbs_h2, "Creating MPC.")
    controller = ModelPredictiveController(
        shape_meta=shape_meta,
        id_list=id_list,
        policy=policy,
        action_to_trajectory=action_to_trajectory,
        execution_horizon=sparse_execution_horizon,
        fix_orientation=control_para["fix_orientation"],
    )
    controller.set_time_offset(env.current_hardware_time_s)

    # Policies trained with absolute poses and/or the EE-centered scan crop work in the
    # fixture frame (origin at tag0). Live RTDE poses are in the right-arm base frame,
    # so hand the controller the base->fixture transform from the workcell extrinsics
    # (override the file with $UMIFT_WORKCELL_EXTRINSICS after moving the part).
    if shape_meta_needs_fixture_frame(shape_meta):
        extrinsics = load_workcell_extrinsics()
        controller.SE3_fixture_from_world = extrinsics.T_fixture_from_right_base
        print(
            "[frames] policy uses fixture-frame poses; extrinsics:",
            extrinsics.path,
            "\nT_fixture_from_right_base =\n",
            controller.SE3_fixture_from_world,
        )

    # Diagnostics for the scan CNN branch (hooks only read tensors).
    scan_monitor: Optional[ScanEncoderMonitor] = None
    if control_para["monitor_scan_encoder"]:
        scan_monitor = ScanEncoderMonitor(policy)
        if not scan_monitor.enabled:
            scan_monitor = None
    scan_is_cropped = scan_crop_cfg(shape_meta) is not None
    show_scan_panel = bool(control_para["show_scan_panel"]) and (
        "scan_heatmap_0" in shape_meta["sample"]["obs"]["sparse"]
    )

    # timestep_idx = 0
    # stiffness = None
    episode_initial_time_s = env.current_hardware_time_s
    execution_duration_s = (
        (control_para["sparse_execution_horizon"] - 1)
        * sparse_action_down_sample_steps
        * p_timestep_s
        * control_para["slow_down_factor"]
    )
    printOrNot(vbs_h2, "Starting main loop.")

    printOrNot(vbs_h2, "Calibrating robot wrench.")
    env.calibrate_robot_wrench(NSamples=100)
    time.sleep(2)
    obs = env.get_observation_from_buffer()
    w = np.mean(obs["wrench_left_0"][-100:], axis=0)  # or wrench_left_0 in layup
    print("wrench: ", w)

    if control_para["pausing_mode"]:
        plt.ion()  # to run GUI event loop
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # x = np.linspace(-0.02, 0.2, 20)
        # y = np.linspace(-0.1, 0.1, 20)
        # z = np.linspace(-0.1, 0.1, 20)
        ax.set_title("Target and virtual target")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        flush_matplotlib_figure(fig)

    horizon_count = 0
    log = None

    # plot the rgb/depth stream, confirm it is correct
    time.sleep(2)

    while True:
        obs_raw = env.get_observation_from_buffer()

        images_to_stack = []
        for key in vision_key:
            im = obs_raw[key][-1]

            if key == "depth_0":
                # Normalize float16 depth to uint8 for display
                depth_raw = im.squeeze()  # (H, W) if needed
                depth_norm = depth_raw / depth_clip  # Normalize to [0, 1]
                im = (depth_norm * 255).astype(np.uint8)
                # im = np.repeat(depth_uint8[:, :, np.newaxis], 3, axis=2)
            else:
                # Convert RGB to BGR for OpenCV
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

            images_to_stack.append(im)

        cv2.imshow("image", stack_for_display(images_to_stack))

        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    ts_pose_initial = []
    for id in id_list:
        ts_pose_initial.append(obs_raw[f"ts_pose_fb_{id}"][-1])
    # Joints at the same moment — used by layup scan to restore right arm via
    # RTDE moveJ (avoids ManipServer zone/incremental limits around right_park).
    right_start_joints: Optional[List[float]] = None
    try:
        right_start_joints = read_robot_joints_via_rtde(
            Path(control_para["keyence_python"]),
            str(control_para["layup_eval_right_ip"]),
        )
        print(
            "[layup-eval] Captured right start joints: "
            + ",".join(f"{q:.4f}" for q in right_start_joints)
        )
    except Exception as exc:
        print(
            f"[layup-eval] WARNING: could not capture right start joints ({exc}). "
            "Menu option 'e' will be unavailable until restart."
        )
    #########################################
    # main loop starts
    #########################################
    while True:
        printOrNot(vbs_h1, "New episode. Episode ID: ", episode_id)
        input("Press Enter to start the episode.")

        # env.start_saving_data_for_a_new_episode(pipeline_para["episode_data_save_path"])
        env.start_saving_data_for_a_new_episode()

        log = None
        horizon_count = 0
        if scan_monitor is not None:
            scan_monitor.reset()
        if control_para["enable_control_log"]:
            episode_log_path = os.path.join(
                control_log_folder_path, f"episode_{int(time.time())}"
            )
            log_store = zarr.DirectoryStore(path=episode_log_path)
            log = zarr.open(store=log_store, mode="w")
            log.attrs["sparse_execution_horizon"] = int(
                control_para["sparse_execution_horizon"]
            )
            log.attrs["enable_ensemble_chunking"] = bool(
                control_para.get("enable_ensemble_chunking", False)
            )
            log.attrs["ensemble_decay_m"] = float(
                control_para.get("ensemble_decay_m", 0.0)
            )
            log.attrs["episode_id"] = int(episode_id)
            log.attrs["episode_data_folder"] = env.get_episode_folder()
            log.attrs["ckpt_path"] = pipeline_para["ckpt_path"]
            log.attrs["hardware_config_path"] = pipeline_para["hardware_config_path"]
            log.attrs["log_policy_vision"] = bool(control_para["log_policy_vision"])
            printOrNot(vbs_h1, "Control log:", episode_log_path)

        # Fresh temporal ensemblers for each episode (zero-added-latency blending).
        action_ensemblers = None
        if control_para.get("enable_ensemble_chunking", False):
            action_ensemblers = make_action_chunk_ensemblers(
                id_list=id_list,
                decay_m=float(control_para.get("ensemble_decay_m", 0.0)),
            )
            printOrNot(
                vbs_h1,
                "Ensemble action chunking ENABLED "
                f"(decay_m={control_para.get('ensemble_decay_m', 0.0)})",
            )

        episode_initial_time_s = env.current_hardware_time_s
        elapsed_clock.start(episode_initial_time_s)
        last_scan_time_s = -float("inf")
        killer = GracefulKiller()
        while not killer.kill_now:
            horizon_initial_time_s = env.current_hardware_time_s
            printOrNot(vbs_h1, "Starting new horizon at ", horizon_initial_time_s)

            # Periodic mid-episode laser scan interrupt: pauses RTDE control,
            # runs a one-shot run_layup_scan.py, loads the new heatmap, then
            # resumes RTDE control -- all inside run_layup_scan_periodic(),
            # blocking this single-threaded loop. Nothing below this point
            # runs until it returns, which is what keeps the D405/pose/wrench
            # buffers' *observed* history uncorrupted (same reasoning the old
            # ScanPauseController relied on); the loop simply resumes calling
            # env.get_observation_from_buffer() fresh afterward.
            if (
                periodic_scan_enabled
                and horizon_initial_time_s - last_scan_time_s
                >= control_para["periodic_scan_interval_s"]
            ):
                scan_id = next_scan_id
                next_scan_id += 1
                printOrNot(
                    vbs_h1, f"Pausing for periodic laser scan (scan_id={scan_id})..."
                )
                outcome = run_layup_scan_periodic(
                    env=env, scan_id=scan_id, control_para=control_para
                )
                if outcome.success:
                    printOrNot(
                        vbs_h1,
                        f"Scan {scan_id} complete: {outcome.session_dir} "
                        f"(duration={outcome.duration_s:.1f}s)",
                    )
                else:
                    printOrNot(vbs_h1, f"Scan {scan_id} FAILED: {outcome.error}")
                # Real wall-clock time passed during the scan; refresh both
                # the horizon clock (used below for execution-duration sleep
                # math) and the periodic-scan clock (measured from when this
                # scan actually finished, not when it was due).
                scan_end_time_s = env.current_hardware_time_s
                # The demo clock kept running during scan pauses in the training data.
                elapsed_clock.add_pause(scan_end_time_s - horizon_initial_time_s)
                horizon_initial_time_s = scan_end_time_s
                last_scan_time_s = horizon_initial_time_s

            obs_raw = env.get_observation_from_buffer()

            # # plot the rgb image
            # if len(id_list) == 1:
            #     rgb = obs_raw["rgb_0"][-1]
            # else:
            #     rgb = np.vstack([obs_raw[f"rgb_{i}"][-1] for i in id_list])
            # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # cv2.imshow("image", bgr)
            # cv2.waitKey(10)

            # visualize all vision modalities
            images_to_stack = []
            for key in vision_key:
                im = obs_raw[key][-1]

                if key == "depth_0":
                    # Normalize float16 depth to uint8 for display
                    depth_raw = im.squeeze()  # (H, W) if needed
                    depth_norm = depth_raw / depth_clip  # Normalize to [0, 1]
                    im = (depth_norm * 255).astype(np.uint8)
                else:
                    # Convert RGB to BGR for OpenCV
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                images_to_stack.append(im)

            # The popup is drawn after inference (below) so the scan panel shows exactly
            # what the policy received for this query. Without the panel, draw it now.
            if not show_scan_panel:
                cv2.imshow("image", stack_for_display(images_to_stack))
                cv2.waitKey(10)

            obs_task = dict()

            # hack for UMIFT
            for id in id_list:
                if f"rgb_{id}" not in obs_raw:
                    # sampling done based on rgb_time_stamps during training. During inference, rgb time and ultrawide time are the same as both sample at 60Hz
                    obs_raw[f"rgb_time_stamps_{id}"] = obs_raw[
                        f"ultrawide_time_stamps_{id}"
                    ]
            # import pdb; pdb.set_trace()

            raw_to_obs_inference(obs_raw, obs_task, shape_meta)

            if has_elapsed_time:
                elapsed_s = elapsed_clock.elapsed_s(horizon_initial_time_s)
                obs_task["obs"]["elapsed_time_0"] = np.array(
                    [[elapsed_s]], dtype=np.float32
                )
                printOrNot(vbs_h2, f"elapsed_time_0 = {elapsed_s:.2f}s (demo time)")

            # hacks for UMIFT
            for id in id_list:
                # Since training data and policy use meter, but real gripper hardware uses mm
                if f"robot{id}_gripper" in obs_task["obs"]:
                    obs_task["obs"][f"robot{id}_gripper"] = (
                        obs_task["obs"][f"robot{id}_gripper"]
                        / control_para["gripper_action_scale"]
                    )
                # Training real data seems to have a mismatch in rgb channels
                # obs_task["obs"][f"rgb_{id}"] = obs_task["obs"][f"rgb_{id}"][:,[2,1,0],:,:]

                if control_para["set_wrench_fb_to_zero"]:
                    if f"robot{id}_eef_wrench_left" in obs_task["obs"]:
                        obs_task["obs"][f"robot{id}_eef_wrench_left"] = np.zeros_like(
                            obs_task["obs"][f"robot{id}_eef_wrench_left"]
                        )
                        obs_task["obs"][f"robot{id}_eef_wrench_right"] = np.zeros_like(
                            obs_task["obs"][f"robot{id}_eef_wrench_right"]
                        )

            if control_para["set_depth_to_zero"] and "depth_0" in obs_task["obs"]:
                obs_task["obs"]["depth_0"] = np.zeros_like(obs_task["obs"]["depth_0"])

            assert (
                action_type == "pose9pose9s1a2"
                or action_type == "pose9"
                or action_type == "pose9g1"
            ), f"Unsupported action type: {action_type}"

            # Run inference
            if action_type == "pose9":
                action_sparse_target_mats = controller.compute_one_horizon_action(
                    obs_task["obs"], device
                )
                action_sparse_vt_mats = action_sparse_target_mats.copy()
                action_stiffnesses = None
                action_eoat = None
            elif action_type == "pose9g1":
                (
                    action_sparse_target_mats,
                    action_eoat_pos,
                ) = controller.compute_one_horizon_action(obs_task["obs"], device)
                action_sparse_vt_mats = action_sparse_target_mats.copy()
                action_stiffnesses = None
                action_eoat = [np.array] * len(id_list)
                for id in id_list:
                    action_eoat[id] = np.zeros([sparse_action_horizon, 2])
                    action_eoat[id][:, 0] = np.squeeze(action_eoat_pos[id])
            elif action_type == "pose9pose9s1a2":
                (
                    action_sparse_target_mats,
                    action_sparse_vt_mats,
                    action_stiffnesses,
                    action_eoat,
                ) = controller.compute_one_horizon_action(obs_task["obs"], device)

            # Scan-encoder diagnostics + popup (never allowed to interrupt control).
            scan_encoder_stats = None
            try:
                if scan_monitor is not None:
                    scan_encoder_stats = scan_monitor.update()
                    if scan_encoder_stats is not None:
                        print(scan_monitor.format_line(scan_encoder_stats))
                if show_scan_panel:
                    vision_img = stack_for_display(images_to_stack)
                    scan_obs = (controller.last_obs_policy or {}).get("scan_heatmap_0")
                    if scan_obs is not None:
                        panel = render_scan_panel(
                            scan_obs,
                            height=vision_img.shape[0],
                            crop=scan_is_cropped,
                            text_lines=(
                                scan_monitor.overlay_lines(scan_encoder_stats)
                                if scan_encoder_stats is not None
                                else None
                            ),
                        )
                        vision_img = np.hstack([vision_img, panel])
                    cv2.imshow("image", vision_img)
                    cv2.waitKey(10)
            except Exception as exc:
                print(f"[scan-enc] display/monitor error (ignored): {exc}")

            outputs_ts_nominal_targets = [np.array] * len(id_list)
            outputs_ts_targets = [np.array] * len(id_list)
            outputs_ts_stiffnesses = [np.array] * len(id_list)

            if action_type == "pose9pose9s1a2" or action_type == "pose9g1":
                for id in id_list:
                    # hack for UMIFT, since training data and policy use meter, but real gripper hardware uses mm
                    print("Gripper action before scaling: ", action_eoat[id][:, 0])
                    print("Gripper force: ", action_eoat[id][:, 1])
                    action_eoat[id][:, 0] *= control_para["gripper_action_scale"]
                    if control_para["gripper_lookahead_steps"] > 0:
                        action_eoat[id][
                            : -control_para["gripper_lookahead_steps"], :
                        ] = action_eoat[id][
                            control_para["gripper_lookahead_steps"] :, :
                        ]

            if action_stiffnesses is not None:
                print("action_stiffness: ", action_stiffnesses[0])
            # print("action_target_z: ", action_sparse_target_mats[0][:, 2, 3])
            # print("action_virtual_target_z: ", action_sparse_vt_mats[0][:, 2, 3])
            print(
                "action_delta z: ",
                action_sparse_vt_mats[0][:, 2, 3]
                - action_sparse_target_mats[0][:, 2, 3],
            )
            outputs_eoat = action_eoat
            # decode stiffness matrix
            for id in id_list:
                SE3_TW = su.SE3_inv(su.pose7_to_SE3(obs_raw[f"ts_pose_fb_{id}"][-1]))
                ts_targets_nominal = su.SE3_to_pose7(
                    action_sparse_target_mats[id].reshape([-1, 4, 4])
                )
                ts_targets_virtual = su.SE3_to_pose7(
                    action_sparse_vt_mats[id].reshape([-1, 4, 4])
                )

                ts_stiffnesses = np.zeros([6, 6 * ts_targets_virtual.shape[0]])
                for i in range(ts_targets_virtual.shape[0]):
                    SE3_target = action_sparse_target_mats[id][i].reshape([4, 4])
                    SE3_virtual_target = action_sparse_vt_mats[id][i].reshape([4, 4])

                    # stiffness: 1. convert vt to tool frame
                    SE3_TVt = SE3_TW @ SE3_virtual_target
                    SE3_Ttarget = SE3_TW @ SE3_target

                    # stiffness: 2. compute stiffness matrix in the tool frame
                    compliance_direction_tool = (
                        SE3_TVt[:3, 3] - SE3_Ttarget[:3, 3]
                    ).reshape(3)

                    if np.linalg.norm(compliance_direction_tool) < 0.001:  #
                        compliance_direction_tool = np.array([1.0, 0.0, 0.0])

                    compliance_direction_tool /= np.linalg.norm(
                        compliance_direction_tool
                    )
                    X = compliance_direction_tool
                    Y = np.cross(X, np.array([0, 0, 1]))
                    Y /= np.linalg.norm(Y)
                    Z = np.cross(X, Y)

                    default_stiffness = 5000
                    default_stiffness_rot = 100
                    stiffness_matrix_full = np.eye(6) * default_stiffness_rot

                    if action_stiffnesses is None:
                        target_stiffness = default_stiffness
                    else:
                        target_stiffness = action_stiffnesses[id][i]

                    M = np.diag(
                        [target_stiffness, default_stiffness, default_stiffness]
                    )
                    S = np.array([X, Y, Z]).T
                    stiffness_matrix = S @ M @ np.linalg.inv(S)
                    stiffness_matrix_full[:3, :3] = stiffness_matrix

                    if (
                        control_para["test_nominal_target"]
                        and control_para["test_nominal_target_stiffness"] > 0
                    ):
                        stiffness_matrix_full[:3, :3] = (
                            np.eye(3) * control_para["test_nominal_target_stiffness"]
                        )

                    ts_stiffnesses[:, 6 * i : 6 * i + 6] = stiffness_matrix_full

                outputs_ts_nominal_targets[id] = ts_targets_nominal
                if control_para["test_nominal_target"]:
                    outputs_ts_targets[id] = ts_targets_nominal
                else:
                    outputs_ts_targets[id] = ts_targets_virtual
                outputs_ts_stiffnesses[id] = ts_stiffnesses

            # the "now" when the observation is taken
            action_start_time_s = obs_raw["robot_time_stamps_0"][-1]
            timestamps = sparse_action_timesteps_s

            if control_para["pausing_mode"]:
                # plot the actions for this horizon using matplotlib
                ax.cla()
                for i, id in enumerate(id_list):
                    # Label only the first robot to avoid duplicate legend entries.
                    label = lambda name: name if i == 0 else None
                    ax.plot3D(
                        action_sparse_target_mats[id][..., 0, 3],
                        action_sparse_target_mats[id][..., 1, 3],
                        action_sparse_target_mats[id][..., 2, 3],
                        color="red",
                        marker="o",
                        markersize=3,
                        label=label("Nominal target"),
                    )

                    ax.plot3D(
                        action_sparse_vt_mats[id][..., 0, 3],
                        action_sparse_vt_mats[id][..., 1, 3],
                        action_sparse_vt_mats[id][..., 2, 3],
                        color="blue",
                        marker="o",
                        markersize=3,
                        label=label("Virtual target"),
                    )

                    ax.plot3D(
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 0, 3
                        ],
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 1, 3
                        ],
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 2, 3
                        ],
                        color="yellow",
                        linewidth=3,
                        marker="o",
                        markersize=4,
                        label=label("Executed nominal"),
                    )

                    ax.plot3D(
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 0, 3
                        ],
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 1, 3
                        ],
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 2, 3
                        ],
                        color="green",
                        linewidth=3,
                        marker="o",
                        markersize=4,
                        label=label("Executed virtual"),
                    )

                    ax.plot3D(
                        obs_raw[f"ts_pose_fb_{id}"][-1][0],
                        obs_raw[f"ts_pose_fb_{id}"][-1][1],
                        obs_raw[f"ts_pose_fb_{id}"][-1][2],
                        color="black",
                        marker="o",
                        markersize=8,
                        label=label("Current TCP"),
                    )

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.legend(loc="best")

                set_axes_equal(ax)

                # Process the GUI event loop before input() blocks the main thread.
                flush_matplotlib_figure(fig)

                input("Press Enter to start executing the plotted actions.")

            # Blend overlapping chunk predictions (or hard-cut) down to execution horizon.
            exec_h = int(control_para["sparse_execution_horizon"])
            if action_ensemblers is not None:
                start_step = horizon_count * exec_h
                for id in id_list:
                    chunk_len = int(np.asarray(outputs_ts_targets[id]).shape[0])
                    if exec_h > chunk_len:
                        raise ValueError(
                            f"sparse_execution_horizon={exec_h} exceeds policy "
                            f"action chunk length={chunk_len} (robot {id}). "
                            f"Lower sparse_execution_horizon to <= {chunk_len} "
                            f"(loaded ckpt horizon), or retrain with a longer "
                            f"sparse_action_horizon."
                        )
                    fields = {
                        "pose7": outputs_ts_targets[id],
                        "nominal_pose7": outputs_ts_nominal_targets[id],
                        "stiffness": _stiffness_6x6n_to_per_step(
                            outputs_ts_stiffnesses[id]
                        ),
                        "eoat": (
                            outputs_eoat[id] if outputs_eoat is not None else None
                        ),
                    }
                    action_ensemblers[id].push(start_step=start_step, fields=fields)
                    blended = action_ensemblers[id].pop(exec_h)
                    outputs_ts_targets[id] = blended["pose7"]
                    outputs_ts_nominal_targets[id] = blended["nominal_pose7"]
                    outputs_ts_stiffnesses[id] = _stiffness_per_step_to_6x6n(
                        blended["stiffness"]
                    )
                    if outputs_eoat is not None and "eoat" in blended:
                        outputs_eoat[id] = blended["eoat"]
            else:
                for id in id_list:
                    outputs_ts_targets[id] = outputs_ts_targets[id][:exec_h, :]
                    outputs_ts_nominal_targets[id] = outputs_ts_nominal_targets[id][
                        :exec_h, :
                    ]
                    outputs_ts_stiffnesses[id] = outputs_ts_stiffnesses[id][
                        :, : 6 * exec_h
                    ]
                    if outputs_eoat is not None:
                        outputs_eoat[id] = outputs_eoat[id][:exec_h, :]
            timestamps = timestamps[:exec_h]

            if control_para["enable_control_log"] and log is not None:
                log_horizon(
                    log=log,
                    horizon_count=horizon_count,
                    id_list=id_list,
                    exe_horizon=exec_h,
                    outputs_ts_targets=outputs_ts_targets,
                    outputs_ts_nominal_targets=outputs_ts_nominal_targets,
                    outputs_ts_stiffnesses=outputs_ts_stiffnesses,
                    timestamps=timestamps,
                    action_start_time_s=action_start_time_s,
                    obs_raw=obs_raw,
                    env=env,
                    controller=controller,
                    obs_task=obs_task,
                    shape_meta=shape_meta,
                    log_policy_vision=control_para["log_policy_vision"],
                    scan_encoder_stats=scan_encoder_stats,
                )

            action_start_time_s = env.current_hardware_time_s

            if len(id_list) == 1:
                outputs_ts_targets = outputs_ts_targets[0].T  # N x 7 to 7 x N
                outputs_ts_stiffnesses = outputs_ts_stiffnesses[0]
                if outputs_eoat is not None:
                    outputs_eoat = outputs_eoat[0].T  # N x 2 to 2 x N
            else:
                outputs_ts_targets = np.hstack(
                    outputs_ts_targets
                ).T  # 2 x N x 7 to 14 x N
                if outputs_eoat is not None:
                    outputs_eoat = np.hstack(outputs_eoat).T  # 2 x N x 2 to 4 x N
                outputs_ts_stiffnesses = np.vstack(
                    outputs_ts_stiffnesses
                )  # 6 x (6xN) to 12 x (6xN)

            env.schedule_controls(
                pose7_cmd=outputs_ts_targets,
                eoat_cmd=outputs_eoat,
                stiffness_matrices_6x6=outputs_ts_stiffnesses,
                timestamps=(timestamps + action_start_time_s) * 1000,
            )

            # # log the truncated action
            # horizon_log = log.create_group(f"horizon_{horizon_count}")
            # horizon_log.create_dataset("ts_targets", data=outputs_ts_targets)
            # horizon_log.create_dataset("timestamps", data=timestamps * 1000)
            # horizon_log.create_dataset("ts_stiffnesses", data=outputs_ts_stiffnesses)
            # horizon_log.create_dataset(
            #     "timestamps_s", data=timestamps + action_start_time_s
            # )
            horizon_count += 1

            if control_para["pausing_mode"]:
                c = input("Press Enter to start the next horizon. q to quit.")
                if c == "q":
                    break

            time_s = env.current_hardware_time_s
            sleep_duration_s = horizon_initial_time_s + execution_duration_s - time_s

            printOrNot(vbs_h1, "sleep_duration_s", sleep_duration_s)
            time.sleep(max(0, sleep_duration_s))

            if not control_para["pausing_mode"]:
                # only check duration when not in pausing mode
                elapsed_s = time_s - episode_initial_time_s
                if elapsed_s > control_para["max_duration_s"]:
                    printOrNot(
                        vbs_h1,
                        f"Episode cutoff: {elapsed_s:.1f}s > "
                        f"max_duration_s={control_para['max_duration_s']}",
                    )
                    break

        if scan_monitor is not None:
            print(scan_monitor.summary())
        printOrNot(vbs_h1, "End of episode.")
        finished_episode_id = episode_id
        episode_id += 1

        env.stop_saving_data()

        print("Options:")
        print("     c: continue to next episode.")
        print("     r: reset to default pose, then continue.")
        print(
            "     e: layup scan eval (scan+restore now; compare runs in background)."
        )
        print(
            "     s: layup scan eval then quit "
            "(wait for compare; save results next to ckpt)."
        )
        print("     b: reset to default pose, then quit the program.")
        print("     others: quit the program.")
        # Surface any compares that finished during the previous episode.
        layup_eval_results.extend(
            collect_layup_eval_futures(layup_eval_futures, wait=False)
        )
        c = input("Please select an option: ")
        if c in ("e", "s"):
            if right_start_joints is None:
                print(
                    "[layup-eval] Cannot run: right start joints were not "
                    "captured at startup."
                )
            else:
                try:
                    fut = run_layup_scan_eval(
                        env=env,
                        episode_id=finished_episode_id,
                        control_para=control_para,
                        right_return_joints=right_start_joints,
                        compare_executor=compare_executor,
                    )
                    layup_eval_futures.append(fut)
                except Exception as exc:
                    print(f"[layup-eval] FAILED: {exc}")
                    print(
                        "RTDE may still be paused; attempting resume before continuing."
                    )
                    try:
                        env.resume_rtde_control()
                    except Exception as resume_exc:
                        print(f"[layup-eval] resume also failed: {resume_exc}")
            if c == "s":
                print(
                    "[layup-eval] Scan queued; waiting for compare, then quitting."
                )
                break
            input("Press Enter to continue.")
        elif c == "r" or c == "b":
            print("Resetting to default pose.")
            home_to_pose7(
                env, id_list, ts_pose_initial, duration_s=5.0, wait=False
            )
        elif c == "c":
            pass
        else:
            print("Quitting the program.")
            break

        if c == "r":
            input("Press Enter to continue.")

        if c == "b":
            input("Press Enter to quit program.")
            break

        print("Continuing to execution.")
    # env.save_whole_episode()
    # env.save_video()
    if layup_eval_futures:
        print(
            f"[layup-eval] Waiting for {len(layup_eval_futures)} background "
            "compare job(s)..."
        )
    layup_eval_results.extend(
        collect_layup_eval_futures(layup_eval_futures, wait=True)
    )
    print_layup_eval_summary(layup_eval_results)
    save_layup_eval_results(ckpt_full_path, layup_eval_results)
    compare_executor.shutdown(wait=True)
    env.cleanup()
    # end of episode
    # # plot


if __name__ == "__main__":
    main()
