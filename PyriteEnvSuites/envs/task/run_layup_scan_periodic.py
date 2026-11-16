"""run_layup_scan_periodic(): periodic mid-episode laser-scan interrupts for
a live diffusion-policy inference loop (bimanual UR5e layup cell), built on
a one-shot run_layup_scan.py call instead of a persistent scan_daemon.py.

Deliberately kept free of any dependency on umift_env_runner.py (which it's
called from) or on real hardware bindings (ManipServer, RTDE), the same way
the module this superseded (scan_pause_controller.py) was -- so this file
can be imported and unit-tested without the full inference stack. Small
helpers (`_parse_tagged_line`, `read_right_arm_joints_via_rtde`) are
deliberately duplicated from umift_env_runner.py rather than imported, for
the same reason: importing umift_env_runner.py pulls in ManipServerLayupEnv,
which requires the compiled `hardware_interfaces` bindings only present on
the real robot control PC.

Never-paused-forever guarantee
-------------------------------
``run_layup_scan_periodic()`` calls ``env.pause_rtde_control()`` exactly
once. From the moment that call returns truthy, every subsequent step (the
scan subprocess, the heatmap compare) runs inside a ``try/finally`` whose
``finally`` clause calls ``env.resume_rtde_control()`` unconditionally --
including when the scan fails, or the compare step fails or times out. A
scan going wrong can make the returned ``ScanOutcome`` report failure, but
it can never leave the arm paused indefinitely.

Resume-after-heatmap ordering
-------------------------------
Unlike a fire-and-forget background compare, ``env.set_latest_scan_heatmap()``
is called (when the compare succeeds) BEFORE ``env.resume_rtde_control()``.
The policy is conditioned on ``scan_heatmap_0``, so control must not resume
against a stale heatmap while a fresh one is about to be available.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


def _parse_tagged_line(stdout: str, tag: str) -> Optional[str]:
    """Return value from a ``TAG=value`` line in subprocess stdout.

    Duplicated from umift_env_runner.py -- see module docstring.
    """
    match = re.search(rf"^{re.escape(tag)}=(.+)$", stdout, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def read_right_arm_joints_via_rtde(keyence_python: Path, robot_ip: str) -> List[float]:
    """Read the right (task) arm's actual joint angles via a throwaway RTDE
    receive connection.

    Duplicated from umift_env_runner.py's ``read_robot_joints_via_rtde`` --
    see module docstring for why.
    """
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
        timeout=10.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to read joints from {robot_ip}: {proc.stderr.strip()}"
        )
    joints = [float(x) for x in proc.stdout.strip().split(",") if x.strip()]
    if len(joints) != 6:
        raise RuntimeError(f"Expected 6 joints, got {joints!r} from {robot_ip}")
    return joints


@dataclass
class ScanOutcome:
    """Result of one run_layup_scan_periodic() call.

    ``success`` reflects the pause/scan/compare/resume sequence as a whole.
    """

    scan_id: int
    success: bool
    session_dir: Optional[Path] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None


def load_compare_heatmap(
    *,
    scan_id: int,
    png_path: Path,
    heatmap_channels: int,
) -> Optional[np.ndarray]:
    """Load the small ML-ready heatmap grid already written by the compare
    that run_layup_scan.py ran internally (via ``--compare-low``, joined
    before the scan subprocess returns -- see its module docstring, steps
    7.5/9.5).

    Returns it replicated to ``heatmap_channels`` channels as ``(C, H, W)``
    float32 -- ready for ``ManipServerLayupEnv.set_latest_scan_heatmap()``.
    Returns None (rather than raising) if the grid is missing -- e.g. the
    compare failed or timed out inside run_layup_scan.py, which logs but
    doesn't fail the scan itself -- so the caller just skips updating the
    heatmap slot.
    """
    small_path = png_path.with_name(png_path.stem + "_small.npy")
    if not small_path.is_file():
        print(
            f"[layup-periodic] scan {scan_id}: no heatmap grid at "
            f"{small_path} (compare inside run_layup_scan.py may have "
            "failed/timed out)"
        )
        return None
    try:
        grid_mm = np.load(small_path).astype(np.float32)  # (H, W); NaN where invalid
        grid_mm = np.nan_to_num(grid_mm, nan=0.0)
        heatmap = np.repeat(grid_mm[None, ...], heatmap_channels, axis=0)  # (C, H, W)
        print(f"[layup-periodic] scan {scan_id}: heatmap ready {heatmap.shape}")
        return heatmap.astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - never let a bad heatmap load propagate
        print(f"[layup-periodic] scan {scan_id}: heatmap load FAILED: {exc}")
        return None


def run_layup_scan_periodic(
    *,
    env,
    scan_id: int,
    control_para: dict,
) -> ScanOutcome:
    """Pause RTDE, run a one-shot layup scan, load+apply the new heatmap,
    THEN resume RTDE control.

    Blocking. This is the periodic, mid-episode counterpart to
    umift_env_runner.run_layup_scan_eval() (used once, post-episode, via
    the "e" menu option): same underlying run_layup_scan.py subprocess
    call, but on a timer instead of a keypress, and with resume gated on
    the new heatmap being loaded (see load_compare_heatmap()'s docstring
    for why) rather than fired off in the background. A scan or compare
    failure still resumes control -- the never-paused-forever guarantee --
    it just means the heatmap slot keeps whatever it already had.
    """
    keyence_ws = Path(control_para["keyence_ws"])
    keyence_python = Path(control_para["keyence_python"])
    low_ply = Path(control_para["layup_eval_low_ply"])
    right_arm_ip = str(control_para["layup_eval_right_ip"])
    scan_script = keyence_ws / "scripts" / "run_layup_scan.py"

    # Snapshot the right arm's CURRENT joints -- NOT the fixed episode-start
    # right_start_joints (that's only correct for the post-episode "e"
    # restore). A mid-episode scan must restore to wherever the arm
    # actually was doing the task right before the pause.
    try:
        pre_scan_joints = read_right_arm_joints_via_rtde(keyence_python, right_arm_ip)
    except Exception as exc:
        print(
            f"[layup-periodic] scan {scan_id}: failed to read right arm "
            f"joints, aborting scan (RTDE control was never paused): {exc}"
        )
        return ScanOutcome(
            scan_id=scan_id, success=False, error=f"joint read failed: {exc}"
        )

    print(f"[layup-periodic] scan {scan_id}: pausing RTDE control...")
    if not env.pause_rtde_control():
        print(
            f"[layup-periodic] scan {scan_id}: pause_rtde_control() "
            "returned False; aborting scan."
        )
        return ScanOutcome(
            scan_id=scan_id, success=False, error="pause_rtde_control failed"
        )

    session_name = f"periodic_scan{scan_id}_{int(time.time())}"
    joints_csv = ",".join(f"{q:.9g}" for q in pre_scan_joints)
    t0 = time.monotonic()
    session_dir: Optional[Path] = None
    error: Optional[str] = None
    duration_s = 0.0
    try:
        try:
            compare_timeout_s = float(control_para["periodic_scan_compare_timeout_s"])
            scan_proc = subprocess.run(
                [
                    str(keyence_python),
                    str(scan_script),
                    "--session-name",
                    session_name,
                    f"--right-return-joints={joints_csv}",
                    f"--compare-low={low_ply}",
                    f"--compare-timeout={compare_timeout_s}",
                    # Only heatmaps + metadata/metrics are kept for periodic scans.
                    "--prune-after=heatmaps",
                ],
                cwd=str(keyence_ws),
                capture_output=True,
                text=True,
                check=False,
                # The scan subprocess now joins its own backgrounded compare
                # (see run_layup_scan.py steps 7.5/9.5) before returning, so
                # this timeout must cover scan + compare, not just the scan.
                timeout=float(control_para["periodic_scan_timeout_s"]) + compare_timeout_s,
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
            # merged.ply is pruned by run_layup_scan.py once the compare is
            # done; a failed compare just yields no heatmap below.
            heatmap = load_compare_heatmap(
                scan_id=scan_id,
                png_path=session_dir / "distance_heatmap.png",
                heatmap_channels=int(control_para.get("layup_heatmap_channels", 3)),
            )
            if heatmap is not None:
                env.set_latest_scan_heatmap(heatmap)
                print(
                    f"[layup-periodic] scan {scan_id}: live scan_heatmap_0 updated."
                )
        except subprocess.TimeoutExpired:
            error = (
                "run_layup_scan timed out after "
                f"{control_para['periodic_scan_timeout_s']}s"
            )
            print(f"[layup-periodic] scan {scan_id}: {error}")
        except Exception as exc:
            error = str(exc)
            print(f"[layup-periodic] scan {scan_id}: FAILED: {exc}")
    finally:
        duration_s = time.monotonic() - t0
        print(
            f"[layup-periodic] scan {scan_id}: resuming RTDE control "
            f"(duration={duration_s:.1f}s)..."
        )
        try:
            resumed = env.resume_rtde_control()
            if not resumed:
                resume_err = "resume_rtde_control() returned False"
                print(f"[layup-periodic] scan {scan_id}: {resume_err}")
                error = error or resume_err
        except Exception as resume_exc:
            print(
                f"[layup-periodic] scan {scan_id}: resume_rtde_control "
                f"raised: {resume_exc}"
            )
            error = error or f"resume_rtde_control raised: {resume_exc}"

    return ScanOutcome(
        scan_id=scan_id,
        success=error is None,
        session_dir=session_dir,
        duration_s=duration_s,
        error=error,
    )
