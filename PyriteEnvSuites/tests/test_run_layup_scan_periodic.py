"""Tests for run_layup_scan_periodic() against fake run_layup_scan.py /
compare_pointcloud_distance.py stand-ins.

No hardware and no umift_env_runner.py import required: this module is
deliberately dependency-light (see its own docstring), so it can be tested
directly the way scan_pause_controller.py (which it replaces) used to be.

Covers:
  * happy path: pause -> scan -> heatmap loaded -> resume, in that exact
    order -- the core correctness property this replaces ScanPauseController
    for (the policy is conditioned on scan_heatmap_0, so control must not
    resume before the new heatmap is applied).
  * a scan failure still resumes RTDE exactly once, without ever pausing
    -- er, without attempting a heatmap update (never-paused-forever
    guarantee).
  * a compare failure/timeout still resumes RTDE exactly once and reports
    the scan itself as successful (matches the old ScanOutcome contract:
    heatmap unavailability doesn't gate scan success).
  * a joint-read failure aborts before ever pausing RTDE control.
  * pause_rtde_control() returning False aborts without calling resume
    (nothing was actually paused).
"""

import stat
import sys
import textwrap
from pathlib import Path

from PyriteEnvSuites.envs.task import run_layup_scan_periodic as mod

_FAKE_SCAN_SRC = textwrap.dedent(
    """
    #!/usr/bin/env python3
    # Fake stand-in for keyence_scan_ws/scripts/run_layup_scan.py.
    # Mirrors the real script's steps 7.5/9.5: when --compare-low is given,
    # runs the (fake) compare_pointcloud_distance.py itself and joins it
    # before exiting -- a compare failure logs but doesn't fail the scan.
    import argparse
    import subprocess
    import sys
    from pathlib import Path

    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--session-name", required=True)
        p.add_argument("--right-return-joints", default=None)
        p.add_argument("--compare-low", default=None)
        p.add_argument("--compare-timeout", type=float, default=120.0)
        p.add_argument("--prune-after", default="final-mesh")
        p.add_argument("--fail", action="store_true")
        args = p.parse_args()

        if args.fail:
            sys.exit(1)

        session_dir = Path.cwd() / "scans" / args.session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "merged.ply").write_bytes(b"fake ply")

        if args.compare_low is not None:
            compare_script = (
                Path(__file__).resolve().parents[1]
                / "utils" / "compare_pointcloud_distance.py"
            )
            cmp_proc = subprocess.run(
                [
                    sys.executable, str(compare_script),
                    "--high", str(session_dir / "merged.ply"),
                    "--low", args.compare_low,
                    "--no-show",
                    "--png", str(session_dir / "distance_heatmap.png"),
                    "--score-json", str(session_dir / "eval_score.json"),
                ],
                check=False,
            )
            if cmp_proc.returncode != 0:
                print(f"compare failed with code {cmp_proc.returncode}", file=sys.stderr)

        print(f"LAYUP_SESSION_DIR={session_dir}", flush=True)
        return 0

    if __name__ == "__main__":
        raise SystemExit(main())
    """
)

_FAKE_COMPARE_SRC = textwrap.dedent(
    """
    #!/usr/bin/env python3
    # Fake stand-in for keyence_scan_ws/utils/compare_pointcloud_distance.py.
    import argparse
    import sys
    import numpy as np
    from pathlib import Path

    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--high")
        p.add_argument("--low")
        p.add_argument("--no-show", action="store_true")
        p.add_argument("--png", required=True)
        p.add_argument("--score-json", required=True)
        p.add_argument("--fail", action="store_true")
        args = p.parse_args()

        if args.fail:
            sys.exit(1)

        png_path = Path(args.png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        grid = np.full((80, 160), 1.5, dtype=np.float32)
        grid[0, 0] = np.nan
        np.save(png_path.with_name(png_path.stem + "_small.npy"), grid)
        return 0

    if __name__ == "__main__":
        raise SystemExit(main())
    """
)


class _DummyEnv:
    """Stands in for ManipServerLayupEnv, recording call ORDER (not just
    counts) so tests can assert set_latest_scan_heatmap happens strictly
    before resume_rtde_control."""

    def __init__(self, pause_ok: bool = True, resume_ok: bool = True):
        self.pause_ok = pause_ok
        self.resume_ok = resume_ok
        self.calls: list = []

    @property
    def pause_calls(self) -> int:
        return self.calls.count("pause")

    @property
    def resume_calls(self) -> int:
        return self.calls.count("resume")

    def pause_rtde_control(self) -> bool:
        self.calls.append("pause")
        return self.pause_ok

    def resume_rtde_control(self) -> bool:
        self.calls.append("resume")
        return self.resume_ok

    def set_latest_scan_heatmap(self, heatmap) -> None:
        self.calls.append("set_heatmap")
        self.last_heatmap = heatmap


def _write_executable(path: Path, src: str) -> Path:
    path.write_text(src, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _make_control_para(tmp_path: Path, *, scan_fails: bool = False, compare_fails: bool = False) -> dict:
    keyence_ws = tmp_path
    (keyence_ws / "scripts").mkdir(parents=True, exist_ok=True)
    (keyence_ws / "utils").mkdir(parents=True, exist_ok=True)
    _write_executable(keyence_ws / "scripts" / "run_layup_scan.py", _FAKE_SCAN_SRC)
    _write_executable(keyence_ws / "utils" / "compare_pointcloud_distance.py", _FAKE_COMPARE_SRC)

    low_ply = tmp_path / "low.ply"
    low_ply.write_bytes(b"fake low ply")

    control_para = {
        "keyence_ws": str(keyence_ws),
        "keyence_python": sys.executable,
        "layup_eval_low_ply": str(low_ply),
        "layup_eval_right_ip": "0.0.0.0",
        "periodic_scan_timeout_s": 10.0,
        "periodic_scan_compare_timeout_s": 10.0,
        "layup_heatmap_channels": 3,
    }

    # The fakes' own --fail flags aren't part of run_layup_scan_periodic()'s
    # real argv-building (which only ever passes --session-name and
    # --right-return-joints=...); simulate "scan fails" / "compare fails" by
    # monkeypatching the fake scripts to always fail instead, since we can't
    # inject extra flags through the real call site.
    if scan_fails:
        _write_executable(
            keyence_ws / "scripts" / "run_layup_scan.py",
            _FAKE_SCAN_SRC.replace("args = p.parse_args()", "args = p.parse_args(); args.fail = True"),
        )
    if compare_fails:
        _write_executable(
            keyence_ws / "utils" / "compare_pointcloud_distance.py",
            _FAKE_COMPARE_SRC.replace("args = p.parse_args()", "args = p.parse_args(); args.fail = True"),
        )

    return control_para


def test_happy_path_resumes_strictly_after_heatmap_is_set(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "read_right_arm_joints_via_rtde", lambda *a, **k: [0.0] * 6)
    control_para = _make_control_para(tmp_path)
    env = _DummyEnv()

    outcome = mod.run_layup_scan_periodic(env=env, scan_id=1, control_para=control_para)

    assert outcome.success is True, outcome.error
    assert outcome.error is None
    assert outcome.session_dir is not None
    assert outcome.session_dir.name.startswith("periodic_scan1_")
    assert outcome.duration_s is not None and outcome.duration_s >= 0.0

    # The core correctness property: heatmap is applied BEFORE resume.
    assert env.calls == ["pause", "set_heatmap", "resume"]
    assert env.last_heatmap.shape == (3, 80, 160)
    assert env.last_heatmap.dtype.name == "float32"
    assert env.last_heatmap[0, 0, 0] == 0.0  # NaN cell replaced with 0.0
    assert env.last_heatmap[0, 1, 1] == 1.5


def test_scan_failure_still_resumes_without_heatmap_update(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "read_right_arm_joints_via_rtde", lambda *a, **k: [0.0] * 6)
    control_para = _make_control_para(tmp_path, scan_fails=True)
    env = _DummyEnv()

    outcome = mod.run_layup_scan_periodic(env=env, scan_id=2, control_para=control_para)

    assert outcome.success is False
    assert outcome.error is not None
    assert env.calls == ["pause", "resume"]  # no set_heatmap
    assert env.pause_calls == 1
    assert env.resume_calls == 1  # never-paused-forever guarantee


def test_compare_failure_still_resumes_and_scan_itself_counts_as_success(tmp_path, monkeypatch):
    """Matches the old ScanOutcome contract: heatmap unavailability doesn't
    gate overall scan success, only the live heatmap slot goes stale."""
    monkeypatch.setattr(mod, "read_right_arm_joints_via_rtde", lambda *a, **k: [0.0] * 6)
    control_para = _make_control_para(tmp_path, compare_fails=True)
    env = _DummyEnv()

    outcome = mod.run_layup_scan_periodic(env=env, scan_id=3, control_para=control_para)

    assert outcome.success is True, outcome.error
    assert env.calls == ["pause", "resume"]  # no set_heatmap
    assert env.resume_calls == 1


def test_joint_read_failure_never_pauses(tmp_path, monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("no RTDE connection")

    monkeypatch.setattr(mod, "read_right_arm_joints_via_rtde", _raise)
    control_para = _make_control_para(tmp_path)
    env = _DummyEnv()

    outcome = mod.run_layup_scan_periodic(env=env, scan_id=4, control_para=control_para)

    assert outcome.success is False
    assert env.calls == []
    assert env.pause_calls == 0
    assert env.resume_calls == 0


def test_pause_failure_returns_without_calling_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "read_right_arm_joints_via_rtde", lambda *a, **k: [0.0] * 6)
    control_para = _make_control_para(tmp_path)
    env = _DummyEnv(pause_ok=False)

    outcome = mod.run_layup_scan_periodic(env=env, scan_id=5, control_para=control_para)

    assert outcome.success is False
    assert env.calls == ["pause"]
    assert env.resume_calls == 0  # nothing was actually paused; no-op is correct
