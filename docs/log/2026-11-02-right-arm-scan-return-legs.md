---
date: 2026-11-02
stream: infra
branch: simple_umi_test
commit: uncommitted
status: superseded-by 2026-11-02-right-arm-return-fk-stale-register.md
artifacts:
  - /home/boeing/keyence_scan_ws/src/keyence_scan/layup/motion.py
  - /home/boeing/keyence_scan_ws/scripts/run_layup_scan.py
  - /home/boeing/keyence_scan_ws/utils/check_right_return_path.py
  - /home/boeing/keyence_scan_ws/tests/test_right_return_path.py
  - /home/boeing/umi_ft_ws/UMI-FT/runs/episode_1793665439   # last episode before the 16:29 fault
  - /mnt/data/boeing/scans/inference_ep0_1793665750          # the scan whose restore faulted
---
# Right-arm restore after scan eval ("e"/"s"): decoupled legs + force-watched final approach

**What changed (keyence_scan_ws repo, uncommitted; HEAD 7716c06 still holds the first attempt):**
`move_right_to_joints` no longer ends with one moveJ onto the startup joints. After the existing moveJ back to the
clearance snapshot it now runs `plan_right_return_legs`: (1) wrist-3-only moveJ to undo the yaw in place at clearance
height, (2) moveL slide to startup x/z holding y, (3) moveL straight down to 3 cm above the startup TCP,
(4) the last 3 cm at 0.03 m/s with a force watch (|dF| > 10 N on `getActualTCPForce`: stopL, retreat 1 cm, raise
`RightReturnContact`). `run_layup_scan.py` treats that contact stop as non-fatal (loud warning, scan result kept) and
now attempts the restore only once (the error handlers used to replay it from an unknown pose). Knobs in
`layup_path.yaml` `motion:` (all optional): `right_return_approach_m`, `right_return_final_speed_l`,
`right_return_final_accel_l`, `right_return_contact_n` (0 = off), `right_return_retreat_m`.
Same code path serves the periodic mid-episode scan (return pose there = the pressed pose, so the force watch may trip; that is non-fatal).

**Why (evidence, 2026-11-02):**
- Every episode ends with the tool yawed ~165-175 deg about its axis from the start orientation (reached within
  5-25 s; `UMI-FT/runs/episode_*/robot_data_0.json`), and the start TCP sits at working height (y ~0.83, ~6 cm above the
  tag plane y ~0.889; end pose same height). So the restore must undo a ~174 deg yaw close to the part.
- Old path (moveJ clearance -> start): TCP descends diagonally while wrist 3 swings 3.12 rad. It faulted with wrist 3 at
  -2.54 rad (83% through), TCP (-0.127, 0.799, 0.461): ~3 cm above start height, ~5 cm from the start pose
  (read over RTDE after the fault; matches the 16:30 stub episode).
- First attempt (hover moveL = yaw + 23 cm slide together, then a full-speed 20 cm drop onto the exact start pose) also
  faulted per the operator ("moved outward"). Offline IK shows that path is kinematically fine (no singularity, reach
  0.5-0.65 of 0.82 m, lands on the start joints), so the cause is physical/unknown; its log was not captured. Both
  faults are within a few cm of the start pose, hence the slow force-watched last approach.
- Not changed, residual risk: the scanner arm's move to `left_park` is an open-loop timed wait in `run_layup_scan.py`;
  the clearance pose lies inside the scanner's sweep volume (y ~0.44-0.47, x -0.39..-0.01, z 0.41-0.54, right_base).

**Result / status:** NOT yet run on the robot. Offline: `tests/test_right_return_path.py` (15, fake RTDE) and the
whole keyence_scan_ws suite (90) pass; `utils/check_right_return_path.py` on the logged joints reports PLAN OK
(yaw leg moves the TCP 0.0 cm; slide needs 0.012 rad of wrist 3; tool never below start height; final = start joints).
Next: first robot run with the arm attended, e-stop in hand; if it faults, capture the full scan stderr (each leg and
the arm state on failure are now logged).

**Repro (no robot):**
`cd /home/boeing/keyence_scan_ws && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python utils/check_right_return_path.py --start-q=-1.49097091,-0.808212237,1.34428102,-3.92690267,-1.65203935,-3.0669089 --lift-q=-1.92884714,-1.43586739,1.99554855,-3.89961423,-1.3404134,0.05480766`
(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed because the ROS pytest plugins on PATH import `lark`.)
