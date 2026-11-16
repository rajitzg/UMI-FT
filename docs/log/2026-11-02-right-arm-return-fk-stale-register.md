---
date: 2026-11-02
stream: infra
branch: simple_umi_test
commit: uncommitted
status: done
artifacts:
  - /home/boeing/keyence_scan_ws/src/keyence_scan/layup/motion.py
  - /home/boeing/keyence_scan_ws/tests/test_right_return_path.py
  - /mnt/data/boeing/scans/inference_ep0_1793667959   # first successful restore (17:07)
---
# Right-arm scan return: root cause was a stale-register FK call (fixed, verified on robot)

Supersedes `2026-11-02-right-arm-scan-return-legs.md` (its design is kept; its "cause unknown" for the two faults is
resolved here).

**Root cause.** `move_right_to_joints` computed the startup TCP with `right_c.getForwardKinematics(q)` (no TCP offset).
In ur_rtde 1.6.4, when `tcp_offset` is empty or all zeros the C++ sends only registers 0-5 (joints); the URScript
(cmd 44) still reads `tcp_offset = pose_from_input_float_registers(6)`, i.e. whatever the previous command left in
registers 6-11. After `moveJ(speed 0.6, accel 0.6)` that is a bogus offset of ~(0.6, 0.6, 0). The controller returned
startup TCP (-0.72, 0.55, 1.02) instead of ~(-0.17, 0.83, 0.42), ~0.85 m away. Both earlier faults ("moves out in +z, spins,
faults") were the arm driving toward that point; the old moveJ-only path was a separate, earlier fault.

**Fix.** `_fk_with_tcp()` always passes an explicit non-zero TCP offset read from `getTCPOffset()` (all-zero is nudged to
1e-9 so the all-registers path is used). Before any leg beyond the clearance moveJ, the controller FK of the *current*
joints must match `getActualTCPPose()` within 5 mm (`RETURN_FK_TOL_M`), else `RuntimeError` and the arm stays put.
Tests (93 in keyence_scan_ws pass; the fake RTDE now models the stale-register behaviour).

**Result (robot, 2026-11-02 17:07, attended).** FK check error 0.0 mm. Real TCP offset is [-0.0049, -0.0003, 0.1442]
(not the nominal 0.143). Clearance moveJ 4.2 s, then slide / descend / guarded final approach all ended within 0.3 mm of
target at (-0.1692, 0.8354, 0.4162); no contact trip. One run only; the force-watch contact path has not fired on hardware.

**Separate, unfixed:** ~40 s stall before the right arm moves (right arm idle at park). `stop_after_next_chunk`
(`pipeline/handle.py`) waited the full 30 s `next_chunk_timeout` ("need chunks_completed >= 10, have 9"), then Keyence
`TE,0`/`TE,1` to 192.168.1.10:8500 timed out (~6 s). Not related to the right arm. Cause of the missing chunk unknown;
shortening `--next-chunk-timeout` would cut most of it but likely loses the last partial chunk (already the case on timeout).
