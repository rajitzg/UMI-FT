---
date: 2026-11-11
stream: data
branch: main
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/analysis/dagger2_pose_jumps/ (diag.py, an.py, an2.py, resolve.py)
  - /mnt/data/umidata/ft/analysis/dagger2_pose_jumps/tag_traj_fixed/ (13 re-solved sidecars, not used by any zarr)
---
# stringer_v3.5_dagger_2 pose jumps come from the T265, not the tags

**What changed:** Analysis only. No dataset, sidecar, or code under `UMI-FT/` or `ee_tag_traj/` was modified. A trial re-solve of the 13 dagger_2 recordings with T265 teleports removed was written to the analysis folder.

**Why:** dagger_2 trajectories show sharp jumps. The surface_frame worktree had found one-sample T265 teleports in the October/November-1 data (`2026-11-10-odom-teleport-sidecar.md`, 15 cm threshold); the question was whether dagger_2 jumps are the same thing or tag detections.

**Result / status:**
- 50 frame-to-frame TCP steps above 1 m/s in the existing `tag_traj/*_ee_master.npz`, in 5 of 13 recordings (16.10.18: 3, 16.27.39: 8, 16.29.50: 11, 16.42.22: 6, 16.52.35: 22). Every one coincides with a one-sample (5 ms) jump in the raw 200 Hz T265 position that the T265's own reported velocity does not explain (reported 0.02–0.4 m/s, actual step 6–17 cm, rotation under 0.5 deg). All 50 happen with no tag in view on either side, and the solved step matches the odometry step to within 2.3 cm.
- 75 raw T265 jumps above 2 cm unexplained across 9 of 13 recordings. Sizes cluster near 13 cm and 7 cm and mostly alternate sign in the T265 body frame (out and back), so the tracker is flipping between two position estimates.
- Tags are not the cause: per-tag PnP agrees with the solved pose at median 0.0–0.1 cm (worst single detection 6.2 cm, one frame in 16.29.50); tags are visible in only 8–27% of frames and never two at once.
- The existing `fix_odom_teleport.py` threshold (15 cm) would miss most of these.
- Trial fix: subtract the velocity-unexplained part of any 200 Hz step above 2 cm before the pose-graph solve. Result: steps above 1 m/s 50 -> 0, above 0.5 m/s 67 -> 0, max speed 0.25–0.41 m/s in every recording, tag reprojection RMS unchanged. Positions move by up to 15 cm in the affected spans (16.42.22: 57% of frames move more than 2 cm).
- Open point: 16.49.55 has three same-direction 2.0–2.3 cm jumps; removing them shifts 44% of frames by more than 2 cm and the worst tag closure goes 7.4 -> 8.8 cm. A 3 cm threshold leaves those alone.
- Not done: the fix is not in `ee_tag_traj`, and `acp_replay_buffer.zarr` / `2_only_buffer.zarr` / `all_buffer.zarr` still carry the jumps.

**Repro:** (pyrite env) `cd /mnt/data/umidata/ft/analysis/dagger2_pose_jumps && python diag.py && python an.py && python an2.py`; re-solve: `python resolve.py $PWD/tag_traj_fixed`.
