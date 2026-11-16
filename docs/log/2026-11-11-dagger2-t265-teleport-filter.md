---
date: 2026-11-11
stream: data
branch: ee_tag_traj repo, branch t265-teleport-filter (UMI-FT: main)
commit: ee_tag_traj 80ea4ad
status: done
artifacts:
  - /mnt/data/umidata/ft/datasets/stringer_v3.5_dagger_2/tag_traj/ (re-solved, filter on)
  - /mnt/data/umidata/ft/datasets/stringer_v3.5_dagger_2/acp_replay_buffer.zarr (rebuilt)
  - /mnt/data/umidata/ft/datasets/stringer_v3.5_dagger_2/2_only_buffer.zarr (re-merged, 63 eps)
  - /mnt/data/umidata/ft/datasets/stringer_v3.5_dagger_2/all_buffer.zarr (re-merged, 81 eps)
  - /mnt/data/umidata/ft/datasets/stringer_v3.5_dagger_2/{tag_traj_raw_t265,acp_replay_buffer_raw_t265.zarr,2_only_buffer_raw_t265.zarr,all_buffer_raw_t265.zarr,episode_source_map_raw_t265.json} (previous versions, kept for revert)
---
# T265 teleport filter in ee_tag_traj; dagger_2 rebuilt

**What changed:** `ee_tag_traj` (separate git repo, new branch `t265-teleport-filter`, `main` untouched) removes one-sample T265 position teleports before the pose-graph solve: for any 200 Hz step whose velocity-unexplained part exceeds `solver.t265_teleport_thresh_m` (default 0.03 m, 0 disables; set in `configs/example_stringer.yaml`), that part is subtracted from all later samples. Sidecars now store `t265_teleport_thresh_m` and `t265_teleport_t_unified`. The 13 dagger_2 sidecars were re-solved, `acp_replay_buffer.zarr` was rebuilt with `combine_umift_dataset.sh`, and `2_only_buffer.zarr` (base + dagger_2) and `all_buffer.zarr` (base + dagger + dagger_2) were re-merged. The previous versions are kept beside them with a `_raw_t265` suffix.

**Why:** follow-up to `2026-11-11-dagger2-pose-jumps-t265.md`: the sharp jumps in dagger_2 are T265 teleports that the solver copies through tag-free spans.

**Result / status:**
- 68 teleports removed in 6 of 13 recordings (16.10.18: 3, 16.15.25: 3, 16.27.39: 8, 16.29.50: 16, 16.42.22: 10, 16.52.35: 28).
- Sidecar frame-to-frame steps above 0.5 m/s: 67 -> 0; max speed per recording 0.25–0.43 m/s.
- `acp_replay_buffer.zarr` `ts_pose_fb_0` robot-rate steps above 1 m/s: 639 -> 0. Episode count, meta lengths, timestamps, wrench, and sampled rgb are identical to the previous zarr.
- ee_tag_traj tests: 36 passed, 1 skipped.
- Not done: `stringer_v3.5_base` and `stringer_v3.5_dagger` episodes inside the merged buffers still use unfiltered sidecars. `2_only_dataset.tar` and `all_dataset.tar` are stale (built 2026-11-09 from the old zarrs). The `episode_*_expert_only_3d.html` viewers in the folder show the old poses. Any normalizer or checkpoint fit on the old dagger_2 poses predates this.
- Any dataset processed with this branch checked out gets the filter; with `main` checked out it does not.

**Repro:**
```
cd /home/boeing/umi_ft_ws/ee_tag_traj && git checkout t265-teleport-filter
python -m ee_tag_traj.pipeline --zarr-dir /mnt/data/umidata/ft/datasets/stringer_v3.5_dagger_2/zarr --config configs/example_stringer.yaml --output-dir /mnt/data/umidata/ft/datasets/stringer_v3.5_dagger_2/tag_traj
bash UMI-FT/UMIFT_Data/bash/combine_umift_dataset.sh stringer_v3.5_dagger_2
python UMI-FT/UMIFT_Data/scripts/merge_umift_zarr_datasets.py --inputs <base> [<dagger>] <dagger_2> --output <2_only_buffer|all_buffer>.zarr
```
Revert data: delete the four new artifacts and rename the `_raw_t265` ones back. Revert code: `git checkout main` in `ee_tag_traj`.
