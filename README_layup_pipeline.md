# Layup pipeline

Commands are from `/home/boeing/umi_ft_ws`. Current data is `stringer_scanner_v2`.

## Datasets

Most recent: `/mnt/data/umidata/ft/datasets/stringer_scanner_v2/`

- Combined zarr (75 episodes, 2026-09-27): `acp_replay_buffer.zarr`
- Heatmaps zeroed: `acp_replay_buffer_zero_scan.zarr`
- Per-episode pieces: `zarr/` and `bags/`
- Open-loop rollouts for the checkpoint wired into inference: `/mnt/data/umidata/ft/training_outputs/stringer_scanner_v2/policy_open_loop_val.zarr` (from `epoch=0025-train_loss=0.020.ckpt`)

Previous scanner set: `/mnt/data/umidata/ft/datasets/stringer_scanner/`

- Tag-pose zarr: `acp_replay_buffer.zarr` (47 episodes)
- T265-pose zarr: `t265_acp_replay_buffer.zarr`
- Heatmaps zeroed: `t265_acp_replay_buffer_zero_scan.zarr`
- Last checkpoint on that ablation: `/mnt/data/umidata/ft/training_outputs/stringer_scanner/no_scan_or_tag/epoch=0175-train_loss=0.013.ckpt`

Live inference loads the v2 `epoch=0025` checkpoint, not that older one.

Older RGB sets (no `scan_heatmap_0`):

| Path | Episodes |
|---|---|
| `/mnt/data/umidata/ft/datasets/stringer_data_v2_full/acp_replay_buffer_10hz_224.zarr` | 52 |
| `/mnt/data/umidata/ft/datasets/stringer_teleop_merged_10hz_224/acp_replay_buffer.zarr` | 99 |
| `/mnt/data/umidata/ft/datasets/combined_umi_teleop/acp_replay_buffer.zarr` | 136 |
| `/mnt/data/umidata/ft/datasets/stringer_data_v1/acp_replay_buffer.zarr` | 15 |

Scanner and v2 both have `rgb_0`, `depth_0`, and `scan_heatmap_0`. `ultrawide_0` is a zero placeholder on the T265/D405 recordings. The reference layup scan used by eval is `/mnt/data/boeing/scans/scan_20260927_151409/merged.ply`.

## Steps

**1. Bags → per-episode zarr** (skip if `zarr/*_unified.zarr` already exists)

```bash
bash UMI-FT/UMIFT_Data/bash/process_ft_wild_recordings.sh stringer_scanner_v2 --skip-scp
```

**2. Combine into one training zarr** (also writes virtual targets)

```bash
bash UMI-FT/UMIFT_Data/bash/combine_umift_dataset.sh stringer_scanner_v2
```

Output: `/mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer.zarr`

Optional heatmap ablation:

```bash
python UMI-FT/UMIFT_Data/scripts/zero_scan_heatmaps.py \
  --input  /mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer.zarr \
  --output /mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer_zero_scan.zarr
```

**3. Train** (`pyrite` env). The yaml path is stale; pass the zarr.

```bash
cd /home/boeing/umi_ft_ws/UMI-FT/PyriteML
export PYRITE_CHECKPOINT_FOLDERS=/mnt/data/umidata/ft/training_outputs
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name=train_umift_workspace \
  task.dataset_path=/mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer.zarr
```

On the cluster, `UMI-FT/docker/train_klone.sbatch` does the same with `task.dataset_path=/data/acp_replay_buffer.zarr`.

**Building the training image (Docker -> Apptainer `.sif`)**

```bash
cd /home/boeing/umi_ft_ws/worktrees/<worktree>
docker build -f docker/Dockerfile --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t umift-train:local .
cd /home/boeing/umi_ft_ws/data/images        # existing images: umift-train_v*.sif
apptainer build umift-train_vN.sif docker-daemon://umift-train:local   # bump N
```

To use it on Hyak, copy the `.sif` over yourself (e.g. `scp`/`rsync`; needs interactive login + Duo) and point `SIF=` at it
when submitting: `SIF=/gscratch/.../umift-train_vN.sif TASK=... sbatch UMI-FT/docker/train_klone.sbatch`
(default is `/gscratch/scrubbed/rzghosh/images/umift-train.sif`). Local runs use the `pyrite` conda env, not the image.

**4. Point inference at the checkpoint** in `UMI-FT/PyriteEnvSuites/env_runners/umift_env_runner.py`.

`ckpt_path` is `/stringer_scanner_v2/epoch=0025-train_loss=0.020.ckpt`, joined with `$PYRITE_CHECKPOINT_FOLDERS`. Full file: `/mnt/data/umidata/ft/training_outputs/stringer_scanner_v2/epoch=0025-train_loss=0.020.ckpt`. A later `epoch=0035` sits next to it and is not what inference loads.

**5. Run inference**

```bash
conda activate pyrite
export PYRITE_CHECKPOINT_FOLDERS=/mnt/data/umidata/ft/training_outputs
export PYRITE_HARDWARE_CONFIG_FOLDERS=/home/boeing/umi_ft_ws/hardware_interfaces/workcell/table_top_manip/config
export PYRITE_CONTROL_LOG_FOLDERS=$HOME/data/control_log
cd /home/boeing/umi_ft_ws
python UMI-FT/PyriteEnvSuites/scripts/kill_ports.py
python UMI-FT/PyriteEnvSuites/env_runners/umift_env_runner.py
```

Jog the UR to the start pose and put it in remote mode first. A video window opens; press `q` when the stream looks right, then follow the on-screen menu. `periodic_scan_enabled` is already `True` in the runner.

Open-loop eval (no robot): `UMI-FT/PyriteEnvSuites/scripts/run_policy_open_loop_val_export.py`

## Scan-crop / absolute-pose experiments (`umift_ss_v2_*` task configs)

Same `stringer_scanner_v2` zarr (the real-scan `acp_replay_buffer.zarr`, not `_zero_scan`); everything is a task config, so one training image covers all of them.

| `task=` | What changes |
|---|---|
| `umift_ss_v2` | Baseline replica of the ss_v2 run (+ scan fill = zeros before the first scan). |
| `umift_ss_v2_crop` | Scan heatmap replaced by an EE-centered, heading-aligned 16 cm / 64 px crop, oriented like the wrist camera image (`crop.heading_axis: "-x"`, `crop.flip_v: true`: right = -tool x, down = -tool y, not mirrored). Scan encoder switches to flatten pooling. Checkpoints trained before these flags existed used heading `x`, no flip (a mirrored view); they keep working because the missing flag defaults to that. To reproduce that old orientation: `task.shape_meta.obs.scan_heatmap_0.crop.heading_axis=x task.shape_meta.obs.scan_heatmap_0.crop.flip_v=false`. |
| `umift_ss_v2_abs_input` | Relative pose obs kept; absolute fixture-frame (tag0) pose added as extra input. |
| `umift_ss_v2_abs_obs` | Pose obs absolute (fixture frame) instead of relative; actions still relative. |
| `umift_ss_v2_abs_obs_action` | Obs and actions both absolute (fixture frame). |

```bash
# local
cd UMI-FT/PyriteML
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name=train_umift_workspace \
  task=umift_ss_v2_crop task.dataset_path=/mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer.zarr \
  logging.name=ss_v2_crop dataloader.batch_size=160 training.num_epochs=100
# cluster
TASK=umift_ss_v2_crop EXTRA_ARGS="logging.name=ss_v2_crop dataloader.batch_size=160 training.num_epochs=100" \
  sbatch UMI-FT/docker/train_klone.sbatch
```

Scan encoder pooling (any task): `policy.obs_encoder.scan_heatmap_encoder_cfg.pooling=avg|flatten|spatial_softmax` (default: flatten for the crop, else avg). `spatial_softmax` outputs learned keypoint (x, y) coordinates plus avg-pooled features; optional knobs `.num_keypoints` (32), `.include_avg` (true), `.init_temperature` (0.1) and `.num_downsample` (4, or 3 for spatial_softmax). On an image built before these keys were declared in `train_umift_workspace.yaml`, prefix each override with `+`. The encoder prints `scan_heatmap ...: pooling=...` at startup.

Frames: absolute poses are in the fixture frame (origin at tag0), so they survive moving the stringer if the tags move with it. On the robot the runner maps RTDE poses into that frame with `PyriteConfig/tasks/umift/layup_workcell_extrinsics.yaml` (`T_right_base_from_fixture`); point `UMIFT_WORKCELL_EXTRINSICS` at a re-measured yaml after moving the part. The crop additionally needs the heatmap extent and `T_scanner_base_from_fixture` of the *current* scan (same yaml). Flags live in `shape_meta` (`pose_frames`, `obs.scan_heatmap_0.crop`, `*_abs_eef_*` keys) and ship with the checkpoint, so the runner needs no extra switches. Code: `workcell_frames.py`, `scan_crop.py`, `umift_type_conversions.sparse_obs_to_obs_sample`.

Inference diagnostics (`umift_env_runner.py`, `control_para`): `show_scan_panel` adds the heatmap / EE-centered crop the policy actually received (colour-mapped, cross = tool position for crops) to the rgb/depth popup, and `monitor_scan_encoder` hooks the scan CNN and prints a `[scan-enc] ...` line per policy query (also logged under `horizon_*/scan_encoder` in the control log, with the 128-d feature). Status: `ok`, `COLLAPSE?` (feature barely moves while the input does), `IGNORES-INPUT` (output equals the all-zero-input feature), `DEAD-RELU?`, `input-static` / `input-blank` (nothing to distinguish). Code: `PyriteUtility/planning_control/scan_encoder_monitor.py`.

## Scripts

| What | Path |
|---|---|
| Per-episode bag to unified zarr | `UMI-FT/UMIFT_Data/bash/process_local_recordings.sh` |
| Combine + virtual targets | `UMI-FT/UMIFT_Data/bash/combine_umift_dataset.sh` |
| Unified zarr to UMIFT schema | `UMI-FT/UMIFT_Data/scripts/convert_unified_t265_wild_to_umift_zarr.py` |
| LeRobot teleop to zarr | `UMI-FT/UMIFT_Data/scripts/convert_lerobot_teleop_to_umift_zarr.py` |
| Merge zarrs | `UMI-FT/UMIFT_Data/scripts/merge_umift_zarr_datasets.py` |
| Virtual-target labels | `UMI-FT/PyriteUtility/PyriteUtility/data_pipeline/postprocessing_add_virtual_target_label_umift.py` |
| Zero scan heatmaps | `UMI-FT/UMIFT_Data/scripts/zero_scan_heatmaps.py` |
| Drop scan-pause windows from training targets | `UMI-FT/PyriteML/diffusion_policy/common/sampler_umift.py` |
| Train | `UMI-FT/PyriteML/train.py` (`train_umift_workspace`) |
| Inference | `UMI-FT/PyriteEnvSuites/env_runners/umift_env_runner.py` |
| Review / flag / export episodes (browser) | `UMI-FT/UMIFT_Data/review/review_server.py` (see `UMIFT_Data/review/README.md`) |

## Reviewing episodes

To browse base and DAgger episodes with synced wrist video, pose/force plots and the checkpoint's observation window, run
`python UMIFT_Data/review/review_server.py --ckpt <ckpt>` and open `http://127.0.0.1:8770`. Flag episodes keep/reject, then
export the kept ones to a new `acp_replay_buffer.zarr` with `source` tags. See `UMIFT_Data/review/README.md`.

## View as LeRobot

UMIFT training still uses zarr. For LeRobot episode viewing, convert:

```bash
# no-scan set
python UMI-FT/UMIFT_Data/scripts/convert_umift_zarr_to_lerobot.py \
  --input  /mnt/data/umidata/ft/datasets/stringer_data_v2_full/acp_replay_buffer_10hz_224.zarr \
  --output /mnt/data/umidata/ft/datasets/stringer_data_v2_full/lerobot_acp_replay_buffer_10hz_224 \
  --repo-id local/stringer_data_v2_full \
  --overwrite

# tag-pose scanner set (includes depth + scan_heatmap)
python UMI-FT/UMIFT_Data/scripts/convert_umift_zarr_to_lerobot.py \
  --input  /mnt/data/umidata/ft/datasets/stringer_scanner/acp_replay_buffer.zarr \
  --output /mnt/data/umidata/ft/datasets/stringer_scanner/lerobot_acp_replay_buffer \
  --repo-id local/stringer_scanner_tag \
  --overwrite
```

Frames follow `rgb_time_stamps_0`. Depth and scan heatmaps are stored as viewable videos plus float arrays. Episode-level `scan_gap_intervals_0` is in `meta/umift_episode_extras.json`.

## Notes

`dataset_path` in `PyriteML/diffusion_policy/config/task/umift_zucchini.yaml` still points at `stringer_teleop_merged_10hz_224`. Pass `task.dataset_path` (the sbatch job does).

`combine_umift_dataset.sh` rebuilds the whole zarr every run. It prepends 5 seconds of the first frame (`--start-pad-seconds`, default 5). Virtual-target postprocessing skips episodes it already labeled.

In `umift_env_runner.py`, `set_depth_to_zero` is `True`. `sparse_execution_horizon` must stay at or below the checkpoint's action horizon (the 10 Hz config uses 32; the runner is set to 16). Scan-pause windows are excluded from action targets, but zeroing heatmaps does not remove those gaps.
