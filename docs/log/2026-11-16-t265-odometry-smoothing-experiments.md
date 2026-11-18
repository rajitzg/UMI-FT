---
date: 2026-11-16
stream: data
branch: ee_tag_traj repo, branch odom-fusion-experiments (worktree worktrees/ee_tag_traj-odom_fusion, from t265-teleport-filter 80ea4ad); UMI-FT: main
commit: uncommitted (ee_tag_traj experiments/odom/)
status: done
artifacts:
  - /mnt/data/umidata/ft/analysis/t265_smoothing/ (all outputs, 1.5 GB; nothing in datasets/ was modified)
  - /mnt/data/umidata/ft/analysis/t265_smoothing/pipeline/{sweep1,sweep2,sweep3}/ (per-recording json + solved fixture<-tcp .npy per variant)
  - /mnt/data/umidata/ft/analysis/t265_smoothing/{sweep1,sweep2,sweep3}.log, contact_sweep{1,2,3}.json, eval_v1..v6.json
  - /mnt/data/umidata/ft/analysis/t265_smoothing/time_offset.json (per-recording T265 vs camera clock offset)
  - /mnt/data/umidata/ft/analysis/t265_smoothing/ir_calib.json (D405 IR1 intrinsics, stereo baseline, T_ir1_color)
  - /mnt/data/umidata/ft/analysis/t265_smoothing/tags/ (per-recording single-tag PnP cache, 93 recordings)
  - /mnt/data/umidata/ft/analysis/t265_smoothing/cuvslam_*/, t265s_*/, t265v_*/, rT_tC/, comp*/ (derived odometry streams)
  - /mnt/data/umidata/ft/analysis/t265_smoothing/cuvslam_venv/ (pycuvslam 17.0.1 venv; wheel in dl/)
  - /mnt/data/umidata/ft/datasets/stringer_v3.5_{base,dagger,dagger_2,dagger_3}/zarr/*_unified.zarr (inputs, read only)
---
# T265 odometry smoothing experiments: cuVSLAM, IMU, clock sync, motion prior, smoothing

**What changed:** No pipeline or dataset change. Added offline experiment scripts in ee_tag_traj
`experiments/odom/` and ran them on all 93 recordings of stringer_v3.5_{base,dagger,dagger_2,dagger_3}.
Inputs are the unified zarrs, which are a lossless repack of the Pi bags (checked identical to `/t265/odom`
read from a bag). No filtering is applied before the tests.

**Why:** Recent T265 tracking looked bad (jumps, noise). Question: can cuVSLAM on the D405, IMU fusion,
or a smoothness heuristic give better TCP trajectories than the current ee_tag_traj solve?

**Method:**
- Odometry-only check: RPE of each odometry stream against independent single-tag PnP (only frames with an
  unambiguous IPPE solution, ratio >= 1.6, largest tag), in gap bins 0.2-0.5 / 0.5-1 / 1-2 / 2-5 / 5-10 s.
- End-to-end check (`exp_pipeline.py`): the full ee_tag_traj solve per variant, with tags held out in 2 s blocks
  (2 folds). Scores: held-out tag error, rotation error, jumps (> 0.5 m/s per camera frame), hf (residual after
  1 Hz low-pass), accel p99, and the fit error on the training tags. A third full solve is used for smoothness.
- Physics check (`contact_metric.py`): during contact (force > 10th percentile + 8 N), the TCP should sit on the
  mold. Score = robust sigma of TCP signed distance to the fixb_field.npz mesh. Lower = more physical.

**Findings, odometry sources:**
- FT accelerometer: it IS saved (`sensors/ft_acceleration`, about 29 Hz, in g, FT frame) but nothing downstream
  uses it. It matches the T265 IMU to about 1 m/s^2 (dynamic part corr 0.86-0.96). It has no gyro, so it adds
  nothing over the T265 IMU. At most a tilt check.
- T265 clock: T265 timestamps are offset by **-34 ms median** (range -25 to -45 ms) vs the camera arrival stamps
  (gyro cross-correlation against cuVSLAM). The current pipeline applies no offset.
- T265 scale: T265 under-reports translation by about 6 % (tag/T265 scale median 1.062, p10-p90 0.95-1.18).
  cuVSLAM scale is 1.00. Tag scale is fine (PnP depth / D405 depth = 1.0125).
- Tag map: the measured separation between tags 1 and 3 is 143-146 mm (n = 5). The config has 135.6 mm, and the
  config comment says 171 mm. The tag map may be wrong. Not yet checked with a ruler.
- RPE vs tags (median of per-recording medians, mm), 0.2-0.5 / 0.5-1 / 1-2 / 2-5 / 5-10 s:
  - T265 raw: 10.2 / 13.9 / 18.6 / 18.9 / 31.0 (873 jumps total)
  - T265 + 3 cm teleport filter + sync: 9.4 / 13.5 / 16.1 / 17.5 / 25.9 (232 jumps)
  - cuVSLAM RGB-D: 5.8 / 10.4 / 16.3 / 58.1 / 130.6. Better than T265 below 1 s, but it drifts badly after that
    (rotation drift 5 deg at 1-2 s, 18 deg at 2-5 s).
  - T265 rotation + cuVSLAM translation (rT_tC): 5.9 / 10.4 / - / 66.6. Still drifts.
  - T265 complementary (low-pass position + high-pass integrated reported velocity, t265v_300 / t265v_1000):
    7.9 / 12.5-13.2 / - / 17.7-18.6. Jumps fall from 251 to about 130-145.
- Dead ends:
  - cuVSLAM IMU and multisensor modes diverge (zero quaternions, km-scale positions).
  - cuVSLAM stereo IR is very poor (18 mm baseline).
  - IR-RGBD gives the same result as color RGB-D.
  - cuVSLAM SLAM (loop closure) gives no gain.
  - Complementary blends of T265 + cuVSLAM are worse than either alone.
  - Plain Gaussian smoothing of T265 (t265s_120) cuts jumps but does not improve RPE.

**Findings, end-to-end** (sweep2/sweep3: 24 recordings, every 4th across all 4 datasets).
Variant tokens: `sync` = -34 ms clock offset; `m<k>` = motion-edge sigma / k, so the solve trusts T265 k times
more (the current default is weak, about 18 mm per frame, so the solve snaps to noisy single-tag PnP);
`s<ms>` = output SE(3) smoothing; `v<ms>` = T265 complementary stream; `scale` = per-recording T265 scale.

| variant | held-out err med / p90 mm | rot deg | contact sigma mm | jumps | accel p99 m/s^2 | fit mm |
|---|---|---|---|---|---|---|
| base (current) | 14.9 / 38.6 | 2.66 | 9.9 | 13 | 4.1 | 0.8 |
| sync | 13.9 / 37.7 | 2.53 | 10.5 | 16 | 4.3 | 0.9 |
| sync+m10 | 13.0 / 33.3 | 1.97 | 7.7 | 22 | 4.2 | 3.4 |
| **sync+m30** | **12.7 / 33.2** | **1.92** | **7.0** | 22 | 4.2 | 5.1 |
| sync+m100 | 14.4 / 39.9 | 2.36 | 8.7 | 18 | 4.1 | 10.9 |
| sync+m300 | 18.7 / 64.4 | 2.56 | 9.0 | 16 | 4.2 | 17.6 |
| sync+m30+s60 | 12.9 / 33.1 | 1.95 | 7.6 | 11 | 1.8 | 6.8 |
| v1000+m30 | 12.2 / 36.0 | 1.90 | 7.8 | 7 | 2.7 | 4.2 |
| v300+m30 | 12.6 / 40.1 | 1.91 | 7.8 | 12 | 2.7 | 4.4 |
| v300+m30+s60 | 12.8 / 40.1 | 1.92 | 8.5 | 15 | 1.6 | 6.7 |
| sync+m100+s60 | 14.4 / 39.4 | 2.41 | 9.2 | 11 | 1.8 | 10.7 |
| sync+scale+m100 | 16.0 / 37.8 | 2.47 | 8.8 | 25 | 4.4 | 9.2 |
| sync+m100+nfev60 | 14.5 / 40.0 | 2.39 | 8.7 | 18 | 4.1 | 11.1 |

sweep1 (9 base recordings only, same trend): base 18.8 / 72.9 mm, contact sigma 12.1. sync+m30: 15.7 / 41.5, 7.5.
rTtC+m3 (cuVSLAM translation in the solve): 45.5 / 172, 21.1.
The rows from sweep2 and sweep3 use the same 24 recordings (sync+m30 is identical in both).

**Result / conclusion:**
- Best accuracy: **sync+m30**. Held-out error -15 %, rotation error 2.66 to 1.92 deg, contact sigma -29 % vs base.
  m10 is slightly worse. m100 and tighter priors over-trust T265 (the fit to the tags degrades to 10-18 mm).
- For smoother targets: add s60 (accel p99 4.2 to 1.8 m/s^2, jumps 22 to 11, accuracy almost the same). Or use
  v1000+m30 (jumps 7, accel 2.7, best median error, but p90 and contact sigma are worse).
- Do not use: cuVSLAM in the solve, cuVSLAM IMU modes, per-recording scale correction, or a higher nfev.
- Remaining jumps when no tag is in view: 31-51 mm per frame, at about 2x the T265 reported speed. These are
  T265 position glitches under the 3 cm teleport filter.
- Next: put sync + motion prior / 30 (optional s60) into ee_tag_traj (config + time offset), re-solve the
  datasets, rebuild the zarrs, and train. Check the tag 1-3 separation with a ruler.

**Repro** (from `worktrees/ee_tag_traj-odom_fusion/experiments/odom`; analysis scripts with
`PYTHONPATH= ~/miniforge3/envs/pyrite/bin/python`, cuVSLAM with `PYTHONPATH= .../t265_smoothing/cuvslam_venv/bin/python`):
```bash
python tags_cache.py                                    # single-tag PnP cache
.../cuvslam_venv/bin/python run_cuvslam.py --mode rgbd  # also ir_rgbd, stereo, *_imu, --slam
python time_offset.py; python calib_ir.py; python calib_color_ir.py
python fuse.py --variants t265s_60 t265s_120 rT_tC comp_700
python t265_velfuse.py                                  # t265v_{vint,100,300,1000}
python eval_odom.py                                     # RPE tables (eval_v*.json)
python scale_check.py; python tag_scale_check.py
PYTHONPATH=/home/boeing/umi_ft_ws/worktrees/ee_tag_traj-odom_fusion ~/miniforge3/envs/pyrite/bin/python \
  exp_pipeline.py --run sweep3 --every 4 --workers 12 \
  --variants sync+m10 sync+m30 v300+m30 v1000+m30 v300+m30+s60 sync+m30+s60
python contact_metric.py --run sweep3
```
`config.py` resolves `parents[2]/Robotiq-portable-real2sim`, so the worktree needs the symlink
`worktrees/Robotiq-portable-real2sim -> ../Robotiq-portable-real2sim`.
