---
date: 2026-11-17
stream: infra
branch: main
commit: uncommitted
status: done
artifacts: [<control log>/episode_*/d405_raw/ (new, per rollout when enabled)]
---
# Optional lossless D405 IR pair + depth recording during robot rollouts

**What changed:**
- New module `PyriteEnvSuites/env_runners/d405_raw_recorder.py` (`D405RawRecorder`).
- `D405VisionRingBuffer` gained optional IR capture (`capture_ir`), a full-resolution raw frame sink, and `stream_info`, which holds the serial, depth scale, intrinsics and extrinsics. If the D405 refuses to start with the IR streams, it retries without them, so the policy camera never goes down because of recording.
- `ManipServerLayupEnv` gained `record_d405_raw`, `set_d405_raw_frame_sink()` and `get_d405_stream_info()`.
- `umift_env_runner.py` has two new `control_para` keys, `record_d405_raw` (default False) and `record_d405_raw_color` (default False). When `record_d405_raw` is on, each episode writes `<control log>/episode_<t>/d405_raw/`:
  - `{infrared_1,infrared_2,depth}.raw` (and `color.raw` if enabled);
  - `manifest.json` in the same layout as the wand's live-ingest output.
- The manifest also carries `device` (intrinsics/extrinsics) and `clock_sync`: pairs of host `time.time()` and ManipServer time at start, at stop and every ~10 s. Use them to map frame `timestamp_sec` (host clock) onto the `ts_pose_fb` timestamps.
- Policy inputs are unchanged.
- Frames are written on their own thread. A slow disk drops frames (counted in `info.dropped_frames`) instead of stalling capture.
- Recording is skipped if less than 10 GB is free.

**Why:** to test FFS depth and the defect methods (`worktrees/UMI-FT-ffs_depth`, logs 2026-11-15/16) with the robot's stable pose instead of the handheld tag/T265 pose. Before this, the runner recorded only colour + depth, and only as the 224x224 policy inputs plus `cameras/wrist_d405.mp4`.

**Result:** camera-only check on the robot D405 (serial 335122272736, FW 5.17.0.9, USB 3.2), with no arm motion:
- color, depth, IR1 and IR2 all open together at 640x480, 15 fps; 80 framesets in 6 s, 0 dropped;
- the policy RGB and depth buffers kept filling.

The device reports:
- IR1 intrinsics fx = fy = 390.32, pp = (325.93, 239.42);
- IR1 to IR2 baseline 18.034 mm.

FFS (`23-36-37`) on a recorded pair: 92% valid pixels vs 58% for native depth; FFS minus native median −0.42 mm, MAD 0.43 mm.

Data rate is about 18 MB/s (~9 GB per 8-minute episode); with color, about 32 MB/s. Tests are in `PyriteEnvSuites/tests/test_d405_raw_recorder.py` (4 tests, pass). The two failing tests in `test_plot_inference_episode_vision.py` exercise code this change does not touch.

Not yet run on a full rollout.

**Repro:** set `"record_d405_raw": True` in `control_para` in `umift_env_runner.py`, then run as usual (AGENTS.md §6). Read the output like the wand's `*_camera` dirs, e.g. `ffs_depth_frames.py` / `ffs_episode.py` in the ffs_depth worktree.
