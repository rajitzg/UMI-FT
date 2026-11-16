# Episode review tool ("lightroom for demos") — design

Date: 2026-11-15. Stream: bc. Checkout: `worktrees/UMI-FT-standard_bc` (branch `simple_umi_test` at time of writing).

## Intent

Adding DAgger data to `stringer_v3.5` training makes the policy worse. Two failure modes seen on the robot:

1. The policy moves to the other side of the stringer before the first side is fully smoothed. Only happens when some DAgger data is included.
2. At a specific point the policy always moves up and away from the stringer into an out-of-distribution state.

Suspected causes are individual bad episodes: bad recoveries, noisy data, trajectories that don't match the wrist video (pose
errors), or misleading data (e.g. what the policy sees in its observation history doesn't justify the action that follows).

The tool lets the user page through every episode of several datasets (base + all DAgger sets), scrub each one smoothly with the
wrist video in sync with pose/force plots, see exactly what the policy observes at any instant, and mark episodes keep/reject.
Kept episodes are exported to a new replay buffer with DAgger episodes tagged `source=dagger`. Workflow model: Lightroom
(browse, flag, export).

## Goals

- One browser page; episode list across multiple zarrs; keyboard-driven flagging.
- Smooth: no lag while dragging the time slider or while the video plays. This is the top non-functional requirement.
- Minimum plots, matching `UMIFT_Data/scripts/plot_dataset_umift_ultrawide_depth.py`: absolute position, orientation, TCP force
  (and torque), each with a cursor at the current time.
- Policy observation history window drawn on the plots, derived from the training config (see "History window").
- 3D tool path with separate start/end sliders that restrict which part of the trajectory is drawn.
- Flags persist automatically; export writes a new `acp_replay_buffer.zarr` with correct `source` tags.

## Non-goals

- No Fix B / mold-frame / surface-frame content (that is the surface_frames front).
- No editing of episode contents (trimming, relabelling). Whole-episode keep/reject only in v1. Notes field allows recording
  "bad after t=40 s" for later.
- No policy inference / open-loop overlay in v1.
- No training launch from the tool.

## Anti-goals

- A viewer that re-renders server-side or rebuilds Plotly figures on every slider event (this is why existing viewers lag).
- Writing to source zarrs. All datasets are opened read-only; the tool only writes its own flag file, its cache, and the export.
- Silent wrong export: an exported episode tagged wrong (dagger as original or vice versa) defeats the purpose.

## Constraints / facts

- Datasets (defaults; list is a CLI arg): `/mnt/data/umidata/ft/datasets/{stringer_v3.5_base, stringer_v3.5_dagger,
  stringer_v3.5_dagger_2, stringer_v3.5_dagger_3}/acp_replay_buffer.zarr` (47, 17, 13, 18 episodes). Episode ids overlap
  across datasets, so the identity of an episode is `(dataset_name, episode_index)`, displayed as `dagger_2/ep7`.
- Which datasets are DAgger: by CLI flag, `--base` vs `--dagger` lists of zarr paths (none of the four source zarrs has a
  `source` attr today; if one is present and disagrees with the CLI, warn and use the CLI). Default: `stringer_v3.5_base` base, the three `_dagger*` sets DAgger.
- Per episode: `rgb_0` (N,224,224,3 uint8, ~15 Hz, chunked 208 frames), `rgb_time_stamps_0`, `ts_pose_fb_0` (T,7 pose7
  `[x,y,z,qw,qx,qy,qz]`, ~194 Hz, fixture frame), `robot_time_stamps_0`, `wrench_left_0` (W,6 TCP frame, ~28 Hz),
  `wrench_time_stamps_0`, `stiffness_0`, `ts_pose_virtual_target_0`. `wrench_right_0` is zeros. Optional `scan_gap_intervals_0`.
  Root attr `start_pad_seconds` (5.0): first 5 s is a padded copy of frame 0; shade it on the plots.
- `ee_tag_traj` sidecars `<dataset>/tag_traj/*_ee_master.npz` (object arrays; `allow_pickle=True`) carry per-frame tag
  visibility. Mapping from episode index to sidecar is via unified zarr names (`<dataset>/zarr/*_unified.zarr`, in
  converter order) — builder to confirm the mapping from the converter; if it can't be made reliable, skip the tag track.
- Episode videos are small (224x224, ~800–1500 frames). `/mnt/data` is 96% full (173 GB free); cache is small but goes under
  `~/.cache/umift_review/` (my call), keyed by zarr path + episode + mtime.
- Env: `pyrite` conda env (has `zarr`, `numpy`, `scipy`, `av` 18, `cv2`). `/usr/bin/ffmpeg` exists. Remember
  `source worktrees/activate.sh UMI-FT-standard_bc` so the right `PyriteUtility` is imported.
- Runs on the control PC; opened in a local browser at `http://127.0.0.1:<port>`.

## Approaches considered

1. **Local web app, client-side scrubbing (chosen).** Python stdlib HTTP server like `surface_frames_scripts/view_oct.py`
   serves episode data once; everything interactive runs in the browser. Most code (~1–1.5k lines) but the only option that
   gives custom flag/export UI plus smoothness.
2. Rerun viewer. Smooth scrubbing/video/plots/3D for free, but no custom keep/reject UI (selection becomes a separate text
   file + CLI) and history-window overlays are awkward. Rejected.
3. Qt/pyqtgraph desktop app. Fast plots, clunkier 3D/video, tied to the control PC display. Rejected.

User picked 1.

## Design

### Location

`UMIFT_Data/review/` in the bc worktree:
- `review_server.py` — CLI + HTTP server + data loading/caching.
- `review_export.py` — export of kept episodes (importable and CLI).
- `static/index.html`, `static/app.js`, `static/style.css`.
- `static/vendor/` — vendored `uPlot` (min js + css) and `three.module.min.js` + `OrbitControls` (my call: vendored so it works
  offline and versions are pinned).
- Tests in `UMIFT_Data/tests/test_review_*.py`.

Run: `python UMIFT_Data/review/review_server.py [--base Z ...] [--dagger Z ...] [--flags FILE] [--port 8770]
[--ckpt CKPT] [--rollout-slowdown 2] [--session NAME]`.

### Server

- On start: open all zarrs read-only, build the episode index (dataset name, index, source, duration, frame count). No heavy
  loading up front.
- `GET /api/episodes` → index + current flags.
- `GET /api/episode/<ds>/<i>/data` → one binary payload (my call: a small JSON header + concatenated little-endian Float32 arrays,
  so the client wraps them as typed arrays with no parsing). Contents, all on a common time base (seconds from episode start):
  - pose stream (~194 Hz, downsampled for plotting to ≤ ~4k points if longer; full-rate kept for 3D path): position xyz,
    orientation as euler xyz deg (same convention as the ultrawide script; unwrap to avoid ±180 jumps), speed (|dp/dt|,
    lightly smoothed), stiffness, virtual-target offset magnitude.
  - wrench stream: force xyz, torque xyz, |F|.
  - rgb frame times.
  - markers: start-pad end, scan gaps, tag-visibility track (if available), pose jumps (step > threshold between consecutive
    samples, my call: flag > 5 mm in one ~5 ms step).
- `GET /api/episode/<ds>/<i>/video.mp4` → H.264 MP4 of `rgb_0` at its real frame rate with keyframe every frame or every few
  frames (`-g 1`–`-g 5`; my call: all-intra `-g 1` since files are tiny and it makes seeking instant). Built on first request,
  cached; served with HTTP Range support (required for `<video>` seeking). Frame-accurate mapping: the client maps video time
  to frame index via the frame-time array, not by assuming constant fps.
- `GET /api/episode/<ds>/<i>/thumb.jpg` → a mid-episode frame for the list.
- Background prefetch: when the user opens episode k, server pre-builds video + data for k±1 (my call).
- `POST /api/flags` → write flags file (atomic replace). Flags are research artifacts outside git: default
  `/mnt/data/umidata/ft/datasets/review_flags/<name>.json` (my call: next to the data, durable, not in a cache), `<name>` from
  `--session` (default `stringer_v3.5`), overridable by `--flags`. Content per episode: `{"dataset_path", "episode", "source", "flag": keep|reject|none,
  "note", "updated", "fingerprint"}`. Fingerprint (my call) = episode length in each stream + first/last robot timestamp.
  Source zarrs get rebuilt/trimmed in place (`dagger_2` went 17 -> 13 episodes), so on load the server compares fingerprints and
  marks mismatched flags stale (shown in the list); export refuses while any selected flag is stale.
- `POST /api/export` → body `{output_path, include: "keep" | "keep+unflagged"}`; runs export; returns summary or error.

### Client layout

```
+------------------+------------------------------------------------------------+
| episode list     |  [wrist video 448x448]      [3D path (three.js)]           |
| (thumb, name,    |                              start ===o======o=== end      |
|  dur, flag dot,  +------------------------------------------------------------+
|  base/dagger     |  position xyz        ───────|▒▒▒▒▒▒|▓▓──────────            |
|  badge)          |  orientation xyz     ───────|▒▒▒▒▒▒|▓▓──────────            |
| filter: all/     |  force xyz + |F|     ───────|▒▒▒▒▒▒|▓▓──────────            |
| base/dagger/     |  torque / speed / stiffness (collapsible)                  |
| unflagged/keep/  |  markers: pad | scan gap | tag lost | pose jump            |
| reject           |  [time slider ===================o=======]  t=23.4 s  1x   |
| [Export...]      |  note: [__________________]   flag: KEEP (P) / REJECT (X)  |
+------------------+------------------------------------------------------------+
▒ = observation history (per modality)   ▓ = action horizon ahead   | = cursor
```

- Plots: uPlot, one chart per group, x axes synced (zoom on one zooms all; double-click resets). Cursor drawn as a vertical
  line via a lightweight overlay, not by redrawing the series.
- History window: shaded spans behind the cursor, per modality, drawn in the overlay (not the series):
  low-dim (pose) on position/orientation, wrench on force/torque, RGB as frame ticks under the video. Action horizon drawn
  ahead of the cursor. Also highlight the RGB history frames in a small filmstrip strip under the video (my call: 10 tiny
  thumbnails of exactly the frames the policy sees, decoded from the same video by seeking a hidden second `<video>` /
  or served as a sprite; builder picks whichever stays smooth).
- 3D path: full-rate polyline coloured by time (or by |F|, toggle), restricted to [start, end] slider range. Current pose
  marker + small tool-axis triad. Optionally show the history segment highlighted. Start/end sliders also optionally lock
  the plots' x range (toggle).
- Video: `<video>` element; play/pause (Space), speed 0.25–4x. While playing, `requestAnimationFrame` loop reads
  `video.currentTime`, updates cursor/marker/history overlays. While dragging the slider, set `video.currentTime` (throttled
  to one pending seek at a time, using the `seeked` event) and update overlays immediately.
- Keys: Space play/pause; ←/→ step one RGB frame; Shift+←/→ ±1 s; ↑/↓ prev/next episode in the filtered list; P keep,
  X reject, U unflag (Lightroom's P/X/U); N focus note; `[`/`]` set 3D start/end to cursor.
- Episode list: filter by source and flag; shows a per-dataset count of kept/rejected/unflagged.

### History window

Shown only when `--ckpt` is given (user decision); without it the plots show no history overlays. The window comes from the
checkpoint's `shape_meta` (`cfg.task.shape_meta`, the same object the robot runner uses). Reference checkpoint (current DAgger
policy): `/home/boeing/umi_ft_ws/data/training_outputs/stringer_v3.5/hist/dagger_123/0.35/epoch=0080-train_loss=0.016.ckpt`.
Ckpts are ~4.4 GB: load with `torch.load(..., map_location="cpu", mmap=True, weights_only=False)` (or equivalent) so only the
cfg is materialised, and cache the extracted `shape_meta` + sampler settings as JSON keyed by ckpt path + mtime (my call).
Expected values for the hist family (from `docker/train_klone_current.sbatch`; the ckpt is authoritative):

| Modality | Horizon × stride | Rate | Window |
|---|---|---|---|
| RGB | 10 × 5 frames | ~15 Hz | ~3.3 s |
| low-dim pose | 50 × 16 samples | ~194 Hz | ~4.1 s |
| wrench | 64 × 2 samples | ~28 Hz | ~4.6 s |
| action (ahead) | 24 targets on the RGB grid | ~15 Hz | ~1.6 s |

The window at query time t must match what `sampler_umift.py` actually samples (index arithmetic on each stream's own
timestamps, newest sample at or before t). Reuse `sparse_action_indices_from_rgb_timestamps` for the action targets; for obs,
replicate the sampler's index logic in one small function with a test against the real sampler on one episode. Show the
individual sample instants as ticks, not only the span, because sparse sampling (every 16th pose sample) is the point.

Rollout-view toggle (my call): the robot runner stretches obs windows by `slow_down_factor` (2 since c3a88d1; read the value
(the runner config is not in the ckpt, so it is a CLI flag `--rollout-slowdown 2`, default 2). Toggle "training / rollout" switches
the spans between the training window and the stretched one the policy sees at inference. Default: training.

### Never-sampled spans (toggleable, user decision)

Toggle (default on when `--ckpt` is given) that shades in grey the times at which `sampler_umift.py` never creates a training
query: episode start before all obs horizons fit, windows dropped by `action_gap_threshold_mult`, scan-pause overlap, bad-pose
filter (`max_abs_pos_m` etc.), episode end without action padding. Settings come from the checkpoint's cfg
(`cfg.task.dataset`). Preferred implementation: instantiate the real sampler on the one episode and read back its kept query
indices, rather than re-implementing the filters (my call; fall back to replicating only if instantiating it per episode is
impractical). Test: on one real episode, the set of non-grey RGB frames equals the sampler's query ids.

### Export

`review_export.py`: given the flags file and choice (`keep` only, or `keep + unflagged`), call
`merge_umift_zarr_datasets.merge_umift_zarr_datasets(...)` once, base zarrs as `input_zarr_paths`, DAgger zarrs as
`dagger_inputs`, with a new per-input drop argument `drop_episodes_by_input: {zarr_path: [idx, ...]}` (my call: the existing
`drop_episodes` is one global set applied to every input, and episode ids overlap across datasets). Existing CLI/kwargs stay
backward-compatible. The merge file is already modified on the bc branch for DAgger tagging, so this change stays on bc and
gets a work-log entry. Also (my call) copy root attrs (`start_pad_seconds`, `image_size_hw`, `resample_hz`, ...) to the output
when all inputs agree; multi-input merges currently drop them, and `start_pad_seconds` matters to the sampler. Then verify: every output episode has `source` matching the
flag file's source, and episode counts match. Write `review_selection.json` next to the output (flags snapshot + source map +
command). Refuse to overwrite an existing output unless asked. Before writing, check free space on the output filesystem and
refuse if the estimated size exceeds it (disk is 96% full). Note: `merge_umift_zarr_datasets` currently tags base episodes
`original` only when `dagger_inputs` is non-empty and otherwise keeps an existing tag — builder must handle "only base
episodes kept" so tags stay correct.

The export gets a work-log entry (`docs/log/`) when first used for a training dataset (manual, per AGENTS.md).

### Testing

- Unit: payload builder on a synthetic zarr (array lengths, time base, euler unwrap), history-window function vs the real
  sampler, flags round-trip, export on synthetic zarrs (selection, `source` tags, refusal to overwrite).
- Manual: open on the real four datasets, scrub fast through a 1500-frame episode with Chrome DevTools performance panel;
  target: no dropped frames (main-thread work per slider event < 4 ms).

## Left to the builder

- Exact JS module structure, styling, colour choices.
- Downsampling method for plot series (LTTB or min/max bucket).
- Whether the history filmstrip uses a sprite or a second video element.
- Tag-visibility track: include only if the episode-to-sidecar mapping is reliable (it is not in `dagger_2`: 11 unified zarrs
  for 13 episodes).
