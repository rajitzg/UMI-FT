# Episode review tool

A browser tool for reviewing the episodes of several UMIFT zarrs at once, such as `stringer_v3.5` base plus the DAgger sets. You can:

- scrub each episode with the wrist video synced to its plots,
- see what the policy observes at each frame,
- flag each episode keep or reject,
- export the kept episodes to a new replay buffer with `source` tags.

Design: `docs/superpowers/specs/2026-11-15-episode-review-tool-design.md`.

## Run

```bash
conda activate pyrite
source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-standard_bc
cd /home/boeing/umi_ft_ws/worktrees/UMI-FT-standard_bc
python UMIFT_Data/review/review_server.py \
  --ckpt /home/boeing/umi_ft_ws/data/training_outputs/stringer_v3.5/hist/dagger_123/0.35/epoch=0080-train_loss=0.016.ckpt
# open http://127.0.0.1:8770  (over SSH: forward the port, e.g. VS Code "Ports", and use a local browser)
```

Options:

- `--base Z ...` / `--dagger Z ...`: the zarrs to load. The default is the `stringer_v3.5_base` zarr as base, and `_dagger`, `_dagger_2` and `_dagger_3` as DAgger.
  - The source tag comes from these lists, not from episode attributes.
- `--ckpt CKPT`: turns on the policy history window and the never-sampled band.
  - It reads `shape_meta` and the dataset/sampler settings from the checkpoint. Loading takes about 1 s with mmap, and the result is cached.
  - Without `--ckpt`, the plots show no history overlays.
- `--rollout-slowdown 2`: the runner's `slow_down_factor`, used by the "rollout (approx.)" history view.
- `--session NAME` / `--flags FILE`: where flags are stored. The default is `/mnt/data/umidata/ft/datasets/review_flags/<session>.json`.
- `--cache-dir`: where MP4s, thumbnails and sprites are cached. The default is `~/.cache/umift_review`. It is safe to delete.

## Screen

The screen has these parts:

- **Left:** the episode list. Filters cover source, flag and stale. There are per-dataset counts and an Export button.
- **Video:** the wrist RGB, with a timeline strip below it.
  - The strip shows the 3D range, the never-sampled frames, the RGB observation ticks (blue) and the action horizon (orange).
  - With `--ckpt`, the filmstrip below the strip shows the 10 RGB frames the policy actually sees at the cursor.
- **3D path:** shown in the fixture frame, with +Y down and the camera on the operator side.
  - The *3D start* and *3D end* sliders limit how much of the path is drawn.
  - *lock plots to 3D range* zooms the plots to the same window.
  - The path can be coloured by time or by |F|. The current pose has an axis triad, and the pose-observation history is drawn as blue points.
- **Plots:** position, orientation, force with |F|, and a collapsible group with torque, speed, VT offset and stiffness.
  - **Orientation** is euler xyz in degrees, relative to one fixed reference rotation (the mean orientation of the first episode). Absolute euler angles sit near gimbal lock for this tool.
  - **Shading:** blue is the pose observation window, red is the wrench observation window, orange is the action horizon ahead, grey is never sampled, light grey is the start pad, purple is a scan gap, and magenta lines are pose jumps (more than 5 mm between consecutive samples).
  - **Training vs. rollout view:** the *training* view uses the exact sampler index math. *Rollout (approx.)* stretches the windows by the slowdown factor.
  - **Never sampled:** grey marks frames that `SequenceSamplerUmiFT` never uses as a training query: the episode start before the history fits, action gaps, scan gaps, bad poses and the end. It comes from running the real sampler on that episode. Query ids are spread across the query stride, so the stride does not show up as holes.

## Keys

| Key | Action |
|---|---|
| Space | play / pause |
| ← / → | step one RGB frame (Shift: 1 s) |
| ↑ / ↓ | previous / next episode in the filtered list |
| P / X / U | keep / reject / unflag |
| N | edit the note (Enter or Esc to leave) |
| [ / ] | set the 3D start / end to the cursor |

On a plot, drag to zoom, double-click to reset, and click to seek. The *auto-advance after flag* checkbox moves to the next episode after P or X.

## Flags and stale detection

Flags are saved on every change, with an atomic replace. Each flag stores a fingerprint of its episode: the lengths of each stream plus the first and last robot timestamps.

- If a source zarr is rebuilt or trimmed in place, the flag becomes **stale**. It is shown orange, and export refuses it.
- Flags for episodes that no longer exist are listed as orphans under the counts.

## Export

Export is available from the dialog, or from the command line:

```bash
python UMIFT_Data/review/review_export.py --flags /mnt/data/umidata/ft/datasets/review_flags/stringer_v3.5.json \
  --output /mnt/data/umidata/ft/datasets/stringer_v3.5_reviewed/acp_replay_buffer.zarr [--include keep|keep+unflagged] [--dry-run]
```

Export runs `UMIFT_Data/scripts/merge_umift_zarr_datasets.py`, which now has a drop list per input (`drop_episodes_by_input`, CLI `--drop-by-input PATH:IDX,...`) and keeps root attributes that all inputs agree on (e.g. `start_pad_seconds`).

- Every output episode gets `attrs["source"]` set to `original` or `dagger`. This works with `task.dataset.dagger_frac`.
- `review_selection.json` is written next to the output. It holds the selected flags, the source map and the command.
- Export refuses:
  - stale flags (with *keep + unflagged*, any stale flag, including stale rejects);
  - an empty selection;
  - an output that is not a `.zarr` path, or that equals, contains or lies inside a source dataset or its folder;
  - an existing output, unless you tick overwrite;
  - an estimated size above 90% of free disk;
  - a second export while one is running.
- The server accepts only same-origin `application/json` POSTs.
- Flag keys are ignored while a flag is being saved or the next episode is loading, so a quick second key press cannot land on the wrong episode.
- A note-only edit does not clear a stale flag; only a new P/X/U decision does.
- Add a work-log entry (`docs/log/`) when an export is used to train.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest UMIFT_Data/tests/test_review_*.py UMIFT_Data/tests/test_merge_umift_zarr_datasets.py -q
```

The tests marked `slow` use the real reference checkpoint and dataset, and skip when they are absent. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids the ROS pytest plugins on this machine.

Vendored front-end libraries are in `static/vendor/`: uPlot 1.6.31 and three.js r170 with `OrbitControls`, whose import was rewritten to `./three.module.min.js`.
