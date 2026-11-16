---
date: 2026-11-15
stream: bc
branch: simple_umi_test
commit: 3d7de12 (merge script change uncommitted, see below)
status: done
artifacts: [/mnt/data/umidata/ft/datasets/review_flags/stringer_v3.5.json (created on first flag), ~/.cache/umift_review/ (video/thumb/sprite/ckpt-cfg cache, disposable)]
---
# Episode review tool for base + DAgger data

**What changed:** Added `UMIFT_Data/review/`, a local browser tool. It loads `stringer_v3.5_base` and `_dagger`, `_dagger_2` and `_dagger_3` (95 episodes) and shows:
- the wrist video, synced to plots of position, orientation, force and torque,
- the checkpoint's observation history and action horizon at each frame, in the training view or the approximate rollout view,
- frames the sampler never uses as training queries, from running `SequenceSamplerUmiFT` on each episode,
- the 3D path with start/end sliders.

Episodes are flagged keep/reject, and the kept ones are exported to a new zarr with `attrs["source"]`.

`UMIFT_Data/scripts/merge_umift_zarr_datasets.py` gained a drop list per input (`drop_episodes_by_input`, `--drop-by-input`), copies root attributes when all inputs agree, and has a `write_source_map` flag. Its summary `source_map` is now the list of rows, with the path in `source_map_path`. That file is still **uncommitted**, together with the earlier DAgger-tagging edits in the same file.

**Why:** Adding DAgger data makes the policy worse. The policy switches to the other side early and lifts away at one spot into an out-of-distribution state. The tool is for finding the specific episodes responsible (bad recoveries, noise, poses that don't match the video) and building a cleaned training set.

**Result / status:**
- Tests: 61 review/merge tests pass.
- Real-data smoke test with `--ckpt dagger_123/0.35/epoch=0080`: episode payload 0.04–0.15 s, MP4 build 0.3–1.2 s, a prefetched neighbour episode 0.05 s.
- Headless Chrome: dragging the slider ran at about 57 fps with no long tasks.
- The reference checkpoint queries every 2nd RGB frame (`sparse_query_frequency_down_sample_steps=2`). The first training query in base/0 is at t=6.3 s, inside the 5 s start pad plus the history.
- Not yet used to build a training set.

**Repro:**

```bash
python UMIFT_Data/review/review_server.py --ckpt /home/boeing/umi_ft_ws/data/training_outputs/stringer_v3.5/hist/dagger_123/0.35/epoch=0080-train_loss=0.016.ckpt
```

Export from the UI or with `UMIFT_Data/review/review_export.py`. Spec: `docs/superpowers/specs/2026-11-15-episode-review-tool-design.md`.
