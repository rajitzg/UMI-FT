---
date: 2026-11-09
stream: bc
branch: simple_umi_test
commit: uncommitted (base c3a88d1)
status: in-progress
artifacts: []
---
# Per-source (DAgger) batch weighting

**What changed:** `combine_umift_dataset.sh --dagger-pattern SUBSTR` -> converter tags each episode group `attrs["source"]` = `dagger`/`original`
(by unified-zarr filename substring). `UmiFTDataset(dagger_frac=f)` builds per-sample `sample_weights` (f of each batch from dagger, 1-f original,
train split only); `train_diffusion_unet_image_workspace.py` uses a `WeightedRandomSampler` when weights exist. Default `None` = unchanged behavior.

**Why:** DAgger data should be prioritized less than the original demos, independent of how many DAgger episodes exist.

**Result / status:** not yet run. Existing zarrs have no tag (re-run combine with `--dagger-pattern`). Epoch = len(dataset) draws with replacement.
`train_diffusion_unet_image_finetune_workspace.py` not patched.

**Repro:** `bash UMIFT_Data/bash/combine_umift_dataset.sh <folder> --dagger-pattern dagger`, then train with `task.dataset.dagger_frac=0.2`.

**Addendum:** `merge_umift_zarr_datasets.py --inputs <base> --dagger-inputs <dagger> --output <out>` tags merged episodes `source=original`/`dagger`
(base episodes keep an existing tag). Smoke-tested on synthetic zarrs only.
