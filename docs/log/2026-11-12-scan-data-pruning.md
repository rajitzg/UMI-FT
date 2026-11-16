---
date: 2026-11-12
stream: infra
branch: main
commit: uncommitted
status: done
artifacts: [/mnt/data/boeing/scans, /home/boeing/keyence_scan_ws/scripts/prune_scan_session.py]
---
# Prune scan sessions automatically; clean existing scans

**What changed:** `prune_scan_session.py` (keyence_scan_ws) gained `--keep-final-mesh` mode and now keeps `meta.json`/`compare.log`.
`run_layup_scan.py --prune-after {final-mesh (default), heatmaps, none}` and `run_scan.py` prune at end of a scan.
Periodic scans pass `--prune-after=heatmaps`; post-episode inference eval prunes (heatmaps only) after the background compare in `umift_env_runner.run_compare_pointcloud`.
Manual `scan_*`: keep merged.ply + metadata/metrics. Periodic/inference: keep heatmaps + meta.json/eval_score.json/compare.log only.

**Why:** /mnt/data/boeing/scans was 376 GB, almost all chunk_*.ply, cloud.h5, raw_session.pkl, merged.ply.

**Result / status:** existing scans cut 376 GB -> 55 GB. 45 sessions left untouched (~28 GB) because they lack merged.ply (manual) or distance_heatmap_small.npy (periodic/inference, i.e. failed/missing compare); prune refuses rather than delete the only data. Deleted data is not recoverable (raw_session.pkl gone, so merged.ply cannot be rebuilt with new params for manual scans).

**Repro:** `keyence_scan_ws/.venv/bin/python scripts/prune_scan_session.py [--keep-final-mesh] [--dry-run] <session_dir>`
