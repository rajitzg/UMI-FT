#!/usr/bin/env python3
"""Export flagged episodes to a new acp_replay_buffer.zarr with source tags.

  python UMIFT_Data/review/review_export.py --flags F --output O \\
      [--base Z ...] [--dagger Z ...] [--include keep|keep+unflagged] [--overwrite] [--dry-run]

Uses merge_umift_zarr_datasets (per-input drops). Every output episode gets
attrs["source"] from the review selection (original/dagger); export refuses stale
flags, an empty selection, an existing output (without --overwrite) and outputs
that would not fit on disk. Writes review_selection.json next to the output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import zarr

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from review.episodes import DatasetIndex  # noqa: E402
from review.flags import FlagStore  # noqa: E402

_MERGE_PATH = _HERE.parent / "scripts" / "merge_umift_zarr_datasets.py"
INCLUDE = ("keep", "keep+unflagged")


def _merge_module():
    spec = importlib.util.spec_from_file_location("merge_umift_zarr_datasets", _MERGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def export_selection(index: DatasetIndex, flags: FlagStore, output: Path, include: str = "keep",
                     overwrite: bool = False, dry_run: bool = False,
                     command: Optional[Sequence[str]] = None) -> dict:
    if include not in INCLUDE:
        raise ValueError(f"include must be one of {INCLUDE}")
    output = Path(output).resolve()
    for src in index.dataset_paths.values():
        src = Path(src).resolve()
        folder = src.parent
        if output == src or output in src.parents or src in output.parents or output == folder or folder in output.parents:
            raise ValueError(f"output {output} overlaps source dataset {src}")
    if output.suffix != ".zarr":
        raise ValueError(f"output must be a .zarr path, got {output}")
    wanted = {"keep"} if include == "keep" else {"keep", "none"}

    selected = []
    stale = []
    for ref in index.episodes:
        st = flags.status(ref, index.summary(ref)["fingerprint"])
        if st["flag"] in wanted:
            selected.append(ref)
            if st["stale"]:
                stale.append(ref.key)
    if include == "keep+unflagged":  # a stale reject would silently drop changed content
        stale += [ref.key for ref in index.episodes
                  if ref.key not in stale and flags.status(ref, index.summary(ref)["fingerprint"])["stale"]]
    if stale:
        raise ValueError(f"stale flags (source zarr changed since flagging): {stale}")
    if not selected:
        raise ValueError("nothing selected to export")

    sel_by_path: Dict[str, set] = {}
    for ref in selected:
        sel_by_path.setdefault(ref.zarr_path, set()).add(ref.index)
    sources = {ref.zarr_path: ref.source for ref in index.episodes}
    all_by_path: Dict[str, List[int]] = {}
    for ref in index.episodes:
        all_by_path.setdefault(ref.zarr_path, []).append(ref.index)
    base_in = [p for p in sel_by_path if sources[p] == "original"]
    dagger_in = [p for p in sel_by_path if sources[p] == "dagger"]
    drops = {p: sorted(set(all_by_path[p]) - sel_by_path[p]) for p in sel_by_path}
    expected = {(p, f"episode_{i}"): sources[p] for p in sel_by_path for i in sel_by_path[p]}

    summary = {"output": str(output), "n_episodes": len(selected),
               "n_original": sum(r.source == "original" for r in selected),
               "n_dagger": sum(r.source == "dagger" for r in selected),
               "selection_json": str(output.parent / "review_selection.json")}
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output} (pass overwrite)")
    need = sum(_dir_size(Path(r.zarr_path) / "data" / f"episode_{r.index}") for r in selected)
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    summary["estimated_bytes"] = need
    if need > 0.9 * free:
        raise OSError(f"export needs ~{need / 1e9:.1f} GB, only {free / 1e9:.1f} GB free at {probe}")
    if dry_run:
        summary["dry_run"] = True
        return summary

    merge = _merge_module()
    res = merge.merge_umift_zarr_datasets(
        base_in, output, overwrite=overwrite, dagger_inputs=dagger_in,
        drop_episodes_by_input=drops, write_source_map=False)

    root = zarr.open(str(output), mode="r+")
    rows = []
    try:
        for row in res["source_map"]:
            key = (str(Path(row["source_zarr"]).resolve()), row["source_episode"])
            want = expected.get(key)
            ep = root["data"][row["out_episode"]]
            got = ep.attrs.get("source")
            if want is None or (got is not None and got != want):
                raise RuntimeError(f"source tag mismatch for {row}: got {got!r}, expected {want!r}")
            ep.attrs["source"] = want
            rows.append(dict(row, source=want))
        if len(rows) != len(selected):
            raise RuntimeError(f"exported {len(rows)} episodes, selected {len(selected)}")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    keys = {r.key for r in selected}
    Path(summary["selection_json"]).write_text(json.dumps({
        "include": include,
        "flags_file": str(flags.path),
        "flags": {k: v for k, v in flags.entries.items() if k in keys},
        "source_map": rows,
        "command": list(command) if command else None,
    }, indent=1) + "\n")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    from review.review_server import DEFAULT_BASE, DEFAULT_DAGGER
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", nargs="*", default=DEFAULT_BASE)
    p.add_argument("--dagger", nargs="*", default=DEFAULT_DAGGER)
    p.add_argument("--flags", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--include", choices=INCLUDE, default="keep")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    fs = FlagStore(Path(a.flags))
    fs.load()
    s = export_selection(DatasetIndex(a.base, a.dagger), fs, Path(a.output), a.include,
                         a.overwrite, a.dry_run, command=sys.argv)
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
