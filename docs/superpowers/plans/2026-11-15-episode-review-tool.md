# Episode Review Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local browser tool to scrub every episode of base + DAgger `stringer_v3.5` zarrs with synced wrist video, pose/force plots, the checkpoint's observation-history window and never-sampled spans, flag episodes keep/reject, and export the kept ones to a new replay buffer with correct `source` tags.

**Architecture:** Python stdlib HTTP server (`UMIFT_Data/review/`) opens zarrs read-only and serves per-episode binary payloads (float32 arrays + JSON header), cached all-intra H.264 MP4s with HTTP Range, and flag/export endpoints. All interaction is client-side: `<video>` seeking, uPlot charts with a canvas overlay for cursor/history (no series redraw), three.js path. History ticks and never-sampled spans are precomputed per RGB frame on the server from the checkpoint's `shape_meta`, so the client only indexes arrays.

**Tech Stack:** Python 3 (`pyrite` env: zarr v2, numpy, scipy, av, torch, dill), `/usr/bin/ffmpeg`, vanilla JS ES modules, vendored uPlot 1.6.x and three.js r16x (+ `OrbitControls`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-episode-review-tool-design.md` (read it; this plan does not repeat its rationale).

## Global Constraints

- All work in `/home/boeing/umi_ft_ws/worktrees/UMI-FT-standard_bc`. Every shell: `conda activate pyrite && source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-standard_bc`; check `python -c "import PyriteConfig; print(PyriteConfig.__file__)"` points into this worktree.
- Source zarrs are opened `mode="r"` only. The tool writes only: flags file, `~/.cache/umift_review/`, export output.
- Default datasets: base `/mnt/data/umidata/ft/datasets/stringer_v3.5_base/acp_replay_buffer.zarr`; dagger `.../stringer_v3.5_dagger/`, `.../stringer_v3.5_dagger_2/`, `.../stringer_v3.5_dagger_3/` `acp_replay_buffer.zarr`. Dataset display name = parent folder name minus `stringer_v3.5_` prefix (`base`, `dagger`, `dagger_2`, `dagger_3`). Episode key = `"<name>/<index>"`, shown `dagger_2/ep7`.
- Source tags come from `--base` / `--dagger` CLI lists. If an episode attr `source` exists and disagrees, log a warning and use the CLI.
- Timestamps in the zarrs are seconds (rgb starts at 0.0). Time base everywhere = raw episode seconds, no re-zeroing.
- pose7 = `[x,y,z,qw,qx,qy,qz]`; convert with `PyriteUtility.spatial_math.spatial_utilities` helpers or reindex explicitly for scipy (`[x,y,z,w]`).
- Flags default path `/mnt/data/umidata/ft/datasets/review_flags/<session>.json`, `--session` default `stringer_v3.5`, `--flags` overrides. Cache dir `~/.cache/umift_review/`. Default port 8770, bind `127.0.0.1`.
- Keys: Space play/pause; ←/→ ±1 RGB frame; Shift+←/→ ±1 s; ↑/↓ prev/next episode in filtered list; P keep, X reject, U unflag; N focus note; `[` / `]` set 3D start/end to cursor.
- History overlays and never-sampled band appear only with `--ckpt`. `--rollout-slowdown` default 2. Reference ckpt: `/home/boeing/umi_ft_ws/data/training_outputs/stringer_v3.5/hist/dagger_123/0.35/epoch=0080-train_loss=0.016.ckpt`.
- **Commits:** the worktree has the user's uncommitted work, including `UMIFT_Data/scripts/merge_umift_zarr_datasets.py` and its test. `git add` only files the task names. Never commit `merge_umift_zarr_datasets.py` or `UMIFT_Data/tests/test_merge_umift_zarr_datasets.py` (Task 1 leaves them modified, uncommitted; report this). Never `git add -A`, never stash.
- `/mnt/data` is ~96% full: tests write only to `tmp_path`.

## Review Focus

1. Episode with RGB but very short robot stream, or a stream ending before the last RGB frame: payload/window code must clamp indices, not raise. Test in Task 3 (`test_window_ticks_clamp_at_stream_end`).
2. Base-only export (no DAgger episode kept) must still tag every output episode `source=original`. Test in Task 1 (`test_drop_by_input_base_only_tags_original`).
3. Source zarr trimmed after flagging (`dagger_2` went 17 -> 13): flag for an index that no longer exists or whose fingerprint changed is stale, not silently re-applied. Test in Task 4 (`test_stale_when_episode_missing`, `test_stale_when_fingerprint_changes`).
4. Fast slider drag queues seeks: the client must keep at most one pending `video.currentTime` seek (latest wins). Manual check in Task 8 step list.
5. Browser `<video>` seeking needs `206 Partial Content` with correct `Content-Range`; a plain 200 makes seeking jump to 0 in Chrome. Test in Task 7 (`test_video_range_request`).

---

## File structure

```
UMIFT_Data/review/
  __init__.py
  episodes.py         dataset index, fingerprints, per-episode payload builder + binary encoding
  video.py            MP4 + thumbnail cache
  policy_window.py    ckpt shape_meta extraction/cache, history ticks, never-sampled mask
  flags.py            flag store (load/save/stale)
  review_export.py    export kept episodes (module + CLI)
  review_server.py    CLI + HTTP server
  static/index.html, static/style.css, static/app.js, static/plots.js, static/path3d.js
  static/vendor/uPlot.esm.js, uPlot.min.css, three.module.min.js, OrbitControls.js
  README.md
UMIFT_Data/tests/
  review_fixtures.py  synthetic zarr writer shared by review tests
  test_review_episodes.py, test_review_video.py, test_review_policy_window.py,
  test_review_flags.py, test_review_export.py, test_review_server.py
UMIFT_Data/scripts/merge_umift_zarr_datasets.py   (modify)
UMIFT_Data/tests/test_merge_umift_zarr_datasets.py (modify)
docs/log/2026-11-15-episode-review-tool.md
```

Tests run from the worktree root: `python -m pytest UMIFT_Data/tests/<file> -v`. Review tests import with `sys.path.insert(0, <worktree>/UMIFT_Data)` then `from review import episodes` (put this in `review_fixtures.py`).

---

### Task 1: Per-input drops and root-attr preservation in the merge

**Files:**
- Modify: `UMIFT_Data/scripts/merge_umift_zarr_datasets.py` (`merge_umift_zarr_datasets`, ~lines 109-240; CLI ~line 320)
- Test: `UMIFT_Data/tests/test_merge_umift_zarr_datasets.py` (reuse `_write_toy_zarr`)

**Interfaces:**
- Produces: `merge_umift_zarr_datasets(input_zarr_paths, output_zarr_path, *, overwrite=False, dry_run=False, drop_episodes=None, dagger_inputs=(), drop_episodes_by_input: Optional[Mapping[str | Path, Sequence[int]]] = None, write_source_map: bool = True) -> Dict[str, object]`. Summary dict additionally carries `"source_map": List[dict]` (same rows as `episode_source_map.json`). `write_source_map=False` skips writing `episode_source_map.json` (export writes its own).

- [ ] **Step 1: Write failing tests**

```python
def test_drop_by_input_only_affects_that_input(tmp_path):
    a, b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    _write_toy_zarr(a, 3, 10); _write_toy_zarr(b, 3, 20)
    out = tmp_path / "out.zarr"
    s = MODULE.merge_umift_zarr_datasets([a], out, dagger_inputs=[b],
                                          drop_episodes_by_input={str(b): [1]})
    root = zarr.open(str(out), "r")
    srcs = [(r["source_zarr"], r["source_episode"]) for r in s["source_map"]]
    assert (str(a.resolve()), "episode_1") in srcs
    assert (str(b.resolve()), "episode_1") not in srcs
    assert len(srcs) == 5
    assert [root["data"][f"episode_{i}"].attrs["source"] for i in range(5)] == ["original"] * 3 + ["dagger"] * 2

def test_drop_by_input_base_only_tags_original(tmp_path):
    a, b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    _write_toy_zarr(a, 2, 10); _write_toy_zarr(b, 2, 20)
    out = tmp_path / "out.zarr"
    MODULE.merge_umift_zarr_datasets([a], out, dagger_inputs=[b],
                                     drop_episodes_by_input={str(b): [0, 1]})
    root = zarr.open(str(out), "r")
    assert [root["data"][f"episode_{i}"].attrs["source"] for i in range(2)] == ["original", "original"]

def test_drop_by_input_unknown_index_raises(tmp_path): ...  # {str(a): [7]} on 3-episode zarr -> KeyError
def test_root_attrs_kept_when_inputs_agree(tmp_path): ...   # both inputs attrs start_pad_seconds=5.0 -> out root has 5.0
def test_root_attrs_dropped_when_inputs_disagree(tmp_path): ...  # 5.0 vs 0.0 -> key absent, warning printed
def test_existing_global_drop_unchanged(tmp_path): ...      # existing drop_episodes behaviour still passes
```

An input whose every episode is dropped is skipped (no error). A fully dropped dagger input contributes no episodes, so the base-only test expects 2 outputs.

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest UMIFT_Data/tests/test_merge_umift_zarr_datasets.py -v`
Expected: new tests FAIL (`unexpected keyword argument 'drop_episodes_by_input'`); existing tests pass.

- [ ] **Step 3: Implement**

Keys of `drop_episodes_by_input` resolved with `Path(k).resolve()`; each must be one of the inputs (else `ValueError`). Per-input drop set = global `drop_set` ∪ that input's set; missing indices raise `KeyError` naming the input. Base episodes are tagged `source` whenever `dagger_inputs` was passed (even if all dagger episodes are dropped). Root attrs: for multi-input, copy every key whose value is equal across all inputs (JSON-compare); print `[merge] WARNING root attr <k> differs across inputs; dropped`. Add CLI flag `--drop-by-input PATH:IDX[,IDX...]` (repeatable).

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest UMIFT_Data/tests/test_merge_umift_zarr_datasets.py -v` → all PASS.

- [ ] **Step 5: Do not commit** (file holds the user's uncommitted DAgger-tagging work). Note in the task report: "merge script + test modified, uncommitted".

---

### Task 2: Dataset index, fingerprints, episode payload

**Files:**
- Create: `UMIFT_Data/review/__init__.py`, `UMIFT_Data/review/episodes.py`, `UMIFT_Data/tests/review_fixtures.py`
- Test: `UMIFT_Data/tests/test_review_episodes.py`

**Interfaces:**
- Produces (in `episodes.py`):
  - `@dataclass(frozen=True) class EpisodeRef: dataset: str; index: int; zarr_path: str; source: str  # "original"|"dagger"`; property `key -> str` = `f"{dataset}/{index}"`.
  - `class DatasetIndex: def __init__(self, base: list[str], dagger: list[str])`; `.episodes -> list[EpisodeRef]` (base datasets first, then dagger, numeric episode order); `.get(key: str) -> EpisodeRef`; `.group(ref) -> zarr.Group` (cached read-only handle); `.summary(ref) -> dict` with `key, dataset, index, source, duration_s, n_rgb, fingerprint`.
  - `dataset_display_name(zarr_path: str) -> str`.
  - `fingerprint(group: zarr.Group) -> dict` = `{"n_rgb", "n_robot", "n_wrench", "t_robot_first", "t_robot_last"}` (floats rounded to 1e-6).
  - `build_payload(group: zarr.Group, extra: dict | None = None) -> tuple[dict, list[np.ndarray]]` (header, float32 arrays).
  - `encode_payload(header: dict, arrays: list[np.ndarray]) -> bytes`.
- `review_fixtures.py`: `write_synthetic_episode_zarr(path, n_eps=2, n_rgb=60, robot_hz=200.0, rgb_hz=15.0, wrench_hz=28.0, start_pad_s=0.0, scan_gaps=None) -> Path` with all keys from the spec's "Per episode" list plus `meta/episode_*_len` arrays matching `merge_umift_zarr_datasets.META_LEN_KEYS`, poses a smooth arc, and `sys.path` setup so `from review import ...` works.

Payload layout (binary): `uint32 LE header_len` | header JSON UTF-8 | zero pad to 4-byte boundary | arrays as float32 LE, concatenated in `header["arrays"]` order. Header:
```
{"arrays": [{"name": str, "length": int}], "duration_s": float, "start_pad_s": float,
 "markers": {"scan_gaps": [[t0,t1],...], "pose_jumps": [t,...]}, "fps": float, ...extra}
```
Array names (all from the episode): `rgb_t`; `pose_t, px, py, pz, qw, qx, qy, qz, rx, ry, rz, speed, stiffness, vt_off`; `wrench_t, fx, fy, fz, tx, ty, tz, fmag`. `rx..rz` = euler xyz degrees from `Rotation.as_euler("xyz")`, `np.unwrap` per axis in radians before converting. `speed` = |Δp|/Δt, then centered moving average of 25 samples. `vt_off` = ‖p_vt − p_fb‖ (m). `pose_jumps` = times where consecutive-sample step > 0.005 m. No tag-visibility track in v1 (episode→sidecar mapping is unreliable in `dagger_2`). No downsampling unless the pose stream exceeds 20000 samples; then take every k-th sample with k = ceil(n/20000), keeping the last.

- [ ] **Step 1: Write failing tests**

```python
def test_index_orders_and_tags(tmp_path):
    b = write_synthetic_episode_zarr(tmp_path / "stringer_v3.5_base" / "acp_replay_buffer.zarr", n_eps=2)
    d = write_synthetic_episode_zarr(tmp_path / "stringer_v3.5_dagger_2" / "acp_replay_buffer.zarr", n_eps=3)
    idx = episodes.DatasetIndex([str(b)], [str(d)])
    assert [e.key for e in idx.episodes] == ["base/0", "base/1", "dagger_2/0", "dagger_2/1", "dagger_2/2"]
    assert [e.source for e in idx.episodes] == ["original"] * 2 + ["dagger"] * 3

def test_payload_roundtrip_lengths(tmp_path):   # decode encode_payload output; each array length == header length; pose_t == robot_time_stamps_0
def test_euler_unwrapped(tmp_path):             # pose yaw sweeping through ±180 deg -> max |diff(rz)| < 10
def test_pose_jump_marker(tmp_path):            # inject 1 cm step at sample 100 -> exactly one jump at pose_t[100]
def test_fingerprint_changes_with_length(tmp_path)
def test_source_attr_disagreement_warns(tmp_path, capsys)  # episode attr source="dagger" in a --base zarr -> ref.source=="original", warning printed
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest UMIFT_Data/tests/test_review_episodes.py -v` → FAIL (`No module named 'review'` / missing names).
- [ ] **Step 3: Implement** `episodes.py` and `review_fixtures.py` per Interfaces.
- [ ] **Step 4: Run, verify pass.** Also a smoke run on real data: `python -c "from review.episodes import *; i=DatasetIndex(['/mnt/data/umidata/ft/datasets/stringer_v3.5_base/acp_replay_buffer.zarr'],[]); h,a=build_payload(i.group(i.episodes[0])); print(len(encode_payload(h,a))/1e6,'MB')"` (from `UMIFT_Data/`) prints < 5 MB.
- [ ] **Step 5: Commit** `git add UMIFT_Data/review/__init__.py UMIFT_Data/review/episodes.py UMIFT_Data/tests/review_fixtures.py UMIFT_Data/tests/test_review_episodes.py && git commit -m "feat(review): dataset index and episode payload"`

---

### Task 3: Policy window from checkpoint

**Files:**
- Create: `UMIFT_Data/review/policy_window.py`
- Test: `UMIFT_Data/tests/test_review_policy_window.py`

**Interfaces:**
- Consumes: `review_fixtures.write_synthetic_episode_zarr`.
- Produces:
  - `@dataclass class WindowSpec: rgb: tuple[int,int]; low_dim: tuple[int,int]; wrench: tuple[int,int]; action: tuple[int,int]; sampler_kwargs: dict; shape_meta: dict` — each tuple = `(horizon, down_sample_steps)`; `action` stride is in RGB frames.
  - `load_window_spec(ckpt_path: str, cache_dir: Path) -> WindowSpec`. Reads `payload["cfg"]` via `torch.load(open(p,"rb"), map_location="cpu", pickle_module=dill, mmap=True, weights_only=False)`; if mmap load raises, fall back to a plain load and print a warning. `rgb` from `shape_meta.sample.obs.sparse.rgb_0`, `low_dim` from `robot0_eef_pos` (first non-rgb/non-wrench low-dim key if absent), `wrench` from `robot0_eef_wrench_left` (first key containing `wrench`), `action` from `shape_meta.sample.action.sparse`. `sampler_kwargs` = the `cfg.task.dataset` keys that `SequenceSamplerUmiFT.__init__` accepts. Cache: `cache_dir / f"window_{sha1(path+mtime)}.json"`.
  - `window_ticks(times: np.ndarray, query_times: np.ndarray, horizon: int, ds: int) -> np.ndarray` shape `(len(query_times), horizon)` float32 of sample times, oldest first, padded at the front with the first sample (sampler semantics: `query_id = searchsorted(times, t)` clipped to `len-1`; `num_valid = min(h, query_id//ds + 1)`; indices `query_id - k*ds`).
  - `action_ticks(action_times, rgb_times, horizon, rgb_stride) -> np.ndarray` `(n_rgb, horizon)` float32 via `sampler_umift.sparse_action_indices_from_rgb_timestamps`, right-padded with the last valid time.
  - `episode_window_arrays(group, spec: WindowSpec) -> dict[str, np.ndarray]` → names `win_rgb`, `win_pose`, `win_wrench`, `win_action` (each flattened `(n_rgb*horizon,)`), plus `sampled` (`(n_rgb,)` 1.0/0.0). Header extra: `{"window": {"rgb": h, "pose": h, "wrench": h, "action": h}}`.
  - `sampled_mask(group, spec) -> np.ndarray[bool]` `(n_rgb,)`: build a one-episode replay buffer (`{"data": {"episode_0": converted}, "meta": {...}}` using `UmiFTDataset.raw_episodes_conversion` logic on an in-memory copy of only the needed keys), instantiate `SequenceSamplerUmiFT(shape_meta, rb, obs_to_obs_sample=..., action_to_action_sample=..., id_list=[0], **sampler_kwargs)`, return mask of `rgb_id` in `sampler.indices`. If this cannot be done without loading RGB, pass a zero-size-image stand-in for `rgb_0`; document the choice in the module docstring.

- [ ] **Step 1: Write failing tests**

```python
def test_window_ticks_matches_sampler_semantics():
    t = np.arange(1000) * 0.005
    ticks = policy_window.window_ticks(t, np.array([0.0, 2.0, 4.999]), horizon=50, ds=16)
    assert ticks.shape == (3, 50)
    assert np.all(ticks[0] == 0.0)                               # front-padded
    qi = np.searchsorted(t, 2.0)
    assert ticks[1, -1] == np.float32(t[qi]) and ticks[1, -2] == np.float32(t[qi - 16])

def test_window_ticks_clamp_at_stream_end():
    t = np.arange(100) * 0.005
    ticks = policy_window.window_ticks(t, np.array([10.0]), horizon=4, ds=2)
    assert ticks[0, -1] == np.float32(t[-1])

def test_action_ticks_on_rgb_grid(): ...       # rgb 15 Hz, horizon 24, stride 1 -> ticks[i] == rgb times i..i+23 (padded past end)

@pytest.mark.skipif(not REF_CKPT.exists(), reason="reference ckpt not present")
def test_load_window_spec_reference_ckpt(tmp_path):
    s = policy_window.load_window_spec(str(REF_CKPT), tmp_path)
    assert s.rgb == (10, 5) and s.low_dim == (50, 16) and s.wrench == (64, 2) and s.action[0] == 24

@pytest.mark.skipif(not (REF_CKPT.exists() and BASE_ZARR.exists()), reason="real data absent")
def test_sampled_mask_equals_real_sampler(tmp_path):
    # episode 0 of base zarr: build UmiFTDataset-equivalent sampler over the whole zarr with
    # episode_mask selecting only episode 0, compare its rgb ids to sampled_mask(...)
```

Mark the two real-data tests `@pytest.mark.slow`; register the marker in `UMIFT_Data/tests/conftest.py` only if one does not exist (check first; create a minimal one if needed and add it to this task's commit).

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run, verify pass**, including slow tests: `python -m pytest UMIFT_Data/tests/test_review_policy_window.py -v` (real ckpt load should take < 60 s first time, < 1 s cached; report the time).
- [ ] **Step 5: Commit** the two files (+ conftest if created).

---

### Task 4: Flag store

**Files:**
- Create: `UMIFT_Data/review/flags.py`
- Test: `UMIFT_Data/tests/test_review_flags.py`

**Interfaces:**
- Consumes: `EpisodeRef`, `DatasetIndex.summary`, `fingerprint`.
- Produces: `class FlagStore: def __init__(self, path: Path)`; `.load() -> None`; `.set(ref: EpisodeRef, fp: dict, flag: str | None = None, note: str | None = None) -> dict` (flag in `{"keep","reject","none"}`, else `ValueError`; updates `updated` ISO time; saves atomically via temp file + `os.replace`); `.entries -> dict[str, dict]` keyed by `ref.key`; `.status(ref, fp) -> dict` = entry + `"stale": bool`. File JSON: `{"version": 1, "episodes": {key: {"dataset_path","episode","source","flag","note","updated","fingerprint"}}}`. Stale when the stored fingerprint differs from the current one; entries whose episode no longer exists in the index are reported by `.orphans(index) -> list[str]`.

- [ ] **Step 1: Write failing tests**: `test_set_and_reload`, `test_invalid_flag_raises`, `test_atomic_save_leaves_no_tmp`, `test_stale_when_fingerprint_changes`, `test_stale_when_episode_missing` (orphan listed after the zarr is rewritten with fewer episodes).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** `flags.py` + test.

---

### Task 5: Video cache

**Files:**
- Create: `UMIFT_Data/review/video.py`
- Test: `UMIFT_Data/tests/test_review_video.py`

**Interfaces:**
- Produces: `episode_video(group: zarr.Group, cache_dir: Path, cache_key: str) -> Path`, `episode_thumb(group, cache_dir, cache_key) -> Path`, and `episode_sprite(group, cache_dir, cache_key) -> tuple[Path, dict]` (JPEG sprite of every RGB frame at 56×56, row-major, `cols = ceil(sqrt(n))`; dict `{"tile": 56, "cols": cols, "n": n}` — used for the RGB-history filmstrip so the client never seeks a second video). `cache_key` = `sha1(zarr_path + episode + str(fingerprint))`. MP4: H.264 via ffmpeg stdin rawvideo pipe (`-f rawvideo -pix_fmt rgb24 -s 224x224 -r <fps> -i - -c:v libx264 -g 1 -crf 18 -preset veryfast -pix_fmt yuv420p -movflags +faststart`), fps = 1/median(diff(rgb_time_stamps_0)). Read `rgb_0` in 208-frame chunks. Write to `*.tmp` then rename (concurrent requests must not see partial files; guard with a per-key `threading.Lock`). Thumb: JPEG 112x112 of the middle frame.

- [ ] **Step 1: Failing tests**: `test_video_frame_count` (synthetic 60 frames → PyAV decodes 60 frames), `test_video_cached_second_call` (mtime unchanged), `test_thumb_size`, `test_sprite_grid` (60 frames → cols 8, image 448×448).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, verify pass.** Smoke on a real 829-frame episode: build time and file size printed (expect a few seconds, < 15 MB).
- [ ] **Step 5: Commit.**

---

### Task 6: Export

**Files:**
- Create: `UMIFT_Data/review/review_export.py`
- Test: `UMIFT_Data/tests/test_review_export.py`

**Interfaces:**
- Consumes: `DatasetIndex`, `FlagStore`, `merge_umift_zarr_datasets(..., drop_episodes_by_input=..., write_source_map=False)` (import via `importlib` from `UMIFT_Data/scripts/` as the merge test does).
- Produces: `export_selection(index: DatasetIndex, flags: FlagStore, output: Path, include: str = "keep", overwrite: bool = False, dry_run: bool = False) -> dict` (`include` in `{"keep","keep+unflagged"}`). Returns `{"output", "n_episodes", "n_original", "n_dagger", "selection_json"}`. CLI: `python UMIFT_Data/review/review_export.py --base Z... --dagger Z... --flags F --output O [--include keep|keep+unflagged] [--overwrite] [--dry-run]`.

Rules: raise `ValueError` if any selected episode is stale or nothing is selected; raise `FileExistsError` if output exists and not `overwrite`; estimate size = sum of `du` of selected episode dirs (via `os.scandir` walk) and raise `OSError` if > 0.9 × `shutil.disk_usage(output.parent).free`. Inputs with zero selected episodes are omitted from the merge call. After merge, re-open output and assert per episode `attrs["source"]` equals the flag store's source for its `(source_zarr, source_episode)`; on mismatch delete output and raise. Write `output.parent / "review_selection.json"` with flags snapshot, `source_map`, `include`, and the CLI command line.

- [ ] **Step 1: Failing tests**: `test_export_keep_only_tags` (base keep 1, dagger keep 2 → 3 episodes, sources `["original","dagger","dagger"]`), `test_export_keep_plus_unflagged`, `test_export_refuses_stale`, `test_export_refuses_existing_output`, `test_export_base_only` (all `original`), `test_selection_json_written`.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** `review_export.py` + test.

---

### Task 7: HTTP server

**Files:**
- Create: `UMIFT_Data/review/review_server.py`
- Test: `UMIFT_Data/tests/test_review_server.py`

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces: `make_server(args: argparse.Namespace) -> ThreadingHTTPServer` (testable without `serve_forever`), `main()`. CLI per spec: `--base`, `--dagger` (defaults from Global Constraints), `--ckpt`, `--rollout-slowdown 2`, `--session stringer_v3.5`, `--flags`, `--port 8770`, `--cache-dir ~/.cache/umift_review`.

Routes:
- `GET /` and `/static/*` → files from `static/` (correct MIME for `.js` = `text/javascript`).
- `GET /api/episodes` → `{"episodes": [summary + flag status...], "orphans": [...], "has_ckpt": bool, "rollout_slowdown": float, "window": {...} | null}`.
- `GET /api/episode/<dataset>/<index>/data` → `application/octet-stream`, `build_payload` + (if ckpt) `episode_window_arrays` merged into the same payload. Cache encoded bytes in an LRU of 6 episodes.
- `GET /api/episode/<dataset>/<index>/video.mp4` → `video/mp4`, supports `Range: bytes=a-b` (206 + `Content-Range` + `Accept-Ranges: bytes`), plain GET → 200.
- `GET /api/episode/<dataset>/<index>/thumb.jpg`, `.../sprite.jpg`; sprite geometry is added to the `data` header as `"sprite": {...}`.
- `POST /api/flag` body `{"key", "flag"?, "note"?}` → updated status.
- `POST /api/export` body `{"output", "include", "overwrite"?, "dry_run"?}` → export summary, or `{"error": str}` with 400.
- After serving `data` for episode k, submit background builds (a 2-worker `ThreadPoolExecutor`) of video + payload for the neighbours k−1, k+1 in `index.episodes` order.
- Unknown key → 404 JSON.

- [ ] **Step 1: Failing tests** (server on port 0 in a thread, synthetic zarrs, no ckpt): `test_episodes_endpoint`, `test_data_endpoint_decodes`, `test_video_range_request` (`Range: bytes=0-99` → 206, `Content-Range: bytes 0-99/<size>`, 100 bytes), `test_flag_post_persists`, `test_export_post_error_is_400`, `test_unknown_episode_404`.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, verify pass.** Then `python UMIFT_Data/review/review_server.py --ckpt <reference ckpt>` starts and `curl -s 127.0.0.1:8770/api/episodes | head -c 300` lists 95 episodes.
- [ ] **Step 5: Commit.**

---

### Task 8: Browser client

**Files:**
- Create: `UMIFT_Data/review/static/{index.html, style.css, app.js, plots.js, path3d.js}`, `UMIFT_Data/review/static/vendor/*`
- Test: manual (below); no JS test harness.

**Interfaces:**
- Consumes: Task 7 routes and the payload layout from Task 2/3.
- `app.js`: `decodePayload(ArrayBuffer) -> {header, arrays: {name: Float32Array}}` (zero-copy `Float32Array` views at the computed offsets); state `{episodes, filtered, current, frame, playing, range3d: [t0,t1], historyMode: "training"|"rollout", showUnsampled, colorBy: "time"|"force"}`.
- `plots.js`: `createPlots(container, arrays, header) -> {setCursor(t, frame), setXRange(t0, t1), destroy()}`. Charts: position xyz, orientation xyz (deg), force xyz + |F|, and a collapsible chart with torque xyz, speed, stiffness, vt_off. x axes synced (`uPlot.sync`). Cursor, history spans/ticks, action-ahead span, start-pad shade, scan gaps, pose-jump marks, and the never-sampled band are drawn on one absolutely positioned `<canvas>` per chart in a `requestAnimationFrame` callback; the uPlot series are never redrawn on cursor moves. History per chart: pose ticks on position/orientation, wrench ticks on force/torque; RGB ticks on a strip under the video plus a filmstrip of the `rgb` history frames drawn from the sprite with `drawImage` (only when `has_ckpt`); action-ahead span on all. Rollout mode draws ticks at `t_q − slowdown·(t_q − tick)` (and action at `t_q + slowdown·(tick − t_q)`), labelled "rollout (approx.)".
- `path3d.js`: `createPath3d(container, arrays) -> {setRange(t0, t1), setCursor(t), setColorBy(mode), destroy()}`. One `THREE.Line` with per-vertex colours over the full-rate pose; range applied with `geometry.setDrawRange` (no rebuild); cursor = small sphere + RGB axis triad from the quaternion; history segment drawn as a second line with draw range.
- Layout per the spec sketch. Video shown at 448×448 (`image-rendering: pixelated` off). Episode list: thumbnail, `dagger_2/ep7`, duration, flag dot (green keep / red reject / grey none / orange stale), source badge, counts per dataset, filter buttons (all / base / dagger / unflagged / keep / reject). Export dialog: output path, include radio, dry-run button, result text.
- Playback: rAF loop reads `video.currentTime`, maps to frame with binary search in `rgb_t` (`rgb_t[0]` aligned to video time 0), updates overlays. Slider drag: update overlays immediately, set `video.currentTime` only if no seek is pending; on `seeked`, apply the latest requested time if it changed. Speeds 0.25/0.5/1/2/4 via `playbackRate`.
- 3D start/end sliders plus `[`/`]`; checkbox "lock plots to 3D range" calls `setXRange`.
- Toggles: history on/off and training/rollout (hidden when `has_ckpt` false), never-sampled band (default on with ckpt), colour by time/|F|.

Steps:
- [ ] **Step 1: Vendor libs** — download uPlot (`dist/uPlot.esm.js`, `dist/uPlot.min.css`) and three.js (`build/three.module.min.js`, `examples/jsm/controls/OrbitControls.js`, fix its `import ... from 'three'` to `./three.module.min.js`) from jsdelivr into `static/vendor/`; record versions in `README.md`.
- [ ] **Step 2: Implement** `index.html`, `style.css`, `app.js`, `plots.js`, `path3d.js`.
- [ ] **Step 3: Manual verification** with `review_server.py --ckpt <reference ckpt>` in Chrome:
  - list shows 95 episodes, filters and counts correct; ↑/↓ changes episode in < 300 ms once prefetched.
  - play a ~1500-frame episode at 1x and 4x: cursor, 3D marker, history ticks follow without stutter.
  - drag the time slider quickly end to end: DevTools Performance shows no long tasks > 16 ms during the drag, video frame keeps up (at most one pending seek).
  - P/X/U update the dot and survive a page reload; N edits note.
  - 3D start/end sliders and `[`/`]` restrict the path; lock-to-range zooms the plots.
  - grey never-sampled band covers the start of each episode (~4.6 s, the longest obs window), and rollout toggle widens the ticks ~2x.
  - Export dry-run on a test selection returns the expected counts.
- [ ] **Step 4: Commit** `UMIFT_Data/review/static`.

---

### Task 9: Docs and work-log entry

**Files:**
- Create: `UMIFT_Data/review/README.md`, `docs/log/2026-11-15-episode-review-tool.md` (from `docs/log/_template.md`, `stream: bc`, branch from `git branch --show-current`, commit SHA of Task 8, `status: done`, artifacts: flags file path, cache dir).
- Modify: `README_layup_pipeline.md` — one short "Reviewing episodes" section pointing to `UMIFT_Data/review/README.md`.

- [ ] **Step 1:** README: run command, keys, toggles, export command, the "merge script per-input drop" change (uncommitted) and stale-flag behaviour.
- [ ] **Step 2:** Log entry: what changed, why (DAgger hurts policy), repro command, result "tool built, not yet used for a training dataset".
- [ ] **Step 3:** `python docs/log/index.py --stream bc` lists the entry.
- [ ] **Step 4: Commit** the three files.
