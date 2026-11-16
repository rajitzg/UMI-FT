"""What the policy observes at each RGB query frame, taken from a checkpoint.

* ``window_ticks`` replicates the sparse-obs index arithmetic in
  ``SequenceSamplerUmiFT.sample_sequence``: ``query_id = searchsorted(times, t)``,
  ``num_valid = min(h, query_id // ds + 1)``, indices ``query_id - k*ds``,
  front-padded with the oldest valid sample.
* ``action_ticks`` reuses ``sparse_action_indices_from_rgb_timestamps``.
* ``sampled_query_ids`` runs the real ``SequenceSamplerUmiFT`` on one episode
  (raw zarr group passed straight to ``raw_to_obs``/``raw_to_action21``, which keep
  image arrays lazy, so no RGB is decoded). ``sampled_mask`` dilates those ids
  forward by ``sparse_query_frequency_down_sample_steps - 1`` frames so the
  query stride does not show up as every-other-frame holes; what stays grey are
  spans the sampler never queries (episode start, gaps, bad poses, end).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import zarr

from diffusion_policy.common.sampler_umift import (
    SequenceSamplerUmiFT,
    sparse_action_indices_from_rgb_timestamps,
)


@dataclass
class WindowSpec:
    rgb: Tuple[int, int]
    low_dim: Tuple[int, int]
    wrench: Tuple[int, int]
    action: Tuple[int, int]
    sampler_kwargs: dict = field(default_factory=dict)
    shape_meta: dict = field(default_factory=dict)

    def header(self) -> dict:
        return {"rgb": self.rgb[0], "pose": self.low_dim[0], "wrench": self.wrench[0],
                "action": self.action[0],
                "strides": {"rgb": self.rgb[1], "pose": self.low_dim[1],
                            "wrench": self.wrench[1], "action": self.action[1]},
                "query_stride": int(self.sampler_kwargs.get("sparse_query_frequency_down_sample_steps", 1))}


def _sampler_params() -> set:
    return set(inspect.signature(SequenceSamplerUmiFT.__init__).parameters) - {
        "self", "shape_meta", "replay_buffer", "obs_to_obs_sample",
        "action_to_action_sample", "id_list", "episode_mask"}


def _pick(sparse: dict, preferred: str, contains: str = None, exclude=()) -> Tuple[int, int]:
    if preferred in sparse:
        a = sparse[preferred]
    else:
        cands = [k for k in sparse if (contains is None or contains in k)
                 and not any(x in k for x in exclude)]
        if not cands:
            raise KeyError(f"no obs key like {preferred!r} in shape_meta.sample.obs.sparse")
        a = sparse[cands[0]]
    return int(a["horizon"]), int(a["down_sample_steps"])


def spec_from_cfg_dicts(shape_meta: dict, dataset_cfg: dict) -> WindowSpec:
    sparse = shape_meta["sample"]["obs"]["sparse"]
    act = shape_meta["sample"]["action"]["sparse"]
    allowed = _sampler_params()
    return WindowSpec(
        rgb=_pick(sparse, "rgb_0"),
        low_dim=_pick(sparse, "robot0_eef_pos", exclude=("rgb", "depth", "ultrawide", "wrench", "scan")),
        wrench=_pick(sparse, "robot0_eef_wrench_left", contains="wrench"),
        action=(int(act["horizon"]), int(act["down_sample_steps"])),
        sampler_kwargs={k: v for k, v in dataset_cfg.items() if k in allowed},
        shape_meta=shape_meta,
    )


def load_window_spec(ckpt_path: str, cache_dir: Path) -> WindowSpec:
    p = Path(ckpt_path).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{p}:{p.stat().st_mtime}".encode()).hexdigest()[:16]
    cache = cache_dir / f"window_{key}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
    else:
        import dill
        import torch
        from omegaconf import OmegaConf
        try:
            payload = torch.load(str(p), map_location="cpu", pickle_module=dill,
                                 mmap=True, weights_only=False)
        except Exception as e:  # older non-zip checkpoints cannot be mmapped
            print(f"[review] WARNING mmap load failed ({e}); loading the whole checkpoint")
            payload = torch.load(str(p), map_location="cpu", pickle_module=dill, weights_only=False)
        cfg = payload["cfg"]
        d = {"shape_meta": OmegaConf.to_container(cfg.task.shape_meta, resolve=True),
             "dataset": OmegaConf.to_container(cfg.task.dataset, resolve=True),
             "ckpt": str(p)}
        del payload
        cache.write_text(json.dumps(d))
    return spec_from_cfg_dicts(d["shape_meta"], d["dataset"])


def window_ticks(times: np.ndarray, query_times: np.ndarray, horizon: int, ds: int) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    q = np.clip(np.searchsorted(times, np.asarray(query_times, dtype=np.float64).reshape(-1)),
                0, len(times) - 1)
    k = np.arange(horizon - 1, -1, -1)  # oldest first
    idx = q[:, None] - k[None, :] * ds
    num_valid = np.minimum(horizon, q // ds + 1)
    oldest = q - (num_valid - 1) * ds
    idx = np.maximum(idx, oldest[:, None])
    return times[idx].astype(np.float32)


def action_ticks(action_times: np.ndarray, rgb_times: np.ndarray, horizon: int, rgb_stride: int) -> np.ndarray:
    action_times = np.asarray(action_times, dtype=np.float64).reshape(-1)
    rgb_times = np.asarray(rgb_times, dtype=np.float64).reshape(-1)
    out = np.empty((len(rgb_times), horizon), dtype=np.float32)
    if len(rgb_times) < 2 or len(action_times) == 0:  # sampler helper needs >= 2 rgb frames
        out[:] = rgb_times[:, None] if len(rgb_times) else 0.0
        return out
    for i in range(len(rgb_times)):
        idx, _ = sparse_action_indices_from_rgb_timestamps(action_times, rgb_times, i, horizon, rgb_stride)
        if len(idx) == 0:
            idx = np.array([len(action_times) - 1])
        vals = action_times[idx]
        out[i, : len(vals)] = vals
        out[i, len(vals):] = vals[-1]
    return out


def _one_episode_buffer(group: zarr.Group, episode_index: int, shape_meta: dict) -> dict:
    from PyriteConfig.tasks.umift.umift_type_conversions import (
        raw_to_action9, raw_to_action10, raw_to_action19, raw_to_action21, raw_to_obs)
    conv = {9: raw_to_action9, 18: raw_to_action9, 10: raw_to_action10, 20: raw_to_action10,
            19: raw_to_action19, 38: raw_to_action19, 21: raw_to_action21, 42: raw_to_action21}
    ep: dict = {}
    raw_to_obs(group, ep, shape_meta)
    conv[int(shape_meta["action"]["shape"][0])](group, ep, shape_meta["id_list"])
    if "scan_gap_intervals_0" in group:
        ep["scan_gap_intervals_0"] = np.asarray(group["scan_gap_intervals_0"][:])
    n_rgb = int(group["rgb_0"].shape[0])
    return {"data": {f"episode_{episode_index}": ep},
            "meta": {"episode_rgb0_len": np.array([n_rgb], dtype=np.int64)}}


def sampled_query_ids(group: zarr.Group, episode_index: int, spec: WindowSpec) -> np.ndarray:
    rb = _one_episode_buffer(group, episode_index, spec.shape_meta)
    sampler = SequenceSamplerUmiFT(
        shape_meta=spec.shape_meta, replay_buffer=rb,
        obs_to_obs_sample=None, action_to_action_sample=None,
        id_list=spec.shape_meta["id_list"], **spec.sampler_kwargs)
    return np.array(sorted(int(r) for _, r in sampler.indices), dtype=np.int64)


def sampled_mask(group: zarr.Group, episode_index: int, spec: WindowSpec) -> np.ndarray:
    n = int(group["rgb_0"].shape[0])
    stride = int(spec.sampler_kwargs.get("sparse_query_frequency_down_sample_steps", 1))
    m = np.zeros(n, dtype=bool)
    for i in sampled_query_ids(group, episode_index, spec):
        m[i: min(n, i + stride)] = True
    return m


def episode_window_arrays(group: zarr.Group, episode_index: int, spec: WindowSpec) -> Tuple[Dict[str, np.ndarray], dict]:
    """Window tick arrays + info. Episodes the training sampler rejects (it raises on
    some malformed episodes) get an all-False ``sampled`` mask and
    ``info["sampled_error"]`` instead of failing the whole payload."""
    try:
        sampled = sampled_mask(group, episode_index, spec)
        err = None
    except Exception as e:
        sampled = np.zeros(int(group["rgb_0"].shape[0]), dtype=bool)
        err = f"{type(e).__name__}: {e}"
    rgb_t = np.ravel(group["rgb_time_stamps_0"][:]).astype(np.float64)
    rob_t = np.ravel(group["robot_time_stamps_0"][:]).astype(np.float64)
    wr_t = np.ravel(group["wrench_time_stamps_0"][:]).astype(np.float64)
    return {
        "win_rgb": window_ticks(rgb_t, rgb_t, *spec.rgb).ravel(),
        "win_pose": window_ticks(rob_t, rgb_t, *spec.low_dim).ravel(),
        "win_wrench": window_ticks(wr_t, rgb_t, *spec.wrench).ravel(),
        "win_action": action_ticks(rob_t, rgb_t, *spec.action).ravel(),
        "sampled": sampled.astype(np.float32),
    }, {"sampled_error": err}
