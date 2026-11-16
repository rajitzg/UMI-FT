import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from review_fixtures import write_synthetic_episode_zarr
from review import policy_window

REF_CKPT = Path(
    "/home/boeing/umi_ft_ws/data/training_outputs/stringer_v3.5/hist/dagger_123/0.35/"
    "epoch=0080-train_loss=0.016.ckpt"
)
BASE_ZARR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5_base/acp_replay_buffer.zarr")
HIST_JSON = Path(__file__).parent / "review_shape_meta_hist.json"


def _hist_spec() -> policy_window.WindowSpec:
    d = json.loads(HIST_JSON.read_text())
    return policy_window.spec_from_cfg_dicts(d["shape_meta"], d["dataset"])


def test_window_ticks_matches_sampler_semantics():
    t = np.arange(1000) * 0.005
    ticks = policy_window.window_ticks(t, np.array([0.0, 2.0, 4.999]), horizon=50, ds=16)
    assert ticks.shape == (3, 50) and ticks.dtype == np.float32
    assert np.all(ticks[0] == 0.0)  # front-padded with the first sample
    qi = np.searchsorted(t, 2.0)
    assert ticks[1, -1] == np.float32(t[qi]) and ticks[1, -2] == np.float32(t[qi - 16])
    assert np.all(np.diff(ticks[2]) >= 0)  # oldest first


def test_window_ticks_clamp_at_stream_end():
    t = np.arange(100) * 0.005
    ticks = policy_window.window_ticks(t, np.array([10.0]), horizon=4, ds=2)
    assert ticks[0, -1] == np.float32(t[-1])
    assert ticks[0, -2] == np.float32(t[-3])


def test_action_ticks_on_rgb_grid():
    rgb = np.arange(30) / 15.0
    act = np.arange(0, 2.0, 0.005)
    ticks = policy_window.action_ticks(act, rgb, horizon=24, rgb_stride=1)
    assert ticks.shape == (30, 24)
    i0 = np.searchsorted(act, rgb[0])
    assert ticks[0, 0] == np.float32(act[i0])
    assert abs(ticks[0, 23] - rgb[23]) < 0.006
    assert np.all(np.diff(ticks[-1]) >= 0)  # padded with the last valid time
    assert ticks[-1, -1] == ticks[-1, -2]


def test_spec_from_hist_cfg():
    s = _hist_spec()
    assert s.rgb == (10, 5) and s.low_dim == (50, 16) and s.wrench == (64, 2) and s.action == (24, 1)
    assert s.sampler_kwargs["sparse_query_frequency_down_sample_steps"] == 2
    assert "dagger_frac" not in s.sampler_kwargs and "dataset_path" not in s.sampler_kwargs


def test_episode_window_arrays_shapes(tmp_path):
    p = write_synthetic_episode_zarr(tmp_path / "s" / "acp_replay_buffer.zarr", n_eps=1, n_rgb=150)
    g = zarr.open(str(p), "r")["data"]["episode_0"]
    arrs = policy_window.episode_window_arrays(g, 0, _hist_spec())
    n = 150
    assert arrs["win_rgb"].shape == (n * 10,)
    assert arrs["win_pose"].shape == (n * 50,)
    assert arrs["win_wrench"].shape == (n * 64,)
    assert arrs["win_action"].shape == (n * 24,)
    assert arrs["sampled"].shape == (n,)


def test_sampled_mask_excludes_episode_start(tmp_path):
    # 150 frames @15 Hz = 10 s. Longest obs window: wrench 63*2 samples @28 Hz = 4.5 s.
    p = write_synthetic_episode_zarr(tmp_path / "s" / "acp_replay_buffer.zarr", n_eps=1, n_rgb=150)
    g = zarr.open(str(p), "r")["data"]["episode_0"]
    m = policy_window.sampled_mask(g, 0, _hist_spec())
    assert m.dtype == bool and m.shape == (150,)
    first = int(np.argmax(m))
    assert first / 15.0 >= 4.4
    assert m[first:].mean() > 0.9  # dilated by the query stride, so no every-other-frame holes


def test_sampled_mask_drops_scan_gap(tmp_path):
    p = write_synthetic_episode_zarr(tmp_path / "s" / "acp_replay_buffer.zarr", n_eps=1, n_rgb=200,
                                     scan_gaps=[[8.0, 9.0]])
    g = zarr.open(str(p), "r")["data"]["episode_0"]
    m = policy_window.sampled_mask(g, 0, _hist_spec())
    assert not m[int(8.5 * 15)]


@pytest.mark.slow
@pytest.mark.skipif(not REF_CKPT.exists(), reason="reference ckpt not present")
def test_load_window_spec_reference_ckpt(tmp_path):
    s = policy_window.load_window_spec(str(REF_CKPT), tmp_path)
    assert s.rgb == (10, 5) and s.low_dim == (50, 16) and s.wrench == (64, 2) and s.action[0] == 24
    assert list(tmp_path.glob("window_*.json"))
    s2 = policy_window.load_window_spec(str(REF_CKPT), tmp_path)  # cached
    assert s2.low_dim == s.low_dim


@pytest.mark.slow
@pytest.mark.skipif(not BASE_ZARR.exists(), reason="real data absent")
def test_sampled_mask_real_episode_matches_query_grid():
    g = zarr.open(str(BASE_ZARR), "r")["data"]["episode_0"]
    spec = _hist_spec()
    ids = policy_window.sampled_query_ids(g, 0, spec)
    assert len(ids) > 0 and np.all(np.diff(ids) % 2 == 0)  # query stride 2
    m = policy_window.sampled_mask(g, 0, spec)
    assert set(np.flatnonzero(m)) >= set(ids.tolist())
