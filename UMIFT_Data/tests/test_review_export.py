import json

import pytest
import zarr

from review_fixtures import write_synthetic_episode_zarr
from review import episodes
from review.flags import FlagStore
from review.review_export import export_selection


def _setup(tmp_path):
    b = write_synthetic_episode_zarr(tmp_path / "stringer_v3.5_base" / "acp_replay_buffer.zarr", n_eps=2, start_pad_s=5.0)
    d = write_synthetic_episode_zarr(tmp_path / "stringer_v3.5_dagger" / "acp_replay_buffer.zarr", n_eps=3, start_pad_s=5.0)
    idx = episodes.DatasetIndex([str(b)], [str(d)])
    fs = FlagStore(tmp_path / "flags.json")
    return idx, fs


def _flag(idx, fs, key, flag):
    ref = idx.get(key)
    fs.set(ref, idx.summary(ref)["fingerprint"], flag=flag)


def _out_rows(out):
    root = zarr.open(str(out), "r")
    eps = sorted(root["data"].group_keys(), key=lambda n: int(n.split("_")[1]))
    return [(root["data"][e].attrs["source"], root["data"][e].attrs["source_episode"]) for e in eps]


def test_export_keep_only_tags(tmp_path):
    idx, fs = _setup(tmp_path)
    _flag(idx, fs, "base/1", "keep")
    _flag(idx, fs, "base/0", "reject")
    _flag(idx, fs, "dagger/0", "keep")
    _flag(idx, fs, "dagger/2", "keep")
    out = tmp_path / "out" / "acp_replay_buffer.zarr"
    s = export_selection(idx, fs, out)
    assert (s["n_episodes"], s["n_original"], s["n_dagger"]) == (3, 1, 2)
    assert _out_rows(out) == [("original", "episode_1"), ("dagger", "episode_0"), ("dagger", "episode_2")]
    assert zarr.open(str(out), "r").attrs["start_pad_seconds"] == 5.0


def test_export_keep_plus_unflagged(tmp_path):
    idx, fs = _setup(tmp_path)
    _flag(idx, fs, "dagger/1", "reject")
    out = tmp_path / "out" / "acp_replay_buffer.zarr"
    s = export_selection(idx, fs, out, include="keep+unflagged")
    assert s["n_episodes"] == 4
    assert ("dagger", "episode_1") not in _out_rows(out)


def test_export_base_only(tmp_path):
    idx, fs = _setup(tmp_path)
    _flag(idx, fs, "base/0", "keep")
    _flag(idx, fs, "base/1", "keep")
    out = tmp_path / "out" / "acp_replay_buffer.zarr"
    export_selection(idx, fs, out)
    assert [r[0] for r in _out_rows(out)] == ["original", "original"]


def test_export_refuses_stale(tmp_path):
    idx, fs = _setup(tmp_path)
    ref = idx.get("dagger/0")
    fp = idx.summary(ref)["fingerprint"]
    fs.set(ref, dict(fp, n_rgb=1), flag="keep")
    with pytest.raises(ValueError, match="stale"):
        export_selection(idx, fs, tmp_path / "out" / "acp_replay_buffer.zarr")


def test_export_refuses_empty_selection(tmp_path):
    idx, fs = _setup(tmp_path)
    with pytest.raises(ValueError, match="nothing"):
        export_selection(idx, fs, tmp_path / "out" / "acp_replay_buffer.zarr")


def test_export_refuses_existing_output(tmp_path):
    idx, fs = _setup(tmp_path)
    _flag(idx, fs, "base/0", "keep")
    out = tmp_path / "out" / "acp_replay_buffer.zarr"
    export_selection(idx, fs, out)
    with pytest.raises(FileExistsError):
        export_selection(idx, fs, out)
    export_selection(idx, fs, out, overwrite=True)


def test_export_dry_run_writes_nothing(tmp_path):
    idx, fs = _setup(tmp_path)
    _flag(idx, fs, "dagger/1", "keep")
    out = tmp_path / "out" / "acp_replay_buffer.zarr"
    s = export_selection(idx, fs, out, dry_run=True)
    assert s["n_episodes"] == 1 and s["n_dagger"] == 1
    assert not out.exists()


def test_selection_json_written(tmp_path):
    idx, fs = _setup(tmp_path)
    _flag(idx, fs, "dagger/1", "keep")
    out = tmp_path / "out" / "acp_replay_buffer.zarr"
    s = export_selection(idx, fs, out)
    sel = json.loads((tmp_path / "out" / "review_selection.json").read_text())
    assert s["selection_json"] == str(tmp_path / "out" / "review_selection.json")
    assert sel["include"] == "keep" and len(sel["source_map"]) == 1
    assert sel["source_map"][0]["source"] == "dagger"
    assert "dagger/1" in sel["flags"]
