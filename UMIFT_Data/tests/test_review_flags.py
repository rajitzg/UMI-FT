import json

import pytest

from review_fixtures import write_synthetic_episode_zarr
from review import episodes
from review.flags import FlagStore


def _index(tmp_path, n_eps=3):
    p = write_synthetic_episode_zarr(tmp_path / "stringer_v3.5_dagger" / "acp_replay_buffer.zarr", n_eps=n_eps)
    return episodes.DatasetIndex([], [str(p)])


def test_set_and_reload(tmp_path):
    idx = _index(tmp_path)
    ref = idx.get("dagger/1")
    fs = FlagStore(tmp_path / "flags" / "s.json")
    fs.load()
    e = fs.set(ref, idx.summary(ref)["fingerprint"], flag="reject", note="bad recovery")
    assert e["flag"] == "reject" and e["source"] == "dagger" and e["episode"] == 1
    fs2 = FlagStore(tmp_path / "flags" / "s.json")
    fs2.load()
    assert fs2.entries["dagger/1"]["note"] == "bad recovery"
    assert json.loads((tmp_path / "flags" / "s.json").read_text())["version"] == 1


def test_note_only_keeps_flag(tmp_path):
    idx = _index(tmp_path)
    ref = idx.get("dagger/0")
    fs = FlagStore(tmp_path / "f.json")
    fp = idx.summary(ref)["fingerprint"]
    fs.set(ref, fp, flag="keep")
    assert fs.set(ref, fp, note="ok")["flag"] == "keep"


def test_invalid_flag_raises(tmp_path):
    idx = _index(tmp_path)
    ref = idx.get("dagger/0")
    with pytest.raises(ValueError):
        FlagStore(tmp_path / "f.json").set(ref, {}, flag="maybe")


def test_atomic_save_leaves_no_tmp(tmp_path):
    idx = _index(tmp_path)
    ref = idx.get("dagger/0")
    fs = FlagStore(tmp_path / "f.json")
    fs.set(ref, idx.summary(ref)["fingerprint"], flag="keep")
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["f.json"]


def test_status_unflagged(tmp_path):
    idx = _index(tmp_path)
    ref = idx.get("dagger/2")
    st = FlagStore(tmp_path / "f.json").status(ref, idx.summary(ref)["fingerprint"])
    assert st["flag"] == "none" and st["stale"] is False


def test_stale_when_fingerprint_changes(tmp_path):
    idx = _index(tmp_path)
    ref = idx.get("dagger/1")
    fs = FlagStore(tmp_path / "f.json")
    fp = idx.summary(ref)["fingerprint"]
    fs.set(ref, fp, flag="keep")
    assert fs.status(ref, fp)["stale"] is False
    assert fs.status(ref, dict(fp, n_rgb=fp["n_rgb"] - 1))["stale"] is True


def test_stale_when_episode_missing(tmp_path):
    idx = _index(tmp_path, n_eps=3)
    fs = FlagStore(tmp_path / "f.json")
    ref = idx.get("dagger/2")
    fs.set(ref, idx.summary(ref)["fingerprint"], flag="keep")
    idx2 = _index(tmp_path, n_eps=2)  # zarr rebuilt with fewer episodes
    assert fs.orphans(idx2) == ["dagger/2"]


def test_note_only_edit_keeps_stale_flag_stale(tmp_path):
    idx = _index(tmp_path)
    ref = idx.get("dagger/1")
    fs = FlagStore(tmp_path / "f.json")
    fp = idx.summary(ref)["fingerprint"]
    fs.set(ref, fp, flag="keep")
    new_fp = dict(fp, n_rgb=fp["n_rgb"] + 5)  # zarr rebuilt underneath
    fs.set(ref, new_fp, note="check this")
    st = fs.status(ref, new_fp)
    assert st["stale"] is True and st["note"] == "check this"
    fs.set(ref, new_fp, flag="keep")  # an explicit re-decision refreshes it
    assert fs.status(ref, new_fp)["stale"] is False
