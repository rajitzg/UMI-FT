import importlib.util
import pathlib
import tempfile

import numpy as np
import zarr
from numcodecs import Blosc


SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "merge_umift_zarr_datasets.py"
)
SPEC = importlib.util.spec_from_file_location("merge_umift_zarr_datasets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_toy_zarr(path: pathlib.Path, n_eps: int, base_len: int) -> None:
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
    # Create in non-numeric disk order when possible; merge must sort numerically.
    create_order = list(range(n_eps - 1, -1, -1))
    for i in create_order:
        T = base_len + i
        ep = data.create_group(f"episode_{i}")
        for key in MODULE.REQUIRED_EPISODE_KEYS:
            if key in {"rgb_0", "ultrawide_0"}:
                shape = (T, 4, 4, 3)
                dtype = np.uint8
            elif key == "depth_0":
                shape = (T, 4, 4, 3)
                dtype = np.float16
            elif key.startswith("wrench"):
                shape = (T, 6)
                dtype = np.float64
            elif key == "ts_pose_fb_0":
                shape = (T, 7)
                dtype = np.float64
            else:
                shape = (T, 1)
                dtype = np.float64
            arr = np.zeros(shape, dtype=dtype)
            if key == "ts_pose_fb_0":
                arr[:, 3] = 1.0
            ep.create_dataset(
                key, data=arr, chunks=True, dtype=dtype, compressor=compressor
            )
    cum = {k: [] for k in MODULE.META_LEN_KEYS}
    for i in range(n_eps):
        T = base_len + i
        for meta_key in MODULE.META_LEN_KEYS:
            MODULE._append_cumulative(cum[meta_key], T)
    for meta_key, values in cum.items():
        meta.create_dataset(meta_key, data=np.asarray(values, dtype=np.int64))


def test_list_episodes_numeric_not_lexicographic():
    with tempfile.TemporaryDirectory() as tmp:
        root = zarr.open(tmp, mode="w")
        data = root.create_group("data")
        for name in ("episode_10", "episode_2", "episode_1"):
            data.create_group(name)
        assert MODULE.list_episodes_numeric(data) == [
            "episode_1",
            "episode_2",
            "episode_10",
        ]


def test_merge_rebuilds_cumulative_meta_and_renumbers():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        a = tmp_path / "a.zarr"
        b = tmp_path / "b.zarr"
        out = tmp_path / "out.zarr"
        _write_toy_zarr(a, n_eps=3, base_len=5)
        _write_toy_zarr(b, n_eps=2, base_len=20)

        summary = MODULE.merge_umift_zarr_datasets(
            [a, b], out, overwrite=True, dry_run=False
        )
        assert summary["n_episodes"] == 5
        MODULE.validate_merged_zarr(out)

        root = zarr.open(str(out), mode="r")
        eps = MODULE.list_episodes_numeric(root["data"])
        assert eps == [f"episode_{i}" for i in range(5)]

        # Lengths: a has T=5,6,7 for ep 0,1,2; b has 20,21.
        rgb_lens = []
        cum = np.asarray(root["meta"]["episode_rgb0_len"][:])
        for i, ep in enumerate(eps):
            start = 0 if i == 0 else int(cum[i - 1])
            rgb_lens.append(int(cum[i]) - start)
            assert root["data"][ep]["rgb_0"].shape[0] == rgb_lens[-1]
        assert rgb_lens == [5, 6, 7, 20, 21]

        # Source map exists and covers both inputs.
        map_path = out.parent / "episode_source_map.json"
        assert map_path.is_file()


def test_naive_concat_meta_would_be_wrong():
    """Document why stock merge breaks: concatenating cumulative arrays."""
    a = np.array([100, 250], dtype=np.int64)  # lens 100,150
    b = np.array([20, 41], dtype=np.int64)  # lens 20,21 (starts over at 0)
    naive = np.concatenate([a, b])
    diffs = np.diff(np.concatenate([[0], naive]))
    assert diffs[2] < 0  # 20 - 250 at dataset boundary
    rebuilt = []
    for lens in ([100, 150], [20, 21]):
        for L in lens:
            MODULE._append_cumulative(rebuilt, L)
    assert rebuilt == [100, 250, 270, 291]


# --- per-input drops + root attrs (episode review tool export) ---

import pytest


def _sources(out):
    root = zarr.open(str(out), mode="r")
    eps = MODULE.list_episodes_numeric(root["data"])
    return [root["data"][e].attrs.get("source") for e in eps]


def test_drop_by_input_only_affects_that_input(tmp_path):
    a, b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    _write_toy_zarr(a, 3, 10)
    _write_toy_zarr(b, 3, 20)
    out = tmp_path / "out.zarr"
    s = MODULE.merge_umift_zarr_datasets(
        [a], out, dagger_inputs=[b], drop_episodes_by_input={str(b): [1]}
    )
    srcs = [(r["source_zarr"], r["source_episode"]) for r in s["source_map"]]
    assert (str(a.resolve()), "episode_1") in srcs
    assert (str(b.resolve()), "episode_1") not in srcs
    assert len(srcs) == 5
    assert _sources(out) == ["original"] * 3 + ["dagger"] * 2
    MODULE.validate_merged_zarr(out)


def test_drop_by_input_base_only_tags_original(tmp_path):
    a, b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    _write_toy_zarr(a, 2, 10)
    _write_toy_zarr(b, 2, 20)
    out = tmp_path / "out.zarr"
    MODULE.merge_umift_zarr_datasets(
        [a], out, dagger_inputs=[b], drop_episodes_by_input={str(b): [0, 1]}
    )
    assert _sources(out) == ["original", "original"]


def test_drop_by_input_unknown_index_raises(tmp_path):
    a = tmp_path / "a.zarr"
    _write_toy_zarr(a, 3, 10)
    with pytest.raises(KeyError):
        MODULE.merge_umift_zarr_datasets(
            [a], tmp_path / "out.zarr", drop_episodes_by_input={str(a): [7]}
        )


def test_drop_by_input_unknown_path_raises(tmp_path):
    a = tmp_path / "a.zarr"
    _write_toy_zarr(a, 1, 10)
    with pytest.raises(ValueError):
        MODULE.merge_umift_zarr_datasets(
            [a], tmp_path / "out.zarr", drop_episodes_by_input={str(tmp_path / "x.zarr"): [0]}
        )


def test_root_attrs_kept_when_inputs_agree(tmp_path):
    a, b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    _write_toy_zarr(a, 1, 10)
    _write_toy_zarr(b, 1, 20)
    for p in (a, b):
        zarr.open(str(p), mode="r+").attrs["start_pad_seconds"] = 5.0
    out = tmp_path / "out.zarr"
    MODULE.merge_umift_zarr_datasets([a, b], out)
    assert zarr.open(str(out), mode="r").attrs["start_pad_seconds"] == 5.0


def test_root_attrs_dropped_when_inputs_disagree(tmp_path, capsys):
    a, b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    _write_toy_zarr(a, 1, 10)
    _write_toy_zarr(b, 1, 20)
    zarr.open(str(a), mode="r+").attrs["start_pad_seconds"] = 5.0
    zarr.open(str(b), mode="r+").attrs["start_pad_seconds"] = 0.0
    out = tmp_path / "out.zarr"
    MODULE.merge_umift_zarr_datasets([a, b], out)
    assert "start_pad_seconds" not in zarr.open(str(out), mode="r").attrs
    assert "start_pad_seconds" in capsys.readouterr().out


def test_write_source_map_false_skips_json(tmp_path):
    a = tmp_path / "a.zarr"
    _write_toy_zarr(a, 1, 10)
    MODULE.merge_umift_zarr_datasets([a], tmp_path / "out.zarr", write_source_map=False)
    assert not (tmp_path / "episode_source_map.json").exists()


def test_cli_drop_by_input_parsing():
    assert MODULE._parse_drop_by_input(["/x/b.zarr:3,7", "/x/b.zarr:9"]) == {"/x/b.zarr": [3, 7, 9]}
    with pytest.raises(ValueError):
        MODULE._parse_drop_by_input(["nocolon"])
