import json
import struct

import numpy as np
import zarr

from review_fixtures import write_synthetic_episode_zarr
from review import episodes


def _decode(buf: bytes):
    (hlen,) = struct.unpack("<I", buf[:4])
    header = json.loads(buf[4 : 4 + hlen].decode())
    off = 4 + hlen
    off += (-off) % 4
    arrays = {}
    for a in header["arrays"]:
        arrays[a["name"]] = np.frombuffer(buf, dtype="<f4", count=a["length"], offset=off)
        off += 4 * a["length"]
    assert off == len(buf)
    return header, arrays


def _two_datasets(tmp_path):
    b = write_synthetic_episode_zarr(tmp_path / "stringer_v3.5_base" / "acp_replay_buffer.zarr", n_eps=2)
    d = write_synthetic_episode_zarr(tmp_path / "stringer_v3.5_dagger_2" / "acp_replay_buffer.zarr", n_eps=3)
    return b, d


def test_index_orders_and_tags(tmp_path):
    b, d = _two_datasets(tmp_path)
    idx = episodes.DatasetIndex([str(b)], [str(d)])
    assert [e.key for e in idx.episodes] == ["base/0", "base/1", "dagger_2/0", "dagger_2/1", "dagger_2/2"]
    assert [e.source for e in idx.episodes] == ["original"] * 2 + ["dagger"] * 3
    assert idx.get("dagger_2/1").index == 1


def test_summary_fields(tmp_path):
    b, d = _two_datasets(tmp_path)
    idx = episodes.DatasetIndex([str(b)], [str(d)])
    s = idx.summary(idx.get("base/1"))
    assert s["key"] == "base/1" and s["source"] == "original" and s["n_rgb"] == 61
    assert abs(s["duration_s"] - 60 / 15.0) < 1e-6
    assert set(s["fingerprint"]) == {"n_rgb", "n_robot", "n_wrench", "t_robot_first", "t_robot_last"}


def test_payload_roundtrip_lengths(tmp_path):
    b, _ = _two_datasets(tmp_path)
    g = zarr.open(str(b), "r")["data"]["episode_0"]
    header, arrays = episodes.build_payload(g)
    h2, a2 = _decode(episodes.encode_payload(header, arrays))
    names = [a["name"] for a in h2["arrays"]]
    for n in ["rgb_t", "pose_t", "px", "py", "pz", "qw", "qx", "qy", "qz", "rx", "ry", "rz",
              "speed", "stiffness", "vt_off", "wrench_t", "fx", "fy", "fz", "tx", "ty", "tz", "fmag"]:
        assert n in names
    np.testing.assert_allclose(a2["pose_t"], g["robot_time_stamps_0"][:].ravel(), rtol=0, atol=1e-5)
    assert len(a2["px"]) == len(a2["pose_t"]) and len(a2["fx"]) == len(a2["wrench_t"])
    np.testing.assert_allclose(a2["vt_off"], 0.01, atol=1e-6)
    np.testing.assert_allclose(a2["fmag"], np.abs(a2["fy"]), atol=1e-5)
    assert h2["fps"] == 15.0 or abs(h2["fps"] - 15.0) < 1e-6


def test_extra_header_merged(tmp_path):
    b, _ = _two_datasets(tmp_path)
    g = zarr.open(str(b), "r")["data"]["episode_0"]
    header, _ = episodes.build_payload(g, extra={"window": {"rgb": 10}})
    assert header["window"] == {"rgb": 10}


def test_euler_unwrapped(tmp_path):
    b, _ = _two_datasets(tmp_path)
    g = zarr.open(str(b), "r+")["data"]["episode_0"]
    pose = g["ts_pose_fb_0"][:]
    yaw = np.linspace(0, 4 * np.pi, len(pose))  # two full turns about z
    pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6] = np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)
    g["ts_pose_fb_0"][:] = pose
    _, arrs = _decode(episodes.encode_payload(*episodes.build_payload(g, r_ref=np.eye(3))))
    assert np.max(np.abs(np.diff(arrs["rz"]))) < 10.0
    assert arrs["rz"][-1] > 700.0  # unwrapped, not folded back into +-180


def test_orientation_relative_to_reference(tmp_path):
    from scipy.spatial.transform import Rotation
    b, _ = _two_datasets(tmp_path)
    g = zarr.open(str(b), "r")["data"]["episode_0"]
    q0 = g["ts_pose_fb_0"][0, 3:]
    r0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
    header, arrs = _decode(episodes.encode_payload(*episodes.build_payload(g, r_ref=r0)))
    assert abs(arrs["rx"][0]) < 1e-3 and abs(arrs["ry"][0]) < 1e-3 and abs(arrs["rz"][0]) < 1e-3
    assert len(header["orientation_ref_quat_wxyz"]) == 4


def test_pose_jump_marker(tmp_path):
    b, _ = _two_datasets(tmp_path)
    g = zarr.open(str(b), "r+")["data"]["episode_0"]
    pose = g["ts_pose_fb_0"][:]
    pose[100:, 0] += 0.01
    g["ts_pose_fb_0"][:] = pose
    header, _ = episodes.build_payload(g)
    t = g["robot_time_stamps_0"][:].ravel()
    assert len(header["markers"]["pose_jumps"]) == 1
    assert abs(header["markers"]["pose_jumps"][0] - t[100]) < 1e-6


def test_scan_gap_marker_and_start_pad(tmp_path):
    p = write_synthetic_episode_zarr(tmp_path / "stringer_v3.5_x" / "acp_replay_buffer.zarr",
                                     n_eps=1, start_pad_s=5.0, scan_gaps=[[1.0, 2.0]])
    g = zarr.open(str(p), "r")["data"]["episode_0"]
    header, _ = episodes.build_payload(g, root_attrs={"start_pad_seconds": 5.0})
    assert header["markers"]["scan_gaps"] == [[1.0, 2.0]]
    assert header["start_pad_s"] == 5.0


def test_fingerprint_changes_with_length(tmp_path):
    a = write_synthetic_episode_zarr(tmp_path / "a" / "acp_replay_buffer.zarr", n_eps=1, n_rgb=60)
    b = write_synthetic_episode_zarr(tmp_path / "b" / "acp_replay_buffer.zarr", n_eps=1, n_rgb=61)
    fa = episodes.fingerprint(zarr.open(str(a), "r")["data"]["episode_0"])
    fb = episodes.fingerprint(zarr.open(str(b), "r")["data"]["episode_0"])
    assert fa != fb and fa["n_rgb"] == 60


def test_source_attr_disagreement_warns(tmp_path, capsys):
    b, d = _two_datasets(tmp_path)
    zarr.open(str(b), "r+")["data"]["episode_0"].attrs["source"] = "dagger"
    idx = episodes.DatasetIndex([str(b)], [str(d)])
    assert idx.get("base/0").source == "original"
    assert "WARNING" in capsys.readouterr().out


def test_display_name():
    assert episodes.dataset_display_name("/x/stringer_v3.5_dagger_2/acp_replay_buffer.zarr") == "dagger_2"
    assert episodes.dataset_display_name("/x/other_set/acp_replay_buffer.zarr") == "other_set"
