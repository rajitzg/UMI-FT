"""D405RawRecorder: wand-compatible manifest + raw files, no hardware required."""

import json
import time

import numpy as np

from PyriteEnvSuites.env_runners import d405_raw_recorder as rr
from PyriteEnvSuites.env_runners.d405_raw_recorder import D405RawRecorder


def _frame(i, with_color=False, drop_ir2=False):
    f = {
        "host_time_s": 1000.0 + i / 15,
        "infrared_1": np.full((480, 640), i % 256, np.uint8),
        "depth": np.full((480, 640), 1000 + i, np.uint16),
        "infrared_1_rs_timestamp_ms": 5.0 + i,
        "infrared_1_rs_frame_number": i,
    }
    if not drop_ir2:
        f["infrared_2"] = np.full((480, 640), (i + 1) % 256, np.uint8)
    if with_color:
        f["color"] = np.zeros((480, 640, 3), np.uint8)
    return f


def test_writes_wand_compatible_manifest_and_raw(tmp_path):
    clock = iter([10.0, 20.0, 30.0])
    rec = D405RawRecorder(tmp_path / "d405_raw", device_info={"serial": "x"},
                          server_clock_s=lambda: next(clock), extra_info={"ckpt_path": "/c"},
                          manifest_every_n=2)
    assert rec.start()
    for i in range(5):
        rec.push(_frame(i, drop_ir2=(i == 3)))
    info = rec.stop()
    assert info["recorded"] and info["frames"] == 5 and info["dropped"] == 0
    assert info["missing"]["infrared_2"] == 1

    m = json.loads((tmp_path / "d405_raw" / "manifest.json").read_text())
    s = m["streams"]
    assert set(s) == {"infrared_1", "infrared_2", "depth"}
    assert s["infrared_1"]["format"] == "y8" and s["depth"]["format"] == "z16"
    assert s["infrared_1"]["frame_count"] == 5 and s["infrared_2"]["frame_count"] == 4
    assert (s["infrared_1"]["width"], s["infrared_1"]["height"]) == (640, 480)
    # same reader logic as ffs_depth_frames.py: memmap + offset/length per frame
    a = s["infrared_1"]["frames"][4]
    m1 = np.memmap(tmp_path / "d405_raw" / "infrared_1.raw", np.uint8, "r")
    img = np.asarray(m1[a["offset"]:a["offset"] + a["length"]]).reshape(480, 640)
    assert img[0, 0] == 4 and a["rs_frame_number"] == 4
    d = s["depth"]["frames"][2]
    dm = np.memmap(tmp_path / "d405_raw" / "depth.raw", np.uint16, "r")
    assert dm[d["offset"] // 2] == 1002
    assert m["device"]["serial"] == "x"
    assert [c["tag"] for c in m["clock_sync"]] == ["start", "stop"]
    assert m["info"]["complete"] is True and m["info"]["ckpt_path"] == "/c"


def test_push_copies_buffer(tmp_path):
    rec = D405RawRecorder(tmp_path / "r")
    rec.start()
    f = _frame(0)
    buf = f["infrared_1"]
    rec.push(f)
    buf[:] = 99  # librealsense reuses its buffer after the callback returns
    rec.stop()
    m1 = np.fromfile(tmp_path / "r" / "infrared_1.raw", np.uint8)
    assert m1[0] == 0


def test_full_queue_drops_instead_of_blocking(tmp_path):
    rec = D405RawRecorder(tmp_path / "r", max_queue=2)
    rec.enabled = True  # no writer thread: the queue fills deterministically
    t0 = time.time()
    for i in range(20):
        rec.push(_frame(i))
    assert time.time() - t0 < 1.0
    assert rec.dropped == 18


def test_refuses_when_disk_low(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "MIN_FREE_GB", 1e12)
    rec = D405RawRecorder(tmp_path / "r")
    assert rec.start() is False
    rec.push(_frame(0))
    assert rec.stop()["recorded"] is False
