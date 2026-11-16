import json
import threading
import urllib.error
import urllib.request

import pytest

from review_fixtures import write_synthetic_episode_zarr
from review import review_server
from test_review_episodes import _decode


@pytest.fixture
def server(tmp_path):
    b = write_synthetic_episode_zarr(tmp_path / "stringer_v3.5_base" / "acp_replay_buffer.zarr", n_eps=2)
    d = write_synthetic_episode_zarr(tmp_path / "stringer_v3.5_dagger" / "acp_replay_buffer.zarr", n_eps=2)
    args = review_server.parse_args([
        "--base", str(b), "--dagger", str(d), "--port", "0",
        "--flags", str(tmp_path / "flags.json"), "--cache-dir", str(tmp_path / "cache")])
    srv = review_server.make_server(args)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, dict(r.headers), r.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_index_html_served(server):
    url, _ = server
    st, h, body = _get(url + "/")
    assert st == 200 and b"<html" in body.lower()


def test_episodes_endpoint(server):
    url, _ = server
    st, _, body = _get(url + "/api/episodes")
    d = json.loads(body)
    assert [e["key"] for e in d["episodes"]] == ["base/0", "base/1", "dagger/0", "dagger/1"]
    assert d["episodes"][2]["source"] == "dagger" and d["episodes"][0]["flag"] == "none"
    assert d["has_ckpt"] is False and d["window"] is None and d["orphans"] == []


def test_data_endpoint_decodes(server):
    url, _ = server
    st, h, body = _get(url + "/api/episode/dagger/1/data")
    assert st == 200 and h["Content-Type"] == "application/octet-stream"
    header, arrays = _decode(body)
    assert len(arrays["rgb_t"]) == 61
    assert header["sprite"]["n"] == 61 and header["video_fps"] == 15


def test_video_range_request(server):
    url, _ = server
    st, h, full = _get(url + "/api/episode/base/0/video.mp4")
    assert st == 200 and h["Content-Type"] == "video/mp4" and h["Accept-Ranges"] == "bytes"
    st, h, part = _get(url + "/api/episode/base/0/video.mp4", {"Range": "bytes=0-99"})
    assert st == 206 and len(part) == 100 and part == full[:100]
    assert h["Content-Range"] == f"bytes 0-99/{len(full)}"
    st, h, tail = _get(url + "/api/episode/base/0/video.mp4", {"Range": "bytes=100-"})
    assert st == 206 and tail == full[100:]


def test_thumb_and_sprite(server):
    url, _ = server
    for k in ("thumb.jpg", "sprite.jpg"):
        st, h, body = _get(url + f"/api/episode/base/1/{k}")
        assert st == 200 and h["Content-Type"] == "image/jpeg" and body[:2] == b"\xff\xd8"


def test_flag_post_persists(server):
    url, tmp = server
    st, d = _post(url + "/api/flag", {"key": "dagger/0", "flag": "reject", "note": "jumps away"})
    assert st == 200 and d["flag"] == "reject" and d["stale"] is False
    saved = json.loads((tmp / "flags.json").read_text())
    assert saved["episodes"]["dagger/0"]["note"] == "jumps away"
    _, _, body = _get(url + "/api/episodes")
    assert json.loads(body)["episodes"][2]["flag"] == "reject"


def test_flag_post_bad_flag_is_400(server):
    url, _ = server
    st, d = _post(url + "/api/flag", {"key": "dagger/0", "flag": "maybe"})
    assert st == 400 and "error" in d


def test_export_post_error_is_400(server):
    url, tmp = server
    st, d = _post(url + "/api/export", {"output": str(tmp / "o" / "acp_replay_buffer.zarr"), "include": "keep"})
    assert st == 400 and "nothing" in d["error"]


def test_export_post_ok(server):
    url, tmp = server
    _post(url + "/api/flag", {"key": "dagger/1", "flag": "keep"})
    st, d = _post(url + "/api/export", {"output": str(tmp / "o" / "acp_replay_buffer.zarr"), "include": "keep"})
    assert st == 200 and d["n_dagger"] == 1


def test_unknown_episode_404(server):
    url, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(url + "/api/episode/nope/3/data")
    assert e.value.code == 404


def _raw_post(url, data, headers):
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_post_requires_json_content_type(server):
    url, tmp = server
    body = json.dumps({"key": "dagger/0", "flag": "reject"}).encode()
    assert _raw_post(url + "/api/flag", body, {"Content-Type": "text/plain"}) == 415
    assert not (tmp / "flags.json").exists()


def test_post_rejects_foreign_origin(server):
    url, _ = server
    body = json.dumps({"key": "dagger/0", "flag": "reject"}).encode()
    st = _raw_post(url + "/api/flag", body, {"Content-Type": "application/json", "Origin": "http://evil.example"})
    assert st == 403
    st = _raw_post(url + "/api/flag", body, {"Content-Type": "application/json", "Origin": url})
    assert st == 200


def test_export_busy_is_409(server):
    url, tmp = server
    srv_state = review_server._LAST_STATE
    with srv_state.export_lock:
        st, d = _post(url + "/api/export", {"output": str(tmp / "o" / "acp_replay_buffer.zarr")})
    assert st == 409


def test_data_endpoint_with_spec_survives_short_robot_stream(server):
    import zarr as _z
    from test_review_policy_window import _hist_spec, _truncate_robot
    url, tmp = server
    _truncate_robot(tmp / "stringer_v3.5_dagger" / "acp_replay_buffer.zarr", 100)
    review_server._LAST_STATE.spec = _hist_spec()
    st, h, body = _get(url + "/api/episode/dagger/0/data")
    header, arrays = _decode(body)
    assert st == 200 and header["sampled_error"] and not arrays["sampled"].any()


def test_malformed_range_is_not_500(server):
    url, _ = server
    st, h, body = _get(url + "/api/episode/base/0/video.mp4", {"Range": "bytes=-"})
    assert st == 200
