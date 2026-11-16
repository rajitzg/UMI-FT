import av
import zarr
from PIL import Image

from review_fixtures import write_synthetic_episode_zarr
from review import video


def _group(tmp_path, n_rgb=60):
    p = write_synthetic_episode_zarr(tmp_path / "s" / "acp_replay_buffer.zarr", n_eps=1, n_rgb=n_rgb)
    return zarr.open(str(p), "r")["data"]["episode_0"]


def test_video_frame_count(tmp_path):
    g = _group(tmp_path)
    mp4 = video.episode_video(g, tmp_path / "cache", "k1")
    with av.open(str(mp4)) as c:
        frames = sum(1 for _ in c.decode(video=0))
        st = c.streams.video[0]
        assert (st.width, st.height) == (224, 224)
    assert frames == 60


def test_video_all_keyframes(tmp_path):
    g = _group(tmp_path, n_rgb=20)
    mp4 = video.episode_video(g, tmp_path / "cache", "k1")
    with av.open(str(mp4)) as c:
        assert all(p.is_keyframe for p in c.demux(video=0) if p.size)


def test_video_cached_second_call(tmp_path):
    g = _group(tmp_path)
    a = video.episode_video(g, tmp_path / "cache", "k1")
    m = a.stat().st_mtime_ns
    b = video.episode_video(g, tmp_path / "cache", "k1")
    assert a == b and b.stat().st_mtime_ns == m
    assert not list((tmp_path / "cache").glob("*.tmp*"))


def test_thumb_size(tmp_path):
    g = _group(tmp_path)
    with Image.open(video.episode_thumb(g, tmp_path / "cache", "k1")) as im:
        assert im.size == (112, 112)


def test_sprite_grid(tmp_path):
    g = _group(tmp_path)
    path, geo = video.episode_sprite(g, tmp_path / "cache", "k1")
    assert geo == {"tile": 56, "cols": 8, "n": 60}
    with Image.open(path) as im:
        assert im.size == (448, 448)
