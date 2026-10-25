"""Unit tests for ManipServerLayupEnv helpers (no hardware required)."""

import numpy as np

from PyriteEnvSuites.envs.task.manip_server_layup_env import (
    D405VisionRingBuffer,
    convert_color_frame_to_policy,
    convert_depth_frame_to_umift,
)


def test_convert_depth_frame_to_umift_clips_and_expands_channels():
    depth_mm = np.array([[0, 500, 1000], [1500, 2000, 3000]], dtype=np.uint16)
    out = convert_depth_frame_to_umift(
        depth_mm, depth_scale=0.001, depth_clip=0.5, output_hw=(2, 3)
    )
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.float16
    assert float(out[0, 1, 0]) == 0.5  # 1000 mm clipped to 0.5 m
    assert float(out[0, 0, 0]) == 0.0


def test_convert_color_frame_to_policy_applies_transform_chain():
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    color[0, 0] = [10, 20, 30]

    def raw_to_input(img):
        assert img.shape == (480, 640, 3)
        return img[::2, ::2, :]

    def to_policy(img):
        assert img.shape == (240, 320, 3)
        return img[:10, :10, :]

    out = convert_color_frame_to_policy(color, raw_to_input, to_policy)
    assert out.shape == (10, 10, 3)
    assert out[0, 0, 0] == 10


def test_d405_vision_ring_buffer_get_last_k_rgb_padding():
    buf = D405VisionRingBuffer.__new__(D405VisionRingBuffer)
    buf.buffer_size = 4
    buf.policy_rgb_hw = (2, 2)
    buf.output_hw = (2, 2)
    buf._lock = __import__("threading").Lock()
    frame = np.ones((2, 2, 3), dtype=np.uint8) * 7
    buf._rgb_frames = __import__("collections").deque([frame], maxlen=4)
    buf._rgb_timestamps_s = __import__("collections").deque([1.0], maxlen=4)

    frames, ts = buf.get_last_k_rgb(3)
    assert frames.shape == (3, 2, 2, 3)
    assert ts.shape == (3,)
    assert ts[0] == 1.0


def test_d405_vision_ring_buffer_get_last_k_depth_padding():
    buf = D405VisionRingBuffer.__new__(D405VisionRingBuffer)
    buf.buffer_size = 4
    buf.policy_rgb_hw = (2, 2)
    buf.output_hw = (2, 2)
    buf._lock = __import__("threading").Lock()
    frame = np.ones((2, 2, 3), dtype=np.float16) * 0.25
    buf._depth_frames = __import__("collections").deque([frame], maxlen=4)
    buf._depth_timestamps_s = __import__("collections").deque([2.0], maxlen=4)

    frames, ts = buf.get_last_k_depth(3)
    assert frames.shape == (3, 2, 2, 3)
    assert ts.shape == (3,)
    assert ts[0] == 2.0


def test_wrench_key_mapping_logic():
    wrench_6 = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float64)
    left = wrench_6[:, :6]
    right = np.zeros_like(left)
    assert left.shape == (1, 6)
    assert np.allclose(right, 0)

    wrench_12 = np.arange(12, dtype=np.float64).reshape(1, 12)
    left = wrench_12[:, :6]
    right = wrench_12[:, 6:12]
    assert np.allclose(left, [0, 1, 2, 3, 4, 5])
    assert np.allclose(right, [6, 7, 8, 9, 10, 11])


def test_layup_raw_to_input_is_letterboxed_like_training():
    """Live 640x480 frames must be padded to square (black bars), not center-cropped."""
    from PyriteUtility.computer_vision.computer_vision_utility import (
        get_image_transform_with_border,
    )

    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    out = get_image_transform_with_border((640, 480), (224, 224))(frame)
    assert out.shape == (224, 224, 3)
    bar = (224 - 168) // 2  # 4:3 content in a square -> 28 rows of black top/bottom
    assert out[:bar].max() == 0 and out[-bar:].max() == 0
    assert out[bar + 2 : -bar - 2].min() > 190


def test_layup_env_uses_letterbox_transform():
    import inspect

    from PyriteEnvSuites.envs.task import manip_server_layup_env as m

    src = inspect.getsource(m.ManipServerLayupEnv.__init__)
    assert "get_image_transform_with_border" in src
