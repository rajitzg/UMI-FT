"""Tests for the elapsed_time_0 observation (umift_ss_v2_elapsed_time task configs).

Run from the UMI-FT repo root:
    PYTHONPATH=PyriteML:PyriteUtility:. pytest PyriteML/tests/test_elapsed_time.py
"""

import os

import hydra
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs
from diffusion_policy.common.sampler_umift import (
    SequenceSamplerUmiFT,
    elapsed_time_since_policy_start,
)
from PyriteConfig.tasks.umift import umift_type_conversions as tc
from PyriteUtility.planning_control.policy_clock import PolicyElapsedClock

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "diffusion_policy", "config")
)


def _compose(task, *extra):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="train_umift_workspace", overrides=[f"task={task}", *extra]
        )
    apply_drop_scan_inputs(cfg)
    return cfg


# ----------------------------------------------------------------- clock (inference)
def test_clock_scales_wall_time_by_slow_down():
    clock = PolicyElapsedClock(slow_down_factor=2.0)
    clock.start(100.0)
    assert clock.elapsed_s(100.0) == 0.0
    assert clock.elapsed_s(120.0) == pytest.approx(10.0)  # 20 s wall = 10 s of demo
    unslowed = PolicyElapsedClock(1.0)
    unslowed.start(0.0)
    assert unslowed.elapsed_s(20.0) == pytest.approx(20.0)


def test_clock_adds_scan_pause_unscaled():
    clock = PolicyElapsedClock(slow_down_factor=2.0)
    clock.start(0.0)
    clock.add_pause(30.0)  # robot stopped for a scan; demo clock kept running
    # 30 s paused + (80 - 30) s moving at half speed = 30 + 25
    assert clock.elapsed_s(80.0) == pytest.approx(55.0)
    # right at the end of the pause nothing has been scaled yet
    assert clock.elapsed_s(30.0) == pytest.approx(30.0)


def test_clock_restart_clears_pauses_and_validates():
    clock = PolicyElapsedClock(2.0)
    with pytest.raises(RuntimeError):
        clock.elapsed_s(1.0)
    clock.start(0.0)
    clock.add_pause(5.0)
    clock.start(50.0)
    assert clock.paused_s == 0.0
    assert clock.elapsed_s(60.0) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        PolicyElapsedClock(0.0)


# ------------------------------------------------------------------ training side
def test_training_elapsed_floors_at_zero_inside_pad():
    assert elapsed_time_since_policy_start(0.0, 0.0, 5.0) == 0.0
    assert elapsed_time_since_policy_start(4.9, 0.0, 5.0) == 0.0
    assert elapsed_time_since_policy_start(15.0, 0.0, 5.0) == pytest.approx(10.0)
    assert elapsed_time_since_policy_start(15.0, 0.5, 0.0) == pytest.approx(14.5)


def test_train_and_inference_agree_with_slow_down():
    # A demo query 10 s after the policy start must look the same to the policy as the
    # robot being 10 demo-seconds in: 20 wall seconds at slow_down_factor=2.
    pad = 5.0
    train = elapsed_time_since_policy_start(pad + 10.0, 0.0, pad)
    clock = PolicyElapsedClock(2.0)
    clock.start(1000.0)
    assert clock.elapsed_s(1020.0) == pytest.approx(train)


# ------------------------------------------------------------------------ configs
def test_elapsed_time_task_adds_the_input():
    cfg = _compose("umift_ss_v2_elapsed_time")
    OmegaConf.resolve(cfg)
    sm = cfg.policy.shape_meta
    assert sm.obs.elapsed_time_0.shape == [1]
    assert sm.obs.elapsed_time_0.type == "low_dim"
    assert sm.obs.elapsed_time_0.start_pad_s == 5.0
    assert sm.sample.obs.sparse.elapsed_time_0.horizon == 1
    assert "scan_heatmap_0" in sm.obs  # scan input untouched

    base = _compose("umift_ss_v2")
    assert "elapsed_time_0" not in base.task.shape_meta.obs


def test_combined_task_has_elapsed_time_without_scan():
    cfg = _compose("umift_ss_v2_no_scan_elapsed_time")
    OmegaConf.resolve(cfg)
    apply_drop_scan_inputs(cfg)
    for sm in (cfg.task.shape_meta, cfg.policy.shape_meta):
        assert "elapsed_time_0" in sm.obs
        assert "elapsed_time_0" in sm.sample.obs.sparse
        assert not any("scan" in k for k in sm.obs)
        assert not any("scan" in k for k in sm.sample.obs.sparse)


def test_pad_can_be_overridden_for_unpadded_datasets():
    cfg = _compose("umift_ss_v2_elapsed_time", "task.elapsed_time_start_pad_s=0")
    OmegaConf.resolve(cfg)
    assert cfg.policy.shape_meta.obs.elapsed_time_0.start_pad_s == 0


def test_elapsed_time_reaches_policy_input_unchanged():
    cfg = _compose("umift_ss_v2_elapsed_time")
    OmegaConf.resolve(cfg)
    sm = OmegaConf.to_container(cfg.policy.shape_meta, resolve=True)
    obs_sparse = {
        "robot0_eef_pos": np.zeros((5, 3)),
        "robot0_eef_rot_axis_angle": np.tile([1, 0, 0, 0, 1, 0], (5, 1)).astype(float),
        "robot0_eef_wrench_left": np.zeros((32, 6)),
        "robot0_eef_wrench_right": np.zeros((32, 6)),
        "elapsed_time_0": np.array([[12.5]], dtype=np.float32),
    }
    out, _ = tc.sparse_obs_to_obs_sample(obs_sparse, sm, "reshape", [0], ignore_rgb=True)
    np.testing.assert_allclose(out["elapsed_time_0"], [[12.5]])
    assert out["elapsed_time_0"].dtype == np.float32


def test_obs_encoder_gets_one_more_feature():
    def width(task):
        cfg = _compose(task)
        cls = hydra.utils.get_class(cfg._target_)
        ws = cls(cfg, output_dir="/tmp/_unused_ws")
        return ws.model.obs_encoder.output_shape()[1]

    assert width("umift_ss_v2_no_scan_elapsed_time") == width("umift_ss_v2_no_scan") + 1


# ------------------------------------------------------------------------ sampler
def _synthetic_sampler(start_pad_s):
    """One 15 Hz episode, 40 s long, with a 5 s frozen pad at the start."""
    n_rgb = 40 * 15
    rgb_t = (np.arange(n_rgb) / 15.0).reshape(-1, 1)
    n_robot = 40 * 200
    robot_t = (np.arange(n_robot) / 200.0).reshape(-1, 1)
    pose9 = np.zeros((n_robot, 3)), np.tile([1, 0, 0, 0, 1, 0], (n_robot, 1)).astype(float)
    ep = {
        "obs": {
            "rgb_0": np.zeros((n_rgb, 4, 4, 3), dtype=np.uint8),
            "rgb_time_stamps_0": rgb_t,
            "robot_time_stamps_0": robot_t,
            "robot0_eef_pos": pose9[0],
            "robot0_eef_rot_axis_angle": pose9[1],
        },
        "action": np.zeros((n_robot, 3), dtype=np.float32),
        "action_time_stamps": robot_t,
    }
    shape_meta = {
        "id_list": [0],
        "obs": {
            "rgb_0": {"shape": [3, 4, 4], "type": "rgb"},
            "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
            "robot0_eef_rot_axis_angle": {"shape": [6], "type": "low_dim"},
            "rgb_time_stamps_0": {"shape": [1], "type": "timestamp"},
            "elapsed_time_0": {
                "shape": [1],
                "type": "low_dim",
                "start_pad_s": start_pad_s,
            },
        },
        "sample": {
            "obs": {
                "sparse": {
                    "rgb_0": {"horizon": 1, "down_sample_steps": 1},
                    "robot0_eef_pos": {"horizon": 2, "down_sample_steps": 1},
                    "robot0_eef_rot_axis_angle": {"horizon": 2, "down_sample_steps": 1},
                    "elapsed_time_0": {"horizon": 1, "down_sample_steps": 1},
                }
            },
            "action": {"sparse": {"horizon": 2, "down_sample_steps": 1}},
        },
    }
    sampler = object.__new__(SequenceSamplerUmiFT)
    sampler.shape_meta = OmegaConf.create(shape_meta)
    sampler.replay_buffer = {"data": {"episode_0": ep}}
    sampler.ignore_rgb_is_applied = False
    sampler.flag_has_dense = False
    sampler.scan_pre_first_fill = "zeros"
    sampler.action_padding = True
    sampler.id_list = [0]
    sampler.indices = [(0, rid) for rid in (0, 30, 75, 150, 300)]
    sampler.obs_to_obs_sample = tc.obs_to_obs_sample
    sampler.action_to_action_sample = lambda **kw: ({"sparse": kw["action_sparse"]}, None)
    return sampler


def test_sampler_emits_elapsed_time_from_query_time():
    sampler = _synthetic_sampler(start_pad_s=5.0)
    got = []
    for i in range(len(sampler.indices)):
        obs, _ = sampler.sample_sequence(i)
        assert obs["sparse"]["elapsed_time_0"].shape == (1, 1)
        got.append(float(obs["sparse"]["elapsed_time_0"][0, 0]))
    # query times: 0, 2, 5, 10, 20 s -> inside the pad is 0, then time past the pad
    np.testing.assert_allclose(got, [0.0, 0.0, 0.0, 5.0, 15.0], atol=1e-5)
