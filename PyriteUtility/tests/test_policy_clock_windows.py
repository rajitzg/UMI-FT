from omegaconf import OmegaConf

from PyriteUtility.planning_control.policy_clock import scale_obs_windows_by_slow_down


def _sparse():
    return OmegaConf.create(
        {
            "rgb_0": {"horizon": 5, "down_sample_steps": 4},
            "robot0_eef_pos": {"horizon": 5, "down_sample_steps": 16},
            "robot0_eef_wrench_left": {"horizon": 32, "down_sample_steps": 1},
            "elapsed_time_0": {"horizon": 1, "down_sample_steps": 1},
            "scan_heatmap_0": {"horizon": 1, "down_sample_steps": 3},
        }
    )


def test_windows_scale_and_horizon_one_untouched():
    s = _sparse()
    ch = scale_obs_windows_by_slow_down(s, 2)
    assert s.rgb_0.down_sample_steps == 8
    assert s.robot0_eef_pos.down_sample_steps == 32
    assert s.robot0_eef_wrench_left.down_sample_steps == 2
    assert s.elapsed_time_0.down_sample_steps == 1
    assert s.scan_heatmap_0.down_sample_steps == 3
    assert set(ch) == {"rgb_0", "robot0_eef_pos", "robot0_eef_wrench_left"}


def test_factor_one_is_noop_and_span_in_demo_time_is_preserved():
    s = _sparse(); scale_obs_windows_by_slow_down(s, 1)
    assert s.rgb_0.down_sample_steps == 4
    # window span in wall time = demo span * factor
    s = _sparse(); f = 2.0; scale_obs_windows_by_slow_down(s, f)
    assert (5 - 1) * s.rgb_0.down_sample_steps == f * (5 - 1) * 4
