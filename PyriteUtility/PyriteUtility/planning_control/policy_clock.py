"""Elapsed-time-since-policy-start, in the time base the policy was trained on.

Training measures the ``elapsed_time_0`` observation in *demonstration time*: seconds
of the recorded demo since the policy would have started (see
``sampler_umift.elapsed_time_since_policy_start``). At inference the robot replays the
policy's action chunks stretched by ``slow_down_factor`` (the runner spaces action steps
``raw_time_step_s * slow_down_factor`` apart), so one wall-clock second of motion is only
``1 / slow_down_factor`` demo seconds. Feeding raw wall-clock time would make the policy
think it is running late by exactly that factor.

Periodic laser scans are different. In scan-collected demos the pause is recorded in real
time (the demo clock keeps running while the operator is away), so it appears in the
training timeline at full length. The pause is therefore added back *unscaled*:

    elapsed = (wall - paused) / slow_down_factor + paused
"""


class PolicyElapsedClock:
    def __init__(self, slow_down_factor: float = 1.0):
        slow_down_factor = float(slow_down_factor)
        if not slow_down_factor > 0.0:
            raise ValueError(f"slow_down_factor must be > 0, got {slow_down_factor}")
        self.slow_down_factor = slow_down_factor
        self._start_s = None
        self._paused_s = 0.0

    def start(self, now_s: float) -> None:
        """Mark policy start (the first query of an episode). Clears recorded pauses."""
        self._start_s = float(now_s)
        self._paused_s = 0.0

    def add_pause(self, duration_s: float) -> None:
        """Record wall time during which the robot was stopped for a scan."""
        if self._start_s is None:
            raise RuntimeError("PolicyElapsedClock.add_pause() called before start()")
        self._paused_s += max(0.0, float(duration_s))

    @property
    def paused_s(self) -> float:
        return self._paused_s

    def elapsed_s(self, now_s: float) -> float:
        """Demo-time seconds since policy start, as of hardware time ``now_s``."""
        if self._start_s is None:
            raise RuntimeError("PolicyElapsedClock.elapsed_s() called before start()")
        wall = float(now_s) - self._start_s
        moving = max(0.0, wall - self._paused_s)
        return moving / self.slow_down_factor + self._paused_s


def scale_obs_windows_by_slow_down(sparse_obs_meta, slow_down_factor: float) -> dict:
    """Stretch every observation history window by ``slow_down_factor``.

    ``shape_meta.sample.obs.sparse.<key>`` = {horizon, down_sample_steps}: a window is
    ``horizon`` samples spaced ``down_sample_steps`` apart, i.e. a span of demo time.
    At inference the robot moves ``slow_down_factor`` times slower in wall time (action
    steps are spaced ``raw_time_step_s * slow_down_factor`` apart), so the same *demo-time*
    window needs ``slow_down_factor`` times more wall-clock samples. Without this the
    policy sees its own history in slow motion.

    Only keys with ``horizon > 1`` are touched (``elapsed_time_0``, scan heatmaps have
    horizon 1). Call this BEFORE the per-sensor rate corrections (those multiply ``ds``
    again). Mutates ``sparse_obs_meta`` in place; returns {key: (old_ds, new_ds)}.
    """
    changes = {}
    for key in list(sparse_obs_meta.keys()):
        attr = sparse_obs_meta[key]
        if int(attr["horizon"]) <= 1:
            continue
        old = int(attr["down_sample_steps"])
        new = max(1, int(round(old * float(slow_down_factor))))
        attr["down_sample_steps"] = new
        changes[key] = (old, new)
    return changes
