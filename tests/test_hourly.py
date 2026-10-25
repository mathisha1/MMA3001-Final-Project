import numpy as np
import pytest
from src.hourly import hourly_time_weighted


def test_partial_hour_is_time_weighted():
    # 3 people for 40 minutes, then empty: mean over the hour is 3 * 40/60 = 2
    t = np.array([0.0, 2400.0, 7200.0])
    v = np.array([3.0, 0.0, 0.0])
    out = hourly_time_weighted(t, v, np.array([0.0, 3600.0]))
    assert out == pytest.approx([2.0, 0.0])


def test_value_held_across_several_hours():
    t = np.array([0.0, 10800.0])
    v = np.array([2.0, 0.0])
    out = hourly_time_weighted(t, v, np.array([0.0, 3600.0, 7200.0]))
    assert out == pytest.approx([2.0, 2.0, 2.0])


def test_change_inside_an_hour():
    # 4 people for half an hour, then 1: (4*1800 + 1*1800) / 3600 = 2.5
    t = np.array([0.0, 1800.0, 3600.0])
    v = np.array([4.0, 1.0, 1.0])
    out = hourly_time_weighted(t, v, np.array([0.0]))
    assert out == pytest.approx([2.5])

import numpy as np
from src.hourly import HOUR_S


def _brute_force(t, v, start):
    """Mean headcount in [start, start + 3600) by walking through every event."""
    end, total = start + HOUR_S, 0.0
    for i in range(len(t)):
        seg_start = max(t[i], start)
        seg_end = min(t[i + 1] if i + 1 < len(t) else end, end)
        if seg_end > seg_start:
            total += v[i] * (seg_end - seg_start)
    return total / HOUR_S


def test_matches_brute_force_on_random_events():
    rng = np.random.default_rng(0)
    t = np.sort(rng.choice(np.arange(0, 20000, 2), size=300, replace=False)).astype(float)
    v = rng.integers(0, 6, size=300).astype(float)
    starts = t[0] + np.array([0.0, 1234.0, 4000.0, 9000.0])
    fast = hourly_time_weighted(t, v, starts)
    slow = [_brute_force(t, v, s) for s in starts]
    assert fast == pytest.approx(slow, abs=1e-9)