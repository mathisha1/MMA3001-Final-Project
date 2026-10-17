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