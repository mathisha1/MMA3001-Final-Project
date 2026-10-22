import numpy as np
import pytest
from src.evaluation import week_bootstrap


def test_bootstrap_of_a_constant_error_has_no_spread():
    y = np.zeros(6)
    weeks = np.array(["a", "a", "b", "b", "c", "c"])
    summary, diff = week_bootstrap(y, {"m": np.ones(6), "ref": np.ones(6)}, weeks,
                                   ref="ref", n_boot=200, seed=1)
    row = summary.loc["m"]
    assert row["MAE"] == pytest.approx(1.0)
    assert row["MAE_lo"] == pytest.approx(1.0) and row["MAE_hi"] == pytest.approx(1.0)
    assert diff.loc["m", "MAE_diff"] == pytest.approx(0.0)


def test_bootstrap_interval_contains_the_point_estimate():
    y = np.zeros(8)
    weeks = np.repeat(["a", "b", "c", "d"], 2)
    p = np.array([0, 0, 3, 3, 0, 0, 1, 1], dtype=float)
    summary, _ = week_bootstrap(y, {"m": p, "ref": np.zeros(8)}, weeks,
                                ref="ref", n_boot=500, seed=2)
    row = summary.loc["m"]
    assert row["MAE"] == pytest.approx(1.0)
    assert row["MAE_lo"] <= row["MAE"] <= row["MAE_hi"]