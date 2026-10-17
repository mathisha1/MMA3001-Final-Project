import numpy as np
import pytest
from src.integration import romberg_table, sample_step

X = np.linspace(0.0, 1.0, 9)   # 8 intervals on [0, 1]


def test_trapezoid_matches_hand_calculation():
    # integral of x^2 is 1/3; trapezoid error is h^2/12 * (f'(1) - f'(0)) = 1/384 at h = 1/8
    R = romberg_table(X ** 2, 1.0)
    assert R[0][0] == pytest.approx(0.5)
    assert R[3][0] == pytest.approx(0.3359375)


def test_simpson_exact_for_quadratic():
    assert romberg_table(X ** 2, 1.0)[3][1] == pytest.approx(1 / 3)


def test_boole_exact_for_quartic():
    assert romberg_table(X ** 4, 1.0)[3][2] == pytest.approx(0.2)


def test_batch_of_problems():
    R = romberg_table(np.vstack([X ** 2, X ** 4]), 1.0)
    assert R[3][2] == pytest.approx([1 / 3, 0.2])


def test_sample_step_holds_value_and_jumps_at_event():
    t = np.array([0.0, 10.0, 20.0])
    v = np.array([1.0, 2.0, 3.0])
    out = sample_step(t, v, np.array([0.0, 5.0, 10.0, 15.0, 20.0]))
    assert out == pytest.approx([1.0, 1.0, 2.0, 2.0, 3.0])