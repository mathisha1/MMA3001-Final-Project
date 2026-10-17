"""Numerical integration helpers: sampling a step function, trapezoid and Romberg."""
import numpy as np


def sample_step(t, v, times):
    """Value of a step function at the requested times.

    Parameters
    ----------
    t : numpy.ndarray
        Sorted event times (s).
    v : numpy.ndarray
        Value set at each event, held until the next event.
    times : numpy.ndarray
        Times to sample, any shape, all at or after t[0]. A time exactly on an
        event gets the new value (right-continuous).

    Returns
    -------
    numpy.ndarray
        Same shape as times.
    """
    i = np.searchsorted(t, times, side="right") - 1   # index of the event in force
    return v[i]


def trapezoid(f, h):
    """Composite trapezoid rule on equally spaced samples.

    Parameters
    ----------
    f : numpy.ndarray
        Samples, shape (..., n + 1). Leading dimensions are independent problems.
    h : float
        Spacing between samples.

    Returns
    -------
    numpy.ndarray
        Integral estimate for each leading index.
    """
    return h * (0.5 * f[..., 0] + f[..., 1:-1].sum(axis=-1) + 0.5 * f[..., -1])


def romberg_table(f, length):
    """Romberg table built from samples on the finest grid.

    Parameters
    ----------
    f : numpy.ndarray
        Samples, shape (..., n + 1), where n is a power of 2. Coarser grids reuse
        every 2nd, 4th, ... sample, so no extra evaluations are needed.
    length : float
        Length of the integration interval.

    Returns
    -------
    list of list of numpy.ndarray
        R[k][j] uses 2**k intervals. R[k][0] is the trapezoid rule, R[k][1] is
        Simpson's rule, R[k][2] is Boole's rule, and so on.
    """
    n = f.shape[-1] - 1
    K = int(round(np.log2(n)))
    if 2 ** K != n:
        raise ValueError("number of intervals must be a power of 2")
    R = [[None] * (k + 1) for k in range(K + 1)]
    for k in range(K + 1):
        g = f[..., :: 2 ** (K - k)]                       # every (2**(K-k))th sample
        R[k][0] = trapezoid(g, length / 2 ** k)
        for j in range(1, k + 1):                         # Richardson extrapolation
            R[k][j] = (4 ** j * R[k][j - 1] - R[k - 1][j - 1]) / (4 ** j - 1)
    return R