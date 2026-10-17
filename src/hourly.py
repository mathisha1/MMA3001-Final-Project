"""Helpers for turning occupancy event logs into regular hourly series.

The sensor logs a new headcount whenever it changes, so between two events the
headcount is assumed constant (a 'step function'). The exact average headcount in
an hour is then the area under the steps divided by the hour length.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HOUR_S = 3600.0   # seconds in an hour


def load_events(path):
    """Load occ.parquet as a compact table sorted by space, then time.

    Parameters
    ----------
    path : pathlib.Path
        Path to occ.parquet (needs columns floorspaceid, headcount, collecteddate).

    Returns
    -------
    pandas.DataFrame
        Columns: space (first 8 characters of floorspaceid), t (seconds since
        1970-01-01 UTC, float64), headcount (float32).
    """
    # distinct spaces first, so every batch shares one category list
    spaces = pq.read_table(path, columns=["floorspaceid"]).column(0).unique().to_pylist()
    cats = [s[:8] for s in spaces]
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1_000_000, columns=["floorspaceid", "headcount", "collecteddate"]):
        df = batch.to_pandas()
        t = pd.to_datetime(df["collecteddate"], format="ISO8601", utc=True)
        parts.append(pd.DataFrame({
            "space": pd.Categorical(df["floorspaceid"].str[:8], categories=cats),
            "t": (t - epoch).dt.total_seconds(),        # timezone-safe seconds
            "headcount": pd.to_numeric(df["headcount"]).astype("float32"),
        }))
        print(f"  loaded {sum(len(p) for p in parts):,} rows", flush=True)
    occ = pd.concat(parts, ignore_index=True)
    return occ.sort_values(["space", "t"], kind="stable").reset_index(drop=True)


def cumulative_integral(t, v):
    """Area under the step function from t[0] to each event time.

    Parameters
    ----------
    t : numpy.ndarray
        Sorted event times in seconds.
    v : numpy.ndarray
        Headcount logged at each event. It is held until the next event.

    Returns
    -------
    numpy.ndarray
        Same length as t. Units are people x seconds.
    """
    return np.concatenate(([0.0], np.cumsum(v[:-1] * np.diff(t))))


def integral_up_to(times, t, v, cum):
    """Area under the step function from t[0] to each requested time.

    Requested times must lie between t[0] and t[-1].
    """
    i = np.searchsorted(t, times, side="right") - 1   # index of the event in force
    return cum[i] + v[i] * (times - t[i])


def hourly_time_weighted(t, v, starts):
    """Exact mean headcount in each hour [start, start + 3600 s).

    Parameters
    ----------
    t, v : numpy.ndarray
        Event times (s) and headcounts.
    starts : numpy.ndarray
        Start time of each hour in seconds.

    Returns
    -------
    numpy.ndarray
        Mean headcount (people) in each hour.
    """
    cum = cumulative_integral(t, v)
    area = integral_up_to(starts + HOUR_S, t, v, cum) - integral_up_to(starts, t, v, cum)
    return area / HOUR_S


def hourly_series(t, v, space):
    """Build the hourly table for one space.

    Returns a DataFrame with one row per whole hour between the first and last event:
    time_weighted (exact mean), event_mean (plain mean of events in the hour, NaN if
    none), snapshot (value in force at the start of the hour), peak (highest value
    seen in the hour), n_events, and stale_s (seconds since the last event at the
    start of the hour, which is how long a value has been held).
    """
    first = np.ceil(t[0] / HOUR_S) * HOUR_S
    last = np.floor(t[-1] / HOUR_S) * HOUR_S
    starts = np.arange(first, last, HOUR_S)
    i = np.searchsorted(t, starts, side="right") - 1
    # plain statistics of the events that fall inside each hour
    ev = (pd.Series(v).groupby(np.floor(t / HOUR_S).astype("int64"))
          .agg(["mean", "max", "size"])
          .reindex((starts / HOUR_S).astype("int64")))
    snapshot = v[i]
    return pd.DataFrame({
        "space": space,
        "hour_utc": pd.to_datetime(starts, unit="s", utc=True),
        "time_weighted": hourly_time_weighted(t, v, starts),
        "event_mean": ev["mean"].to_numpy(),
        "snapshot": snapshot,
        "peak": np.fmax(ev["max"].to_numpy(), snapshot),
        "n_events": ev["size"].fillna(0).astype("int32").to_numpy(),
        "stale_s": starts - t[i],
    })