"""Calendar features and the time-ordered train/validation/test split."""
import pandas as pd

LOCAL_TZ = "Australia/Melbourne"                       # sensors sit in Melbourne
TRAIN_END = pd.Timestamp("2025-01-01", tz=LOCAL_TZ)    # train: before this
VAL_END = pd.Timestamp("2025-09-01", tz=LOCAL_TZ)      # validation: before this, test after


def add_calendar_features(df):
    """Add local-time calendar columns to a table with a UTC 'hour_utc' column.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'hour_utc' (timezone-aware UTC timestamps).

    Returns
    -------
    pandas.DataFrame
        Copy with local_time, hour (0-23), dow (0 = Monday), hour_of_week (0-167),
        month (1-12) and is_weekend (0/1). Daylight saving is handled by the
        timezone conversion.
    """
    out = df.copy()
    local = out["hour_utc"].dt.tz_convert(LOCAL_TZ)
    out["local_time"] = local
    out["hour"] = local.dt.hour
    out["dow"] = local.dt.dayofweek
    out["hour_of_week"] = out["dow"] * 24 + out["hour"]
    out["month"] = local.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out


def split_by_time(df):
    """Split by local time into train, validation and test (never shuffled).

    Parameters
    ----------
    df : pandas.DataFrame
        Output of add_calendar_features.

    Returns
    -------
    dict of pandas.DataFrame
        Keys 'train', 'val', 'test'.
    """
    t = df["local_time"]
    return {"train": df[t < TRAIN_END],
            "val": df[(t >= TRAIN_END) & (t < VAL_END)],
            "test": df[t >= VAL_END]}

PERIODS = {"teaching", "non_teaching", "closure"}   # allowed labels in the term calendar


def load_term_calendar(path, first="2023-11-01", last="2026-04-30"):
    """Read the hand-made term calendar into one label per local calendar day.

    Parameters
    ----------
    path : pathlib.Path
        CSV with columns start, end, period. Dates are local (YYYY-MM-DD) and
        inclusive. Text after # on a line is ignored. Where rows overlap, the
        later row wins.
    first, last : str
        First and last day the calendar must be able to label.

    Returns
    -------
    pandas.Series
        Index is the date as 'YYYY-MM-DD' text, value is the period label (missing
        for days that no row covers).

    Raises
    ------
    ValueError
        If a period label is not in PERIODS, or a row ends before it starts.
    """
    cal = pd.read_csv(path, comment="#", parse_dates=["start", "end"])
    cal["period"] = cal["period"].str.strip()            # remove spaces left before a # comment
    bad = set(cal["period"]) - PERIODS
    if bad:
        raise ValueError(f"unknown period labels {sorted(bad)}; use {sorted(PERIODS)}")
    if (cal["end"] < cal["start"]).any():
        raise ValueError("a calendar row ends before it starts")
    days = pd.date_range(first, last, freq="D")
    label = pd.Series(None, index=days.strftime("%Y-%m-%d"), dtype="object")
    for _, row in cal.iterrows():                        # later rows override earlier ones
        label[(days >= row["start"]) & (days <= row["end"])] = row["period"]
    return label


def add_term(df, calendar):
    """Add a 'period' column (teaching, non_teaching or closure) from the local date.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of add_calendar_features (needs the tz-aware local_time column).
    calendar : pandas.Series
        Output of load_term_calendar.

    Returns
    -------
    pandas.DataFrame
        Copy with the period column added.

    Raises
    ------
    ValueError
        If any row falls on a date the calendar does not label.
    """
    out = df.copy()
    day = out["local_time"].dt.strftime("%Y-%m-%d")      # local date, so daylight saving edges are right
    out["period"] = day.map(calendar)
    missing = out["period"].isna()
    if missing.any():
        raise ValueError("calendar does not label these dates: "
                         f"{sorted(day[missing].unique())[:5]} ...")
    return out