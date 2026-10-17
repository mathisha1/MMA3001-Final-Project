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