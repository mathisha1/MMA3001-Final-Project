"""Shared evaluation helpers: error metrics and group-mean lookup baselines."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "time_weighted"      # exact mean headcount in the hour (people)


def metrics(y, p):
    """MAE and RMSE (people) and R2 for predictions p against truth y."""
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p)}


def lookup_prediction(train, keys, df, fallback):
    """Predict each row of df as the training mean of its group (keys).
    Groups unseen in training get the fallback value."""
    table = train.groupby(keys)[TARGET].mean().rename("pred").reset_index()
    return df[keys].merge(table, on=keys, how="left")["pred"].fillna(fallback).to_numpy()


def coverage(train, keys, df):
    """Share of rows in df whose group (keys) also appears in train."""
    seen = train[keys].drop_duplicates().assign(_seen=True)
    return float(df[keys].merge(seen, on=keys, how="left")["_seen"].notna().mean())


def weekday_day(df):
    """Boolean mask for weekday 9am to 5pm rows (needs dow and hour columns)."""
    return ((df["dow"] < 5) & (df["hour"] >= 9) & (df["hour"] < 17)).to_numpy()