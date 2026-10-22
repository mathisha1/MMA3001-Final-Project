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

def week_bootstrap(y, preds, weeks, ref, n_boot=1000, seed=0):
    """Block bootstrap over whole weeks, to see how much a score would wobble.

    Hourly errors in the same week are correlated, so single hours cannot be resampled
    independently. Instead whole weeks are redrawn at random, with replacement, n_boot
    times, and every model is scored on the same redraws.

    Parameters
    ----------
    y : array-like
        Actual values, one per hour.
    preds : dict
        Model name -> predictions, same length as y.
    weeks : array-like
        Week label for each hour.
    ref : str
        Name in preds that the other models are compared against.
    n_boot : int
        Number of redraws.
    seed : int
        Random seed, so the run is repeatable.

    Returns
    -------
    summary : pandas.DataFrame
        Per model: MAE and RMSE with 2.5th and 97.5th percentile bounds (_lo, _hi).
    diff : pandas.DataFrame
        Per model other than ref: MAE and RMSE minus the reference's (negative means better
        than ref) with bounds, and p_better_MAE / p_better_RMSE, the share of redraws in
        which the model beat ref.
    """
    y = np.asarray(y, dtype=float)
    codes, uniq = pd.factorize(pd.Series(np.asarray(weeks)))
    n_weeks = len(uniq)
    names = list(preds)
    err = {n: y - np.asarray(preds[n], dtype=float) for n in names}
    abs_sum = np.column_stack([np.bincount(codes, weights=np.abs(err[n]), minlength=n_weeks)
                               for n in names])
    sq_sum = np.column_stack([np.bincount(codes, weights=err[n] ** 2, minlength=n_weeks)
                              for n in names])
    n_obs = np.bincount(codes, minlength=n_weeks).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_weeks, np.full(n_weeks, 1.0 / n_weeks), size=n_boot).astype(float)
    total = draws @ n_obs
    mae_b = (draws @ abs_sum) / total[:, None]
    rmse_b = np.sqrt((draws @ sq_sum) / total[:, None])
    point_mae = abs_sum.sum(axis=0) / n_obs.sum()
    point_rmse = np.sqrt(sq_sum.sum(axis=0) / n_obs.sum())
    lo, hi = 2.5, 97.5
    summary = pd.DataFrame({
        "MAE": point_mae, "MAE_lo": np.percentile(mae_b, lo, axis=0),
        "MAE_hi": np.percentile(mae_b, hi, axis=0),
        "RMSE": point_rmse, "RMSE_lo": np.percentile(rmse_b, lo, axis=0),
        "RMSE_hi": np.percentile(rmse_b, hi, axis=0)}, index=names)
    r = names.index(ref)
    rows = {}
    for j, n in enumerate(names):
        if n == ref:
            continue
        dm, dr = mae_b[:, j] - mae_b[:, r], rmse_b[:, j] - rmse_b[:, r]
        rows[n] = {"MAE_diff": point_mae[j] - point_mae[r],
                   "MAE_diff_lo": np.percentile(dm, lo), "MAE_diff_hi": np.percentile(dm, hi),
                   "p_better_MAE": float((dm < 0).mean()),
                   "RMSE_diff": point_rmse[j] - point_rmse[r],
                   "RMSE_diff_lo": np.percentile(dr, lo), "RMSE_diff_hi": np.percentile(dr, hi),
                   "p_better_RMSE": float((dr < 0).mean())}
    return summary, pd.DataFrame(rows).T