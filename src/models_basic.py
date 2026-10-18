"""Step 6: linear regression and decision tree on calendar features.

Hyperparameters are chosen on the validation period. The test period is not used here;
it is kept for the final evaluation."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.evaluation import TARGET, metrics, lookup_prediction, weekday_day
from src.features import add_calendar_features, add_term, load_term_calendar, split_by_time

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


h = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
h = h[~((h["n_events"] == 0) & (h["snapshot"] > 0))]       # drop the ambiguous held-value hours
h = add_term(add_calendar_features(h), load_term_calendar(ROOT / "data" / "term_calendar.csv"))
parts = split_by_time(h)
train, val = parts["train"], parts["val"]                  # the test part is deliberately unused
y_tr, y_va = train[TARGET], val[TARGET]

SPACES = sorted(h["space"].unique())
PERIOD_LEVELS = ["non_teaching", "teaching", "closure"]    # first level is the reference
PERIOD_CODE = {"closure": 0, "non_teaching": 1, "teaching": 2}   # ordered code for the tree


def one_hot(series, categories, prefix):
    """Indicator columns for a categorical column. The first category is the reference
    level and is dropped, so the columns are not perfectly collinear."""
    cat = pd.Series(pd.Categorical(series, categories=categories), index=series.index)
    return pd.get_dummies(cat, prefix=prefix, drop_first=True, dtype=float)


def linear_features(df, kind):
    """Design matrix for linear regression.

    kind 'hour+weekend': space + term + hour of day + weekend flag.
    kind 'hour-of-week': space + term + one indicator per hour of the week (0-167)."""
    cols = [one_hot(df["space"], SPACES, "space"), one_hot(df["period"], PERIOD_LEVELS, "period")]
    if kind == "hour+weekend":
        cols += [one_hot(df["hour"], list(range(24)), "hour"), df[["is_weekend"]].astype(float)]
    else:
        cols.append(one_hot(df["hour_of_week"], list(range(168)), "how"))
    return pd.concat(cols, axis=1)


def tree_features(df):
    """Numeric features for the tree: hour, day of week, ordered term code, space indicators."""
    X = pd.DataFrame({"hour": df["hour"], "dow": df["dow"],
                      "period_code": df["period"].map(PERIOD_CODE)})
    for s in SPACES:
        X[f"space_{s}"] = (df["space"] == s).astype(int)
    return X


results = {}                      # model name -> (validation predictions, fit seconds)
g = y_tr.mean()
results["constant: training mean"] = (np.full(len(val), g), 0.0)
results["lookup: space + weekend + hour (no term)"] = (
    lookup_prediction(train, ["space", "is_weekend", "hour"], val, g), 0.0)
results["lookup: space + term + weekend + hour"] = (
    lookup_prediction(train, ["space", "period", "is_weekend", "hour"], val, g), 0.0)

# ---- linear regression
for kind in ["hour+weekend", "hour-of-week"]:
    X_tr, X_va = linear_features(train, kind), linear_features(val, kind)
    t0 = time.perf_counter()
    lin = LinearRegression().fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    raw = lin.predict(X_va)
    log(f"linear ({kind}): {X_tr.shape[1]} columns, "
        f"{(raw < 0).mean():.1%} of validation predictions below zero (set to 0)")
    results[f"linear: space + term + {kind}"] = (np.clip(raw, 0, None), fit_s)
    if kind == "hour+weekend":
        coefs = pd.Series(lin.coef_, index=X_tr.columns)
        keep = [c for c in coefs.index if c.startswith(("space_", "period_")) or c == "is_weekend"]
        log("coefficients, in people versus the reference level (reference space is the first "
            "alphabetically, reference term is non_teaching):")
        log(coefs[keep].round(3).to_string())

# ---- decision tree: grid over depth and minimum leaf size, chosen on validation RMSE
Xt_tr, Xt_va = tree_features(train), tree_features(val)
rows, fitted = [], {}
for depth in [2, 3, 4, 6, 8, 10, 14]:
    for leaf in [10, 50, 200]:
        t0 = time.perf_counter()
        m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf,
                                  random_state=0).fit(Xt_tr, y_tr)
        fit_s = time.perf_counter() - t0
        p_tr, p_va = m.predict(Xt_tr), m.predict(Xt_va)
        a, b = metrics(y_tr, p_tr), metrics(y_va, p_va)
        rows.append({"depth": depth, "min_leaf": leaf, "leaves": m.get_n_leaves(),
                     "train_MAE": a["MAE"], "val_MAE": b["MAE"],
                     "train_RMSE": a["RMSE"], "val_RMSE": b["RMSE"], "fit_s": fit_s})
        fitted[(depth, leaf)] = (m, p_va, fit_s)
grid = pd.DataFrame(rows)
log("\n== TREE GRID (train error keeps falling; validation error shows where overfitting starts) ==")
log(grid.round(3).to_string(index=False))

best = grid.loc[grid["val_RMSE"].idxmin()]
key = (int(best["depth"]), int(best["min_leaf"]))
tree, p_best, fit_best = fitted[key]
log(f"\nchosen tree: depth {key[0]}, min leaf {key[1]} (lowest validation RMSE)")
log("feature importance:")
log(pd.Series(tree.feature_importances_, index=Xt_tr.columns).sort_values(ascending=False)
    .round(3).to_string())
results[f"tree: depth {key[0]}, min leaf {key[1]}"] = (p_best, fit_best)


def table(mask):
    """Metrics for every model on the validation rows picked by mask."""
    out = [{"model": n, **metrics(y_va[mask], p[mask]), "fit_s": s}
           for n, (p, s) in results.items()]
    return pd.DataFrame(out).round(3).to_string(index=False)


day = weekday_day(val)
log(f"\n== VALIDATION, ALL HOURS (mean actual {y_va.mean():.3f} people) ==")
log(table(np.ones(len(val), bool)))
log(f"\n== VALIDATION, WEEKDAY 9AM-5PM ({day.sum():,} hours) ==")
log(table(day))

(OUTDIR / "models_basic.txt").write_text("\n".join(report), encoding="utf-8")
print("\nsaved outputs/models_basic.txt")