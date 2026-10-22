"""Shared data preparation, feature builders and model wrappers.

Used by the robustness study, the Gaussian process study and the final test evaluation,
so every script fits exactly the same models."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.evaluation import TARGET, lookup_prediction
from src.features import add_calendar_features, add_term, load_term_calendar

ROOT = Path(__file__).resolve().parent.parent
PERIOD_LEVELS = ["non_teaching", "teaching", "closure"]          # first level is the reference
PERIOD_CODE = {"closure": 0, "non_teaching": 1, "teaching": 2}  # ordered code for tree and GP
TREE_DEPTH, TREE_LEAF = 6, 50                                    # chosen on validation in step 6


def load_prepared():
    """Load the hourly table with calendar and term features.

    Returns
    -------
    pandas.DataFrame
        One row per space and hour, with the 138 ambiguous held-value hours removed,
        local-time features added (hour, dow, is_weekend, ...) and the term label (period).
    """
    h = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    h = h[~((h["n_events"] == 0) & (h["snapshot"] > 0))]
    h = add_term(add_calendar_features(h), load_term_calendar(ROOT / "data" / "term_calendar.csv"))
    return h.reset_index(drop=True)


def one_hot(series, categories, prefix):
    """Indicator columns for a categorical column. The first category is the reference
    level and is dropped, so the columns are not perfectly collinear."""
    cat = pd.Series(pd.Categorical(series, categories=categories), index=series.index)
    return pd.get_dummies(cat, prefix=prefix, drop_first=True, dtype=float)


def linear_features(df, spaces, use_space=True, interaction=False):
    """Design matrix for linear regression: [space] + term + hour of day + weekend flag.

    With interaction=True, every term indicator is also multiplied by every hour
    indicator, so a teaching week can have a different daily shape from a break.
    """
    cols = []
    if use_space:
        cols.append(one_hot(df["space"], spaces, "space"))
    period = one_hot(df["period"], PERIOD_LEVELS, "period")     # period_teaching, period_closure
    hour = one_hot(df["hour"], list(range(24)), "hour")         # hour_1 ... hour_23
    cols += [period, hour, df[["is_weekend"]].astype(float)]
    if interaction:
        inter = {f"{p}_x_{q}": period[p] * hour[q] for p in period.columns for q in hour.columns}
        cols.append(pd.DataFrame(inter, index=df.index))
    return pd.concat(cols, axis=1)


def tree_features(df, spaces, use_space=True):
    """Numeric features for the tree: hour, day of week, ordered term code, [space indicators]."""
    X = pd.DataFrame({"hour": df["hour"].to_numpy(), "dow": df["dow"].to_numpy(),
                      "period_code": df["period"].map(PERIOD_CODE).to_numpy()})
    if use_space:
        for s in spaces:
            X[f"space_{s}"] = (df["space"] == s).astype(int).to_numpy()
    return X


def gp_features(df, spaces):
    """Numeric inputs for the Gaussian process.

    Hour of day and day of week are mapped onto circles (sin and cos), so 23:00 is close to
    00:00 and Sunday is close to Monday. The term is an ordered code (closure 0,
    non_teaching 1, teaching 2) and each space gets an indicator column.

    Returns
    -------
    pandas.DataFrame
        5 + len(spaces) columns.
    """
    hour_angle = 2 * np.pi * df["hour"].to_numpy(float) / 24
    dow_angle = 2 * np.pi * df["dow"].to_numpy(float) / 7
    cols = {"hour_sin": np.sin(hour_angle), "hour_cos": np.cos(hour_angle),
            "dow_sin": np.sin(dow_angle), "dow_cos": np.cos(dow_angle),
            "period_code": df["period"].map(PERIOD_CODE).to_numpy(float)}
    for s in spaces:
        cols[f"space_{s}"] = (df["space"] == s).to_numpy(float)
    return pd.DataFrame(cols)


def _spaces(train, test):
    """All space names seen in either table, sorted, so train and test columns line up."""
    return sorted(set(train["space"]) | set(test["space"]))


def predict_linear(train, test, use_space=True, interaction=False):
    """Fit linear regression on train and predict test. Negative predictions are set to 0."""
    spaces = _spaces(train, test)
    model = LinearRegression().fit(
        linear_features(train, spaces, use_space, interaction), train[TARGET])
    return np.clip(model.predict(linear_features(test, spaces, use_space, interaction)), 0, None)


def predict_tree(train, test, use_space=True):
    """Fit the chosen decision tree (depth TREE_DEPTH, leaf TREE_LEAF) and predict test."""
    spaces = _spaces(train, test)
    model = DecisionTreeRegressor(max_depth=TREE_DEPTH, min_samples_leaf=TREE_LEAF, random_state=0)
    model.fit(tree_features(train, spaces, use_space), train[TARGET])
    return model.predict(tree_features(test, spaces, use_space))


def model_zoo(use_space=True):
    """The calendar models, as functions fn(train, test) -> predictions (people).

    The lookup is the training mean of each (space, term, weekend, hour) group, or of each
    (term, weekend, hour) group when use_space is False. The names are the same either way."""
    keys = (["space"] if use_space else []) + ["period", "is_weekend", "hour"]
    return {
        "training mean": lambda tr, te: np.full(len(te), tr[TARGET].mean()),
        "lookup": lambda tr, te: lookup_prediction(tr, keys, te, tr[TARGET].mean()),
        "linear": lambda tr, te: predict_linear(tr, te, use_space),
        "tree": lambda tr, te: predict_tree(tr, te, use_space),
    }