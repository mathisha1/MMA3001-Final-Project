import numpy as np
import pandas as pd
import pytest
from src.evaluation import TARGET
from src.modeling import gp_features, model_zoo, predict_linear, predict_tree


def _toy():
    """Two spaces, every hour, weekday and weekend, all in teaching term."""
    rows = []
    for space in ["a", "b"]:
        for hour in range(24):
            for weekend in (0, 1):
                rows.append({"space": space, "hour": hour, "dow": 5 if weekend else 1,
                             "is_weekend": weekend, "period": "teaching",
                             TARGET: float(hour % 5) + weekend})
    return pd.DataFrame(rows)


def test_linear_predictions_are_nonnegative_and_aligned():
    df = _toy()
    p = predict_linear(df, df)
    assert len(p) == len(df) and p.min() >= 0


def test_models_without_space_accept_an_unseen_space():
    tr = _toy()
    te = tr.assign(space="z")
    for fn in (predict_linear, predict_tree):
        assert len(fn(tr, te, use_space=False)) == len(te)


def test_gp_features_width_and_hour_wraps_around():
    df = _toy()
    X = gp_features(df, ["a", "b"])
    assert X.shape == (len(df), 5 + 2)
    assert np.allclose(X["hour_sin"] ** 2 + X["hour_cos"] ** 2, 1.0)

    def point(hr):
        return X[(df["hour"] == hr).to_numpy()].iloc[0][["hour_sin", "hour_cos"]].to_numpy(float)

    assert np.linalg.norm(point(23) - point(0)) < np.linalg.norm(point(23) - point(12))


def test_model_names_match_with_and_without_space():
    assert list(model_zoo(True)) == list(model_zoo(False))


def test_lookup_uses_training_mean_for_an_unseen_group():
    tr = _toy()
    te = tr.assign(hour=99).head(5)           # hour 99 never appears in training
    p = model_zoo(True)["lookup"](tr, te)
    assert p == pytest.approx(np.full(5, tr[TARGET].mean()))