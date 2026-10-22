"""Step 6b: does adding term x hour interaction columns improve linear regression?

Validation data only; the test period is never touched.
Rule, stated before the test period is scored: the variant with the lower validation RMSE
(all hours) becomes the primary linear model. A difference under 0.002 counts as a tie and
goes to the simpler additive model."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.evaluation import TARGET, metrics, weekday_day
from src.features import LOCAL_TZ, VAL_END, split_by_time
from src.modeling import linear_features, load_prepared, predict_linear

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


def main():
    """Compare the additive and interaction linear models and apply the stated rule."""
    pd.set_option("display.width", 200)
    OUTDIR.mkdir(exist_ok=True)
    h = load_prepared()
    parts = split_by_time(h)
    train, val = parts["train"], parts["val"]          # parts["test"] is never used
    y, day = val[TARGET].to_numpy(), weekday_day(val)
    spaces = sorted(set(train["space"]) | set(val["space"]))

    rows = []
    for name, inter in (("additive", False), ("interaction", True)):
        X_tr = linear_features(train, spaces, True, inter)
        X_va = linear_features(val, spaces, True, inter)
        t0 = time.perf_counter()
        model = LinearRegression().fit(X_tr, train[TARGET])
        fit_s = time.perf_counter() - t0
        raw = model.predict(X_va)
        p = np.clip(raw, 0, None)                      # people cannot be negative
        m, md = metrics(y, p), metrics(y[day], p[day])
        rows.append({"variant": name, "columns": X_tr.shape[1], "MAE": m["MAE"],
                     "RMSE": m["RMSE"], "R2": m["R2"], "MAE_day": md["MAE"],
                     "RMSE_day": md["RMSE"], "pct_negative_raw": float((raw < 0).mean() * 100),
                     "fit_s": fit_s})
    res = pd.DataFrame(rows).set_index("variant")
    log("== VALIDATION (train period -> validation period) ==")
    log(res.round(3).to_string())

    gain = res.loc["additive", "RMSE"] - res.loc["interaction", "RMSE"]
    winner = "interaction" if gain > 0.002 else "additive"
    log(f"\nRMSE gain from the interaction: {gain:.4f}")
    log(f"RULE RESULT: LINEAR_PRIMARY = \"{winner}\"")

    # supporting check: is the same ordering seen at other forecast origins?
    pool = h[h["local_time"] < VAL_END]
    assert pool["local_time"].max() < VAL_END           # guard: the test period is excluded
    origins = ([pd.Timestamp(2024, m, 1, tz=LOCAL_TZ) for m in range(7, 13)]
               + [pd.Timestamp(2025, m, 1, tz=LOCAL_TZ) for m in range(1, 7)])
    rows = []
    for origin in origins:
        tr = pool[pool["local_time"] < origin]
        te = pool[(pool["local_time"] >= origin)
                  & (pool["local_time"] < origin + pd.DateOffset(months=3))]
        yt = te[TARGET].to_numpy()
        a = metrics(yt, predict_linear(tr, te, True, False))
        b = metrics(yt, predict_linear(tr, te, True, True))
        rows.append({"origin": f"{origin:%Y-%m}", "MAE_additive": a["MAE"],
                     "MAE_interaction": b["MAE"], "RMSE_additive": a["RMSE"],
                     "RMSE_interaction": b["RMSE"]})
    r = pd.DataFrame(rows)
    log("\n== SUPPORTING CHECK: ROLLING ORIGINS (next 3 months after each origin) ==")
    log(r.round(3).to_string(index=False))
    log(f"interaction has the lower RMSE in {(r['RMSE_interaction'] < r['RMSE_additive']).sum()} "
        f"of {len(r)} origins (supporting evidence only; the rule uses the validation row)")

    (OUTDIR / "linear_variants.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nsaved outputs/linear_variants.txt")


if __name__ == "__main__":
    main()