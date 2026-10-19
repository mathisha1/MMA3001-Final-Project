"""Step 9: final evaluation on the held-out test period.

Frozen before the test period was scored:
  - features: hour of day, day of week, university term, space
  - lookup groups: space x term x weekend x hour
  - tree: depth 6, minimum leaf 50 (chosen on validation, outputs/models_basic.txt)
  - GP: constant x ARD-RBF + white noise on n = 2000 random training hours, three subsamples
    (seeds 0, 1, 2); one extra larger fit shows the cost/accuracy trade-off
  - linear: LINEAR_PRIMARY, set from outputs/linear_variants.txt by the rule stated there

Usage
-----
python -m src.final_test --dry-run   train on train, score validation; must reproduce earlier numbers
python -m src.final_test             train on train + validation, score test; refuses to run twice
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation import TARGET, metrics, week_bootstrap, weekday_day
from src.features import VAL_END, split_by_time
from src.gp_study import fit_gp, predict_gp
from src.modeling import gp_features, load_prepared, model_zoo, predict_linear

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
PROC = ROOT / "data" / "processed"

LINEAR_PRIMARY = "additive"      # "additive" or "interaction": set from outputs/linear_variants.txt
GP_N, GP_SEEDS = 2000, [0, 1, 2]
GP_EXTRA_N = 3000                # one extra, larger fit (final run only)

# validation numbers the dry run must reproduce: (MAE, RMSE, tolerance)
EXPECTED = {"training mean": (0.775, 1.210, 0.0015), "lookup": (0.497, 1.072, 0.0015),
            "linear (additive)": (0.559, 1.067, 0.0015), "tree": (0.495, 1.044, 0.0015),
            "GP seed 0": (0.477, 1.042, 0.003), "GP seed 1": (0.512, 1.065, 0.003),
            "GP seed 2": (0.516, 1.095, 0.003)}

report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


def fit_all(train, test, spaces, extra):
    """Fit every model on train and predict test.

    Returns
    -------
    preds : dict
        Model name -> predicted headcount (people), non-negative.
    mus, sds : dict
        GP name -> raw predictive mean and standard deviation (for interval checks).
    secs : dict
        Model name -> fit time in seconds.
    """
    zoo = model_zoo(True)
    preds, mus, sds, secs = {}, {}, {}, {}

    def run(name, fn):
        t0 = time.perf_counter()
        preds[name] = fn()
        secs[name] = time.perf_counter() - t0
        print(f"  fitted {name}: {secs[name]:.1f} s", flush=True)

    run("training mean", lambda: zoo["training mean"](train, test))
    run("lookup", lambda: zoo["lookup"](train, test))
    run("linear (additive)", lambda: predict_linear(train, test, True, False))
    run("linear (+term x hour)", lambda: predict_linear(train, test, True, True))
    run("tree", lambda: zoo["tree"](train, test))

    X_tr = gp_features(train, spaces).to_numpy()
    X_te = gp_features(test, spaces).to_numpy()
    y_tr = train[TARGET].to_numpy()
    runs = [(GP_N, s) for s in GP_SEEDS] + ([(GP_EXTRA_N, 0)] if extra else [])
    for n, seed in runs:
        name = f"GP seed {seed}" if n == GP_N else f"GP n={n}"
        idx = np.sort(np.random.default_rng(seed).choice(len(train), size=n, replace=False))
        try:
            gp, fit_s, n_fail, n_bound = fit_gp(X_tr[idx], y_tr[idx], seed)
        except MemoryError:
            print(f"  {name}: not enough memory, skipped", flush=True)
            continue
        mu, sd = predict_gp(gp, X_te)
        mus[name], sds[name], preds[name] = mu, sd, np.clip(mu, 0, None)
        secs[name] = fit_s
        print(f"  fitted {name}: {fit_s:.1f} s ({n_fail} optimiser warnings, "
              f"{n_bound} hyperparameters at a bound)", flush=True)
    return preds, mus, sds, secs


def metric_table(y, preds, mask):
    """MAE, RMSE, R2 and bias (mean predicted minus mean actual) per model on masked rows."""
    rows = []
    for name, p in preds.items():
        m = metrics(y[mask], p[mask])
        rows.append({"model": name, **m, "bias": float(p[mask].mean() - y[mask].mean())})
    return pd.DataFrame(rows).set_index("model")


def mae_by(groups, y, preds):
    """MAE per model within each group, plus hours and the mean actual headcount."""
    groups = np.asarray(groups, dtype=str)
    rows = {}
    for g in sorted(set(groups)):
        sel = groups == g
        rows[g] = {"hours": int(sel.sum()), "mean_actual": y[sel].mean(),
                   **{n: np.abs(y[sel] - p[sel]).mean() for n, p in preds.items()}}
    out = pd.DataFrame(rows).T
    out["hours"] = out["hours"].astype(int)
    return out


def week_labels(test):
    """ISO year-week label for every row (local time)."""
    iso = test["local_time"].dt.isocalendar()
    return (iso["year"].astype(int).astype(str) + "-" + iso["week"].astype(int).astype(str)).to_numpy()


def ci_text(df, col):
    """Format 'value [low, high]' for a column that has _lo and _hi companions."""
    return df.apply(lambda r: f"{r[col]:.3f} [{r[col + '_lo']:.3f}, {r[col + '_hi']:.3f}]", axis=1)


def main():
    """Run the dry run or the final evaluation and save the report and predictions."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run
    out_path = OUTDIR / ("final_dry_run.txt" if dry else "final_test.txt")
    if not dry and out_path.exists():
        sys.exit(f"{out_path.name} already exists: the test period has been scored once. Do not rerun.")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    OUTDIR.mkdir(exist_ok=True)

    h = load_prepared()
    parts = split_by_time(h)
    if dry:
        train, test = parts["train"], parts["val"]
        log("DRY RUN: train period -> validation period (the test period is not touched)")
    else:
        train, test = pd.concat([parts["train"], parts["val"]]), parts["test"]
        log("FINAL: train + validation -> test period (scored once)")
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    assert train["local_time"].max() < test["local_time"].min()
    if dry:
        assert test["local_time"].max() < VAL_END

    log(f"train: {len(train):,} hours, {train['local_time'].min().date()} to "
        f"{train['local_time'].max().date()}; test: {len(test):,} hours, "
        f"{test['local_time'].min().date()} to {test['local_time'].max().date()}, "
        f"spaces {sorted(test['space'].unique())}")
    log(f"mean actual headcount: train {train[TARGET].mean():.3f}, test {test[TARGET].mean():.3f}")

    spaces = sorted(h["space"].unique())
    y, day = test[TARGET].to_numpy(), weekday_day(test)
    preds, mus, sds, secs = fit_all(train, test, spaces, extra=not dry)
    primary = "linear (additive)" if LINEAR_PRIMARY == "additive" else "linear (+term x hour)"
    hp = {"training mean": preds["training mean"], "lookup": preds["lookup"],
          "linear": preds[primary], "tree": preds["tree"],
          "GP seed 0": preds["GP seed 0"], "GP seed 1": preds["GP seed 1"],
          "GP seed 2": preds["GP seed 2"]}
    log(f"primary linear model: {primary}")

    if dry:
        log("\n== CHECK: DOES THE DRY RUN REPRODUCE THE VALIDATION NUMBERS? ==")
        ok = True
        for name, (mae, rmse, tol) in EXPECTED.items():
            m = metrics(y, preds[name])
            good = bool(abs(m["MAE"] - mae) < tol and abs(m["RMSE"] - rmse) < tol)
            ok = ok and good
            log(f"{name}: MAE {m['MAE']:.3f} (expected {mae:.3f}), RMSE {m['RMSE']:.3f} "
                f"(expected {rmse:.3f}) {'OK' if good else 'MISMATCH'}")
        log("reproduction passed" if ok else "reproduction FAILED: do not run the final evaluation")

    log("\n== A. ALL HOURS ==")
    log(metric_table(y, preds, np.ones(len(y), bool)).round(3).to_string())
    log(f"\n== B. WEEKDAY 9AM-5PM ({day.sum():,} hours) ==")
    log(metric_table(y, preds, day).round(3).to_string())

    seeds = [f"GP seed {s}" for s in GP_SEEDS if f"GP seed {s}" in preds]
    rows = []
    for subset, mask in (("all hours", np.ones(len(y), bool)), ("weekday 9-5", day)):
        ms = [metrics(y[mask], preds[n][mask]) for n in seeds]
        rows.append({"subset": subset, "MAE": f"{np.mean([m['MAE'] for m in ms]):.3f} +/- "
                     f"{np.std([m['MAE'] for m in ms], ddof=1):.3f}",
                     "RMSE": f"{np.mean([m['RMSE'] for m in ms]):.3f} +/- "
                     f"{np.std([m['RMSE'] for m in ms], ddof=1):.3f}"})
    log(f"\n== C. GP OVER {len(seeds)} SUBSAMPLES OF n = {GP_N} (mean +/- standard deviation) ==")
    log(pd.DataFrame(rows).to_string(index=False))

    log("\n== D. BOOTSTRAP OVER WHOLE WEEKS (1000 redraws, 95% interval), reference = lookup ==")
    weeks = week_labels(test)
    for label, mask in (("all hours", np.ones(len(y), bool)), ("weekday 9am-5pm", day)):
        summ, diff = week_bootstrap(y[mask], {k: v[mask] for k, v in hp.items()},
                                    weeks[mask], ref="lookup", n_boot=1000, seed=0)
        log(f"\n-- {label}: score with interval --")
        log(pd.DataFrame({"MAE": ci_text(summ, "MAE"), "RMSE": ci_text(summ, "RMSE")}).to_string())
        log(f"-- {label}: difference from lookup (negative = better than lookup) --")
        log(pd.DataFrame({"MAE_diff": ci_text(diff, "MAE_diff"),
                          "p_better_MAE": diff["p_better_MAE"].round(2),
                          "RMSE_diff": ci_text(diff, "RMSE_diff"),
                          "p_better_RMSE": diff["p_better_RMSE"].round(2)}).to_string())

    log("\n== E. MAE BY SPACE ==")
    log(mae_by(test["space"], y, hp).round(3).to_string())
    log("\n== F. MAE BY TERM ==")
    log(mae_by(test["period"], y, hp).round(3).to_string())
    log("\n== G. MAE BY MONTH (mean_actual shows level changes) ==")
    log(mae_by(test["local_time"].dt.strftime("%Y-%m").to_numpy(), y, hp).round(3).to_string())

    log("\n== H. WEEKDAYS: MEAN ACTUAL AND PREDICTED BY HOUR OF DAY ==")
    wkd = (test["dow"] < 5).to_numpy()
    cols = {"actual": y[wkd], **{k: v[wkd] for k, v in hp.items()}}
    log(pd.DataFrame(cols).groupby(test["hour"].to_numpy()[wkd]).mean().round(2).to_string())
    log("\nshare of hours with absolute error above 2 people:")
    log(pd.Series({k: float((np.abs(y - v) > 2).mean()) for k, v in hp.items()}).round(3).to_string())

    log("\n== I. GP INTERVALS AND FIT TIME ==")
    rows = []
    for n, mu in mus.items():
        inside = np.abs(y - mu) <= 1.96 * sds[n]
        rows.append({"model": n, "coverage95": inside.mean(), "coverage95_day": inside[day].mean(),
                     "mean_sd": sds[n].mean(), "fit_s": secs[n]})
    log(pd.DataFrame(rows).set_index("model").round(3).to_string())
    log("\nfit seconds, other models: " + ", ".join(
        f"{k} {v:.2f}" for k, v in secs.items() if not k.startswith("GP")))

    out = pd.DataFrame({"space": test["space"].astype(str), "hour_utc": test["hour_utc"],
                        "local_time": test["local_time"], "actual": y,
                        **{k: v for k, v in preds.items()}, "linear": preds[primary]})
    out["GP seed 0 mu"], out["GP seed 0 sd"] = mus["GP seed 0"], sds["GP seed 0"]
    out.to_parquet(PROC / ("final_predictions_dry.parquet" if dry else "final_predictions.parquet"))

    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nsaved outputs/{out_path.name}")


if __name__ == "__main__":
    main()