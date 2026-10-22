"""Step 11: descriptive residual analysis of the one-shot test predictions.

Reads data/processed/final_predictions.parquet. It refits nothing and changes no setting;
it only describes where the errors are."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import add_calendar_features, add_term, load_term_calendar

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
MODELS = ["training mean", "lookup", "linear", "tree", "GP seed 0", "GP seed 1", "GP seed 2"]
report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


def main():
    """Run the descriptive checks and save outputs/residual_analysis.txt."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    OUTDIR.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "processed" / "final_predictions.parquet")
    d = add_term(add_calendar_features(df),
                 load_term_calendar(ROOT / "data" / "term_calendar.csv"))
    y = d["actual"].to_numpy(float)
    preds = {m: d[m].to_numpy(float) for m in MODELS}
    log(f"{len(d):,} test hours, mean actual {y.mean():.3f}")

    log("\n== A. MAE BY TERM, AND THE MAE OF ALWAYS PREDICTING 0 ==")
    rows = []
    for per in ["closure", "non_teaching", "teaching"]:
        m = (d["period"] == per).to_numpy()
        row = {"period": per, "hours": int(m.sum()), "mean_actual": y[m].mean(),
               "predict 0": np.abs(y[m]).mean()}
        for name, p in preds.items():
            row[name] = np.abs(y[m] - p[m]).mean()
        rows.append(row)
    log(pd.DataFrame(rows).set_index("period").round(3).to_string())

    log("\n== B. HOW CONCENTRATED IS THE ERROR? (share of total absolute error) ==")
    for name in ["lookup", "tree", "GP seed 0"]:
        err = np.sort(np.abs(y - preds[name]))[::-1]
        n1, n5 = int(len(err) * 0.01), int(len(err) * 0.05)
        log(f"{name}: worst 1% of hours = {err[:n1].sum() / err.sum():.1%}, "
            f"worst 5% = {err[:n5].sum() / err.sum():.1%}")

    log("\n== C. BUSY HOURS (actual >= 3 people) ==")
    busy = y >= 3
    log(f"{busy.sum():,} hours ({busy.mean():.1%} of all), mean actual {y[busy].mean():.2f}")
    rows = []
    for name, p in preds.items():
        err = np.abs(y - p)
        rows.append({"model": name, "mean_predicted": p[busy].mean(),
                     "MAE_busy": err[busy].mean(),
                     "share_of_all_error": err[busy].sum() / err.sum()})
    log(pd.DataFrame(rows).set_index("model").round(3).to_string())

    log("\n== D. OVER-PREDICTIONS BY MORE THAN 2 PEOPLE ==")
    tree = preds["tree"]
    lv = (pd.DataFrame({"tree_prediction": tree.round(2), "actual": y})
          .groupby("tree_prediction").agg(hours=("actual", "size"), mean_actual=("actual", "mean")))
    log("most-used tree prediction levels:")
    log(lv.sort_values("hours", ascending=False).head(12).round(3).to_string())
    for name in ["tree", "lookup"]:
        over = (preds[name] - y) > 2
        log(f"\n{name}: {over.sum()} hours ({over.mean():.2%}) over-predicted by more than 2 people. "
            "Most common (space, term, is_weekend):")
        log(d[over].groupby(["space", "period", "is_weekend"]).size()
            .sort_values(ascending=False).head(8).to_string())

    log("\n== E. BIAS BY SPACE (mean predicted minus mean actual, people) ==")
    bias = pd.DataFrame({m: pd.Series(p - y).groupby(d["space"].to_numpy()).mean()
                         for m, p in preds.items()})
    log(bias.round(3).to_string())

    (OUTDIR / "residual_analysis.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nsaved outputs/residual_analysis.txt")


if __name__ == "__main__":
    main()