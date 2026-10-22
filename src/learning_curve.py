"""Validation-only learning curve: does each model improve as it sees more training data?

Training windows start at the first training day. The test period is never touched. Early
windows have not seen a full year of seasons, so read trends, not exact values."""
from pathlib import Path
import pandas as pd
from src.evaluation import TARGET, coverage, metrics, weekday_day
from src.features import split_by_time
from src.modeling import load_prepared, model_zoo

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


def main():
    """Fit each model on growing training windows and score on validation."""
    pd.set_option("display.width", 200)
    OUTDIR.mkdir(exist_ok=True)
    h = load_prepared()
    parts = split_by_time(h)
    train, val = parts["train"], parts["val"]          # parts["test"] is never used
    start = train["local_time"].min()
    zoo = model_zoo(True)
    day = weekday_day(val)
    keys = ["space", "period", "is_weekend", "hour"]
    rows = []
    for months in (6, 9, 12, None):
        tr = train if months is None else train[train["local_time"] < start + pd.DateOffset(months=months)]
        label = "all" if months is None else f"{months} months"
        log(f"{label}: {len(tr):,} training hours, lookup cells seen in validation: "
            f"{coverage(tr, keys, val):.1%}")
        for name, fn in zoo.items():
            p = fn(tr, val)
            m, md = metrics(val[TARGET], p), metrics(val[TARGET][day], p[day])
            rows.append({"window": label, "model": name, "MAE": m["MAE"], "RMSE": m["RMSE"],
                         "MAE_day": md["MAE"], "RMSE_day": md["RMSE"]})
    res = pd.DataFrame(rows)
    order = ["6 months", "9 months", "12 months", "all"]
    for col in ["MAE", "RMSE", "MAE_day", "RMSE_day"]:
        log(f"\n{col} on validation:")
        log(res.pivot_table(index="model", columns="window", values=col)[order].round(3).to_string())
    (OUTDIR / "learning_curve.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nsaved outputs/learning_curve.txt")


if __name__ == "__main__":
    main()