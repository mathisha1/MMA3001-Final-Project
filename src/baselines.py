"""Step 4: baseline forecasts evaluated on the validation period only."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import add_calendar_features, split_by_time
from src.evaluation import metrics, lookup_prediction

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
TARGET = "time_weighted"      # exact mean headcount in the hour (people)

report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


def main ():
    '''Score the calendar baselines on validation.'''
    h = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    h = h[~((h["n_events"] == 0) & (h["snapshot"] > 0))]     # drop 138 ambiguous held-value hours
    h = add_calendar_features(h)
    parts = split_by_time(h)
    train, val = parts["train"], parts["val"]

    log("== SPLIT (local time) ==")
    for name, d in parts.items():
        log(f"{name}: {len(d):,} hours, {d['local_time'].min().date()} to {d['local_time'].max().date()}, "
            f"spaces: {sorted(d['space'].unique())}")
    log("test set is not used in this step.")

    g = train[TARGET].mean()
    preds = {
        "overall mean": np.full(len(val), g),
        "space mean": lookup_prediction(train, ["space"], val, g),
        "hour-of-week mean (all spaces)": lookup_prediction(train, ["hour_of_week"], val, g),
        "hour-of-week mean per space": lookup_prediction(train, ["space", "hour_of_week"], val, g),
    }

    log(f"\n== BASELINES ON VALIDATION (mean target in val: {val[TARGET].mean():.3f} people) ==")
    rows = [{"baseline": k, **metrics(val[TARGET], p)} for k, p in preds.items()]
    log(pd.DataFrame(rows).round(3).to_string(index=False))

    # the hours that matter for ventilation: weekdays 9am to 5pm
    day = ((val["dow"] < 5) & (val["hour"] >= 9) & (val["hour"] < 17)).to_numpy()
    log(f"\n== SAME, WEEKDAY 9AM-5PM ONLY ({day.sum():,} hours) ==")
    rows = [{"baseline": k, **metrics(val[TARGET][day], p[day])} for k, p in preds.items()]
    log(pd.DataFrame(rows).round(3).to_string(index=False))

    best = preds["hour-of-week mean per space"]
    log("\n== BEST BASELINE, PER SPACE ==")
    rows = [{"space": s, "hours": int(m.sum()), **metrics(val[TARGET][m], best[m])}
            for s in sorted(val["space"].unique()) for m in [(val["space"] == s).to_numpy()]]
    log(pd.DataFrame(rows).round(3).to_string(index=False))

    (OUTDIR / "baselines.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nsaved outputs/baselines.txt")

if __name__ == "__main__":
    main()