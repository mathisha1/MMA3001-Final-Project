"""Step 5: baselines that know the university term (teaching, non_teaching, closure)."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation import TARGET, metrics, lookup_prediction, coverage, weekday_day
from src.features import add_calendar_features, split_by_time, load_term_calendar, add_term

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)

report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


h = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
h = h[~((h["n_events"] == 0) & (h["snapshot"] > 0))]       # drop the ambiguous held-value hours
h = add_term(add_calendar_features(h), load_term_calendar(ROOT / "data" / "term_calendar.csv"))
parts = split_by_time(h)
train, val = parts["train"], parts["val"]

log("== HOURS PER PERIOD LABEL ==")
log(pd.DataFrame({k: v["period"].value_counts() for k, v in parts.items()})
    .fillna(0).astype(int).to_string())

# sanity check of the calendar against the training data only
log("\n== CALENDAR CHECK: TRAIN, WEEKDAY 9AM-5PM ==")
d = train[weekday_day(train)]
log(d.groupby("period")[TARGET].agg(
    hours="size", mean="mean", pct_zero=lambda s: (s == 0).mean() * 100).round(2).to_string())
log("expect: teaching highest, closure near zero. If not, recheck the dates in term_calendar.csv")

g = train[TARGET].mean()                                   # fallback for groups unseen in training
specs = {
    "hour-of-week (all spaces)": ["hour_of_week"],
    "term + hour-of-week (all spaces)": ["period", "hour_of_week"],
    "term + hour-of-week, per space": ["space", "period", "hour_of_week"],
    "term + weekend + hour, per space": ["space", "period", "is_weekend", "hour"],
}
preds = {n: lookup_prediction(train, k, val, g) for n, k in specs.items()}


def table(mask):
    """Metrics for every baseline on the validation rows picked by mask."""
    rows = [{"baseline": n, **metrics(val[TARGET][mask], p[mask]),
             "cells_seen": coverage(train, specs[n], val[mask])} for n, p in preds.items()]
    return pd.DataFrame(rows).round(3).to_string(index=False)


day = weekday_day(val)
log(f"\n== VALIDATION, ALL HOURS (mean actual {val[TARGET].mean():.3f} people) ==")
log(table(np.ones(len(val), bool)))
log(f"\n== VALIDATION, WEEKDAY 9AM-5PM ({day.sum():,} hours) ==")
log(table(day))
log("cells_seen = share of validation hours whose group also exists in training")

best = min(preds, key=lambda n: metrics(val[TARGET], preds[n])["MAE"])
log(f"\n== LOWEST VALIDATION MAE: {best}. ERROR BY PERIOD ==")
rows = []
for per in sorted(val["period"].unique()):
    m = (val["period"] == per).to_numpy()
    rows.append({"period": per, "hours": int(m.sum()), "mean_actual": val[TARGET][m].mean(),
                 **metrics(val[TARGET][m], preds[best][m])})
log(pd.DataFrame(rows).round(3).to_string(index=False))

(OUTDIR / "baselines_term.txt").write_text("\n".join(report), encoding="utf-8")
print("\nsaved outputs/baselines_term.txt")