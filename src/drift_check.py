"""Step 4b: has occupancy level or pattern shifted between train, validation and test?"""
from pathlib import Path
import pandas as pd
from src.features import add_calendar_features, split_by_time

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg)
    report.append(str(msg))

def main():
    '''Check the level shift between periods.'''
    h = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    h = h[~((h["n_events"] == 0) & (h["snapshot"] > 0))]      # drop the ambiguous held-value hours
    h = add_calendar_features(h)
    parts = split_by_time(h)
    h = pd.concat([parts["train"], parts["val"]])   # keep the test period out of exploration
    h["ym"] = h["local_time"].dt.strftime("%Y-%m")             # calendar month label
    day = (h["dow"] < 5) & (h["hour"] >= 9) & (h["hour"] < 17)  # weekday 9am to 5pm
    d = h[day]

    log("== MEAN HEADCOUNT, WEEKDAY 9AM-5PM, PER MONTH AND SPACE ==")
    log(d.pivot_table(index="ym", columns="space", values="time_weighted", aggfunc="mean").round(2).to_string())

    log("\n== SHARE OF THOSE HOURS WITH ZERO PEOPLE (%) ==")
    log(d.pivot_table(index="ym", columns="space", values="time_weighted",
                    aggfunc=lambda s: (s == 0).mean() * 100).round(0).to_string())

    log("\n== BY PERIOD ==")
    rows = []
    for name, part in split_by_time(h).items():
        pd_day = part[(part["dow"] < 5) & (part["hour"] >= 9) & (part["hour"] < 17)]
        rows.append({"period": name, "mean_all_hours": part["time_weighted"].mean(),
                    "mean_weekday_day": pd_day["time_weighted"].mean(),
                    "pct_zero_weekday_day": (pd_day["time_weighted"] == 0).mean() * 100})
    log(pd.DataFrame(rows).round(3).to_string(index=False))

    (OUTDIR / "drift_check.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nsaved outputs/drift_check.txt")

if __name__ == "__main__":
    main()