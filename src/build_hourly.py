"""Step 2: build hourly occupancy series three ways and measure how much they differ."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.hourly import load_events, hourly_series

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))

def main():
    '''Build hourly series three ways and compare them.'''
    occ = load_events(PROC / "occ.parquet")
    tables = []
    for space, g in occ.groupby("space", observed=True):
        t = g["t"].to_numpy(dtype="float64")
        v = g["headcount"].to_numpy(dtype="float64")     # float64 so long sums stay accurate
        tables.append(hourly_series(t, v, str(space)))
    hourly = pd.concat(tables, ignore_index=True)
    hourly.to_parquet(PROC / "hourly.parquet")
    hourly["no_events"] = hourly["n_events"] == 0

    log(f"hourly table: {len(hourly):,} rows")
    log("\n== A. HOURS PER SPACE, AND SHARE WITH NO EVENTS ==")
    log(hourly.groupby("space").agg(
        hours=("n_events", "size"),
        pct_no_events=("no_events", lambda s: s.mean() * 100)).round(1).to_string())

    held = hourly[hourly["no_events"]]
    amb = held[held["snapshot"] > 0]
    log(f"\nhours with no events: {len(held):,} ({len(held) / len(hourly):.1%})")
    log(f"  of those, held value is 0 (empty): {(held['snapshot'] == 0).mean():.1%}")
    log(f"  held value above 0 (ambiguous): {len(amb):,} hours")
    if len(amb):
        log(f"  how long that value had been held, in hours: median "
            f"{amb['stale_s'].median() / 3600:.2f}, 90th pct {amb['stale_s'].quantile(.9) / 3600:.2f}, "
            f"max {amb['stale_s'].max() / 3600:.1f}")

    log("\nlongest hold per space:")
    idx = hourly.groupby("space")["stale_s"].idxmax()
    top = hourly.loc[idx, ["space", "hour_utc", "snapshot"]].assign(
        stale_hours=(hourly.loc[idx, "stale_s"] / 3600).round(1))
    log(top.to_string(index=False))

    # compare the methods on hours that contain at least one event
    ev = hourly[~hourly["no_events"]].copy()
    log(f"\n== B. METHODS vs EXACT TIME-WEIGHTED MEAN (hours with events: {len(ev):,}) ==")
    log(f"mean time-weighted headcount: {ev['time_weighted'].mean():.3f} people")
    rows = []
    for name in ["event_mean", "snapshot", "peak"]:
        d = ev[name] - ev["time_weighted"]
        rows.append({"method": name, "bias": d.mean(), "MAE": d.abs().mean(),
                    "RMSE": np.sqrt((d ** 2).mean()), "p95_abs": d.abs().quantile(.95),
                    "max_abs": d.abs().max()})
    log(pd.DataFrame(rows).round(3).to_string(index=False))

    ev["d_event"] = ev["event_mean"] - ev["time_weighted"]
    log("\nevent_mean minus time_weighted, per space:")
    log(ev.groupby("space")["d_event"].agg(bias="mean", MAE=lambda s: s.abs().mean())
        .round(3).to_string())

    ev["local_hour"] = ev["hour_utc"].dt.tz_convert("Australia/Melbourne").dt.hour
    log("\nby local hour of day:")
    log(ev.groupby("local_hour").agg(
        mean_time_weighted=("time_weighted", "mean"),
        event_mean_bias=("d_event", "mean"),
        event_mean_MAE=("d_event", lambda s: s.abs().mean())).round(3).to_string())

    (OUTDIR / "hourly_comparison.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nsaved outputs/hourly_comparison.txt")

if __name__ == "__main__":
    main()