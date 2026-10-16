"""Step 1: what do the occupancy events actually mean?

Before turning events into an hourly series we need to know whether silence between
events means 'nobody there' or 'sensor stopped reporting'. This script measures it."""
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "processed" / "occ.parquet"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

report = []   # every logged line is also saved to a text file at the end


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


# distinct values of the repeated text columns, so every batch shares one category list
spaces = pq.read_table(PATH, columns=["floorspaceid"]).column(0).unique().to_pylist()
statuses = pq.read_table(PATH, columns=["occupancystatus"]).column(0).unique().to_pylist()
short = [s[:8] for s in spaces]   # first 8 characters is enough to tell the 5 spaces apart

# stream the 9M rows in batches, shrinking types as we go so memory stays small
parts = []
for batch in pq.ParquetFile(PATH).iter_batches(
        batch_size=1_000_000,
        columns=["floorspaceid", "occupancystatus", "headcount", "collecteddate"]):
    df = batch.to_pandas()
    parts.append(pd.DataFrame({
        "space": pd.Categorical(df["floorspaceid"].str[:8], categories=short),
        "status": pd.Categorical(df["occupancystatus"], categories=statuses),
        "headcount": pd.to_numeric(df["headcount"]).astype("float32"),
        "t": pd.to_datetime(df["collecteddate"], format="ISO8601", utc=True),
    }))
    print(f"  loaded {sum(len(p) for p in parts):,} rows", flush=True)
occ = pd.concat(parts, ignore_index=True)
del parts

# sort by space then time, then look at each event's neighbour in the same space
occ = occ.sort_values(["space", "t"], kind="stable").reset_index(drop=True)


def by_space(col):
    """Group one column by space (fresh each call, so it sees newly added columns)."""
    return occ.groupby("space", observed=True)[col]


occ["gap_s"] = by_space("t").diff().dt.total_seconds()   # seconds since previous event
occ["prev_status"] = by_space("status").shift()          # status of previous event
occ["prev_hc"] = by_space("headcount").shift()           # headcount of previous event

# A. how far apart are events?
log("== A. SECONDS BETWEEN CONSECUTIVE EVENTS, per space ==")
q = by_space("gap_s").quantile([.5, .9, .99, .999]).unstack()
q.columns = ["median", "p90", "p99", "p99.9"]
q["max"] = by_space("gap_s").max()
q["pct_same_timestamp"] = occ["gap_s"].eq(0).groupby(occ["space"], observed=True).mean() * 100
log(q.round(1).to_string())

# B. does the headcount change at every event, or repeat?
has_prev = occ["prev_hc"].notna()
same_hc = (occ["headcount"] == occ["prev_hc"]) & has_prev
log("\n== B. REPEATED READINGS ==")
log(f"events whose headcount equals the previous event's: {same_hc.sum() / has_prev.sum():.1%}")

# C. do the status labels agree with the headcount?
log("\n== C. HEADCOUNT BY STATUS ==")
log(occ.groupby("status", observed=True)["headcount"].agg(
    events="size", mean="mean", pct_zero=lambda s: (s == 0).mean() * 100, max="max"
).round(2).to_string())

# D. the key test: what came before each long silence?
LONG = 600   # seconds. A gap over 10 minutes counts as 'long'
long_gap = occ["gap_s"] > LONG
log(f"\n== D. LONG GAPS (over {LONG // 60} min): {long_gap.sum():,} ({long_gap.mean():.2%} of events) ==")
log("status of the event BEFORE each long gap (%):")
log(occ.loc[long_gap, "prev_status"].value_counts(normalize=True).mul(100).round(1).to_string())
log("headcount of the event before each long gap:")
log(occ.loc[long_gap, "prev_hc"].describe(percentiles=[.5, .9, .99]).round(2).to_string())
log("gap length in hours by previous status (median, 90th percentile):")
g = occ.loc[long_gap].groupby("prev_status", observed=True)["gap_s"].quantile([.5, .9])
log((g.unstack() / 3600).round(2).to_string())

# E. when in the day do events happen (Melbourne local time)?
local_hour = occ["t"].dt.tz_convert("Australia/Melbourne").dt.hour
log("\n== E. SHARE OF EVENTS BY LOCAL HOUR OF DAY (%) ==")
log(local_hour.value_counts(normalize=True).sort_index().mul(100).round(1).to_string())

(OUTDIR / "event_semantics.txt").write_text("\n".join(report), encoding="utf-8")
print("\nsaved outputs/event_semantics.txt")