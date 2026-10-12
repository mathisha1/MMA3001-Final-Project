"""Do the two healthy env sensors track any occupancy space once the shared daily/weekly
rhythm is removed? Also: which spaces map to rooms, and how densely each sensor reports."""
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

HEALTHY = ["6012002000326", "6012002000869"]   # the two sensors with real timestamps
MIN_HOURS = 100                                 # ignore pairs with fewer shared hours
report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


# 1. hourly mean headcount per space, built batch by batch so memory stays small
parts = []
for batch in pq.ParquetFile(PROC / "occ.parquet").iter_batches(
        batch_size=1_000_000, columns=["floorspaceid", "headcount", "collecteddate"]):
    df = batch.to_pandas()
    t = pd.to_datetime(df["collecteddate"], format="ISO8601", utc=True)
    df["hour"] = t.dt.floor("h")                                   # start of the hour
    df["headcount"] = pd.to_numeric(df["headcount"])
    parts.append(df.groupby(["floorspaceid", "hour"])["headcount"].agg(["sum", "count"]))
    print(f"  processed {len(parts)} batch(es)", flush=True)
tot = pd.concat(parts).groupby(level=[0, 1]).sum()                 # combine batches
occ_h = (tot["sum"] / tot["count"]).unstack(0)                     # rows = hour, cols = space
log(f"hourly occupancy table: {occ_h.shape[0]:,} hours x {occ_h.shape[1]} spaces")
log("(crude: a plain mean of the events logged in each hour, not time-weighted)")

# 2. which occupancy spaces appear in the mapping sheet, and as which rooms
occ_map = pd.read_parquet(PROC / "occ_map.parquet")
sid = occ_map["SpaceId"].astype("string").str.strip().str.lower()
log("\n== SPACES THAT MATCH THE MAPPING SHEET ==")
log(occ_map[sid.isin(occ_h.columns)][["Building", "Floor/Room", "Area", "Note", "SpaceId"]].to_string())

# 3. how densely do the two healthy sensors report?
long = pd.read_parquet(PROC / "env_long.parquet")
co2 = (long[(long["variable"] == "Carbon dioxide") & long["sensorid"].isin(HEALTHY)]
       .drop_duplicates(["sensorid", "time"]).sort_values(["sensorid", "time"]))
for s in HEALTHY:
    d = co2[co2["sensorid"] == s]
    gap = d["time"].diff().dt.total_seconds() / 60
    log(f"\n== SENSOR {s}: {len(d):,} distinct readings ==")
    log(f"gap between readings (minutes): median {gap.median():.1f}, "
        f"90th pct {gap.quantile(.9):.1f}, 99th pct {gap.quantile(.99):.1f}")
    log("readings per month:")
    log(d.groupby(d["time"].dt.strftime("%Y-%m")).size().to_string())

# 4. correlate hourly CO2 with hourly headcount, raw and with the weekly rhythm removed
co2_h = (co2.set_index("time").groupby("sensorid")["value"].resample("1h").mean().unstack(0))
rows = []
for s in HEALTHY:
    for sp in occ_h.columns:
        both = pd.concat([co2_h[s], occ_h[sp]], axis=1, sort=True).dropna()
        both.columns = ["co2", "hc"]
        n = len(both)
        if n < MIN_HOURS:
            rows.append((s[-3:], sp[:8], n, float("nan"), float("nan")))
            continue
        local = both.index.tz_convert("Australia/Melbourne")        # local hour-of-week
        how = (local.dayofweek * 24 + local.hour).to_numpy()
        resid = both - both.groupby(how).transform("mean")          # departure from the norm
        rows.append((s[-3:], sp[:8], n, both["co2"].corr(both["hc"]),
                     resid["co2"].corr(resid["hc"])))
res = pd.DataFrame(rows, columns=["sensor", "space", "shared_hours", "r_raw", "r_after_rhythm_removed"])
log("\n== CO2 vs HEADCOUNT, per sensor and space ==")
log(res.round(2).to_string(index=False))
log("\nA real pairing should keep a clearly higher r_after_rhythm_removed than the other pairs.")
log("If all pairs fall to about 0, the sensors cannot be tied to any space with this data.")

(OUTDIR / "diagnose_link.txt").write_text("\n".join(report), encoding="utf-8")