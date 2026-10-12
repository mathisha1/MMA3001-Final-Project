"""Diagnostic: do env sensors and occupancy overlap in time, which env clock is right,
and does any env sensor track any occupancy space?"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

report = []   # everything logged is also saved to a text file at the end


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg)
    report.append(str(msg))


# ------------------------------------------------------------------ load
env_long = pd.read_parquet(PROC / "env_long.parquet")   # one row per reading
occ = pd.read_parquet(PROC / "occ.parquet")
env_loc = pd.read_parquet(PROC / "env_loc.parquet")

# 'created' = when the reading reached the database, 'time' = timestamp inside the JSON
env_long["created"] = pd.to_datetime(env_long["created"], format="ISO8601",
                                     utc=True, errors="coerce")
occ["t"] = pd.to_datetime(occ["collecteddate"], format="ISO8601", utc=True, errors="coerce")
occ["headcount"] = pd.to_numeric(occ["headcount"], errors="coerce")

occ_start, occ_end = occ["t"].min(), occ["t"].max()
log(f"OCCUPANCY WINDOW (UTC): {occ_start} to {occ_end}")

# CO2 only: every row has it, so it represents when each sensor was reporting
co2 = env_long[env_long["variable"] == "Carbon dioxide"].copy()
co2["lag_days"] = (co2["created"] - co2["time"]).dt.total_seconds() / 86400

# label each sensor with its room from the location sheet, if it has one
loc_label = (env_loc["Building"].astype(str) + " " + env_loc["Floor/Room"].astype(str)
             + " " + env_loc["Area"].fillna("").astype(str))
loc_lookup = dict(zip(env_loc["ID"].astype(str).str.strip(), loc_label.str.strip()))

# ------------------------------------------------- 1. the 5 env sensors
log("\n== 1. THE ENV SENSORS ==")
g = co2.groupby("sensorid")
summary = g.agg(
    readings=("value", "size"),
    created_first=("created", "min"), created_last=("created", "max"),
    time_first=("time", "min"), time_last=("time", "max"),
    lag_median_days=("lag_days", "median"),
    lag_min_days=("lag_days", "min"), lag_max_days=("lag_days", "max"),
)
summary["room"] = [loc_lookup.get(s, "NO LOCATION") for s in summary.index]
summary["pct_future_stamped"] = (co2["lag_days"] < 0).groupby(co2["sensorid"]).mean() * 100
summary["pct_time_in_occ_window"] = (
    co2["time"].between(occ_start, occ_end).groupby(co2["sensorid"]).mean() * 100)
summary["pct_created_in_occ_window"] = (
    co2["created"].between(occ_start, occ_end).groupby(co2["sensorid"]).mean() * 100)
log(summary.T.to_string())   # transposed so 5 sensors fit side by side

# ------------------------------------------------- 2. which clock to trust
log("\n== 2. LAG BETWEEN JSON TIME AND DATABASE TIME (share of readings, %) ==")
bins = [-float("inf"), 0, 1, 30, 90, 180, float("inf")]
labels = ["future", "0-1 d", "1-30 d", "30-90 d", "90-180 d", ">180 d"]
co2["lag_bucket"] = pd.cut(co2["lag_days"], bins=bins, labels=labels)
log((pd.crosstab(co2["sensorid"], co2["lag_bucket"], normalize="index") * 100)
    .round(1).to_string())

# sampling cadence and duplicates, using the JSON clock
co2 = co2.sort_values(["sensorid", "time"])
gap_min = co2.groupby("sensorid")["time"].diff().dt.total_seconds() / 60
log("\nmedian gap between consecutive readings (minutes), JSON clock:")
log(gap_min.groupby(co2["sensorid"]).median().round(1).to_string())
dup = co2.groupby("sensorid").apply(lambda d: len(d) - d["time"].nunique(),
                                    include_groups=False)
log("duplicate timestamps per sensor:")
log(dup.to_string())

# ------------------------------------------------- 3. occupancy spaces
log("\n== 3. OCCUPANCY SPACES ==")
log(occ.groupby("floorspaceid").agg(
    events=("id", "size"), first=("t", "min"), last=("t", "max"),
    headcount_median=("headcount", "median"), headcount_max=("headcount", "max"),
).to_string())


# ------------------------------------------------- 4. does any sensor track any space?
def hourly_corr(time_col):
    """Hourly mean CO2 vs hourly mean headcount for every sensor/space pair,
    using the chosen env clock ('time' = JSON, 'created' = database)."""
    inside = co2[co2[time_col].between(occ_start, occ_end)]
    if inside.empty:
        return None, None
    co2_h = (inside.set_index(time_col).groupby("sensorid")["value"]
             .resample("1h").mean().unstack(0))
    occ_h = (occ.set_index("t").groupby("floorspaceid")["headcount"]
             .resample("1h").mean().unstack(0))   # crude: mean of the events in each hour
    r = pd.DataFrame(index=co2_h.columns, columns=occ_h.columns, dtype=float)
    n = r.copy()
    for s in co2_h.columns:
        for sp in occ_h.columns:
            both = pd.concat([co2_h[s], occ_h[sp]], axis=1).dropna()
            n.loc[s, sp] = len(both)
            if len(both) >= 24:                    # need at least a day of shared hours
                r.loc[s, sp] = both.iloc[:, 0].corr(both.iloc[:, 1])
    return r, n


for clock, name in [("time", "JSON clock"), ("created", "database clock")]:
    log(f"\n== 4. CORRELATION CO2 vs HEADCOUNT, {name} ==")
    r, n = hourly_corr(clock)
    if r is None:
        log("no CO2 readings fall inside the occupancy window on this clock")
        continue
    log("shared hours (rows = env sensor, columns = occupancy space):")
    log(n.fillna(0).astype(int).to_string())
    log("Pearson r (blank = under 24 shared hours):")
    log(r.round(2).to_string())

log("\nNote: 5 sensors x 5 spaces = 25 comparisons, so one moderate r could be chance.")
log("A real match should show r clearly above the rest, with hundreds of shared hours.")

(OUTDIR / "diagnose_overlap.txt").write_text("\n".join(report), encoding="utf-8")