"""Audit of the raw tables: sizes, date ranges, data quality and how the IDs link."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

report = []   # every logged line is also saved to a text file at the end


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg)
    report.append(str(msg))


def clean(s):
    """Turn IDs into lowercase stripped strings so different tables can be compared."""
    return s.astype("string").str.strip().str.lower()


def to_time(s):
    """Parse ISO-style timestamps (with +11 style offsets) into UTC datetimes."""
    return pd.to_datetime(s, format="ISO8601", utc=True, errors="coerce")


env = pd.read_parquet(PROC / "env.parquet")
occ = pd.read_parquet(PROC / "occ.parquet")
env_loc = pd.read_parquet(PROC / "env_loc.parquet")
occ_map = pd.read_parquet(PROC / "occ_map.parquet")

# ---------------------------------------------------------------- occupancy
log("== OCCUPANCY ==")
log(f"rows: {len(occ):,}")
if len(occ) >= 1_048_000:
    log("WARNING: row count is at Excel's 1,048,576 limit, so the file is truncated")
occ["t"] = to_time(occ["collecteddate"])
log(f"time range (UTC): {occ['t'].min()} to {occ['t'].max()}  "
    f"(unparseable: {occ['t'].isna().sum()})")
log("events per month:")
log(occ.groupby(occ["t"].dt.strftime("%Y-%m")).size().to_string())
occ["headcount"] = pd.to_numeric(occ["headcount"], errors="coerce")
log("headcount values:")
log(occ["headcount"].value_counts(dropna=False).sort_index().to_string())
log("status values:")
log(occ["occupancystatus"].value_counts(dropna=False).to_string())
log(f"unique deviceid: {occ['deviceid'].nunique()}, "
    f"unique floorspaceid: {occ['floorspaceid'].nunique()}")
log("events per floorspaceid (summary):")
log(occ.groupby("floorspaceid").size().describe().to_string())

# ------------------------------------------------------------ environmental
log("\n== ENVIRONMENTAL ==")
log(f"raw rows: {len(env):,}, unique sensors: {env['sensorid'].nunique()}")
log("status values:")
log(env["status"].value_counts(dropna=False).to_string())
log(f"rows with a processing error: {env['processing_errors'].notna().sum()}")

records, bad_json, no_reading = [], 0, 0
for sid, js, created in zip(env["sensorid"], env["jsondata"], env["createdate"]):
    try:
        blocks = json.loads(js)          # fails if empty or cut off by Excel's cell limit
    except (TypeError, ValueError):
        bad_json += 1
        continue
    got = False
    for b in blocks:
        var = b.get("variable", {})
        for v in b.get("values", []):
            if v.get("time") is None or v.get("value") is None:
                continue
            records.append((sid, created, var.get("name"), var.get("unit"),
                            v["time"], v["value"]))
            got = True
    if not got:
        no_reading += 1

log(f"rows with unreadable JSON: {bad_json}, rows with no readings: {no_reading}")
long = pd.DataFrame(records, columns=["sensorid", "created", "variable", "unit",
                                      "unix", "value"])
long["time"] = pd.to_datetime(pd.to_numeric(long["unix"], errors="coerce"),
                              unit="s", utc=True)
long["value"] = pd.to_numeric(long["value"], errors="coerce")
log(f"total readings: {len(long):,}")
log("readings per variable:")
log(long.groupby(["variable", "unit"]).size().to_string())
log(f"reading time range (from JSON): {long['time'].min()} to {long['time'].max()}")

# gap between when a reading was stamped and when it reached the database
lag_days = (to_time(long["created"]) - long["time"]).dt.total_seconds() / 86400
log("days between reading time and database time (summary):")
log(lag_days.describe().to_string())
long.drop(columns=["unix"]).to_parquet(PROC / "env_long.parquet")

# ------------------------------------------------------------------ linkage
log("\n== ID LINKAGE ==")
env_ids = set(clean(env["sensorid"]).dropna())
loc_ids = set(clean(env_loc["ID"]).dropna())
log(f"env sensors in data: {len(env_ids)}, in location sheet: {len(loc_ids)}, "
    f"shared: {len(env_ids & loc_ids)}")
log(f"env sensors with NO location: {sorted(env_ids - loc_ids)}")

# does any occupancy ID column match any column of the mapping sheet?
for occ_col in ["deviceid", "floorspaceid"]:
    occ_vals = set(clean(occ[occ_col]).dropna())
    for map_col in ["ID", "SpaceId", "FloorId"]:
        map_vals = set(clean(occ_map[map_col]).dropna())
        log(f"occ {occ_col} vs mapping {map_col}: "
            f"{len(occ_vals & map_vals)} of {len(occ_vals)} shared")

# fallback: link through room labels (building + room number)
env_loc["room"] = clean(env_loc["Building"]) + "|" + clean(env_loc["Floor/Room"])
occ_map["room"] = clean(occ_map["Building"]) + "|" + clean(occ_map["Floor/Room"])
log(f"rooms in env sheet: {env_loc['room'].nunique()}, "
    f"in occupancy mapping sheet: {occ_map['room'].nunique()}, "
    f"in both: {len(set(env_loc['room']) & set(occ_map['room']))}")

lookup = (occ_map.assign(sid=clean(occ_map["SpaceId"]))[["sid", "room"]]
          .dropna().drop_duplicates("sid"))
occ["sid"] = clean(occ["floorspaceid"])
occ = occ.merge(lookup, on="sid", how="left")
log(f"occupancy rows whose floorspaceid maps to a room: {occ['room'].notna().mean():.1%}")

env_rooms = set(env_loc.loc[clean(env_loc["ID"]).isin(env_ids), "room"])
occ_rooms = set(occ["room"].dropna())
usable = sorted(env_rooms & occ_rooms)
log(f"rooms with BOTH env readings and mapped occupancy: {len(usable)}")
log(usable)

(OUTDIR / "data_audit.txt").write_text("\n".join(report), encoding="utf-8")