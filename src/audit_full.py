"""Memory-safe audit of the full extract: occupancy coverage, ID linkage, env sensor health."""
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

report = []   # every logged line is also saved to a text file at the end


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


def clean(s):
    """Lowercase and strip IDs so different tables can be compared."""
    return s.astype("string").str.strip().str.lower()


# ------------------------------------------------------------ load occupancy
OCC_PATH = PROC / "occ.parquet"
CAT_COLS = ["deviceid", "floorspaceid", "occupancystatus"]   # repeated text, so store as category
NEEDED = CAT_COLS + ["headcount", "collecteddate"]

# first pass: the list of distinct values per category column (cheap, one column at a time)
cats = {c: pq.read_table(OCC_PATH, columns=[c]).column(c).unique().to_pylist()
        for c in CAT_COLS}

# second pass: stream the file in 1M-row batches, parse times, shrink types as we go
parts = []
for batch in pq.ParquetFile(OCC_PATH).iter_batches(batch_size=1_000_000, columns=NEEDED):
    df = batch.to_pandas()
    df["t"] = pd.to_datetime(df["collecteddate"], format="ISO8601", utc=True, errors="coerce")
    df["headcount"] = pd.to_numeric(df["headcount"], errors="coerce").astype("float32")
    for c in CAT_COLS:   # same category list in every batch so they concat cleanly
        df[c] = pd.Categorical(df[c], categories=cats[c])
    parts.append(df.drop(columns="collecteddate"))
    print(f"  loaded {sum(len(p) for p in parts):,} rows", flush=True)
occ = pd.concat(parts, ignore_index=True)
del parts

# ------------------------------------------------------------ occupancy audit
log("== OCCUPANCY (full extract) ==")
log(f"rows: {len(occ):,}   unparseable timestamps: {occ['t'].isna().sum():,}")
occ_start, occ_end = occ["t"].min(), occ["t"].max()
log(f"time range (UTC): {occ_start} to {occ_end}")
log(f"unique deviceid: {occ['deviceid'].nunique()}, unique floorspaceid: {occ['floorspaceid'].nunique()}")

log("\nevents per month:")
month = occ["t"].dt.tz_localize(None).dt.to_period("M")   # drop tz only to label months
log(month.value_counts().sort_index().to_string())

log("\nstatus values:")
log(occ["occupancystatus"].value_counts(dropna=False).to_string())

log("\nheadcount summary (percentiles):")
log(occ["headcount"].describe(percentiles=[.5, .9, .99, .999]).to_string())
log(f"headcount NaN: {occ['headcount'].isna().sum():,}, negative: {(occ['headcount'] < 0).sum():,}")

log("\nper occupancy space:")
spaces = occ.groupby("floorspaceid", observed=True).agg(
    events=("headcount", "size"), first=("t", "min"), last=("t", "max"),
    hc_median=("headcount", "median"), hc_p99=("headcount", lambda s: s.quantile(.99)),
    hc_max=("headcount", "max"))
spaces["days_active"] = occ.groupby("floorspaceid", observed=True)["t"].agg(
    lambda s: s.dt.normalize().nunique())
log(spaces.sort_values("events", ascending=False).to_string())

# ------------------------------------------------------------ ID linkage
log("\n== ID LINKAGE (occupancy vs mapping sheet) ==")
occ_map = pd.read_parquet(PROC / "occ_map.parquet")
occ_spaces = {str(x).strip().lower() for x in cats["floorspaceid"]}
for col in ["SpaceId", "FloorId", "ID"]:
    shared = occ_spaces & set(clean(occ_map[col]).dropna())
    log(f"occ floorspaceid vs mapping {col}: {len(shared)} of {len(occ_spaces)} shared")
occ_devs = {str(x).strip().lower() for x in cats["deviceid"]}
log(f"occ deviceid vs mapping ID/SpaceId/FloorId: "
    f"{[len(occ_devs & set(clean(occ_map[c]).dropna())) for c in ['ID', 'SpaceId', 'FloorId']]}"
    f" shared of {len(occ_devs)}")
mapped = [s for s in cats["floorspaceid"] if str(s).strip().lower() in set(clean(occ_map['SpaceId']).dropna())]
log(f"share of events in a space that appears in the mapping: "
    f"{occ['floorspaceid'].isin(mapped).mean():.1%}")

# ------------------------------------------------------------ env readings
log("\n== ENV SENSORS ==")
env = pd.read_parquet(PROC / "env.parquet")
log(f"raw rows: {len(env):,}, unique sensors: {env['sensorid'].nunique()}")
rows, bad = [], 0
for sid, js, created in zip(env["sensorid"], env["jsondata"], env["createdate"]):
    try:
        blocks = json.loads(js)           # fails on empty or cut-off JSON
    except (TypeError, ValueError):
        bad += 1
        continue
    for b in blocks:
        var = b.get("variable", {})
        for v in b.get("values", []):
            if v.get("time") is not None and v.get("value") is not None:
                rows.append((sid, created, var.get("name"), var.get("unit"), v["time"], v["value"]))
long = pd.DataFrame(rows, columns=["sensorid", "created", "variable", "unit", "unix", "value"])
long["time"] = pd.to_datetime(pd.to_numeric(long["unix"], errors="coerce"), unit="s", utc=True)
long["value"] = pd.to_numeric(long["value"], errors="coerce")
long.drop(columns="unix").to_parquet(PROC / "env_long.parquet")
log(f"unreadable JSON rows: {bad}, total readings: {len(long):,}")

# healthy sensor = many distinct timestamps; frozen clock = one timestamp repeated
co2 = long[long["variable"] == "Carbon dioxide"]
stats = co2.groupby("sensorid").agg(readings=("time", "size"), unique_times=("time", "nunique"),
                                    first=("time", "min"), last=("time", "max"))
stats["pct_unique"] = (stats["unique_times"] / stats["readings"] * 100).round(1)
inside = co2[co2["time"].between(occ_start, occ_end)].groupby("sensorid")["time"].nunique()
stats["unique_times_inside_occ_window"] = inside.reindex(stats.index).fillna(0).astype(int)
env_loc = pd.read_parquet(PROC / "env_loc.parquet")
lookup = dict(zip(clean(env_loc["ID"]), env_loc["Building"].astype(str) + " " + env_loc["Floor/Room"].astype(str)))
stats["room"] = [lookup.get(str(s).strip().lower(), "NO LOCATION") for s in stats.index]
log(stats.to_string())

log("\nvariables available (readings per variable):")
log(long.groupby(["variable", "unit"]).size().to_string())

(OUTDIR / "audit_full.txt").write_text("\n".join(report), encoding="utf-8")
log("\nsaved outputs/audit_full.txt")