"""Prints the key numbers that decide whether the data can support each project option.
Old values (from the first extract) are in the labels so you can compare by eye."""
from pathlib import Path
import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
occ = pd.read_parquet(PROC / "occ.parquet")
env_long = pd.read_parquet(PROC / "env_long.parquet")   # written by audit_data.py
env_raw = pd.read_parquet(PROC / "env.parquet")

occ["t"] = pd.to_datetime(occ["collecteddate"], format="ISO8601", utc=True, errors="coerce")

print("== OCCUPANCY ==")
print(f"rows: {len(occ):,}   (old: 1,048,575, which was Excel's cap)")
print("still truncated?", len(occ) >= 1_048_000)
print(f"time range: {occ['t'].min()} to {occ['t'].max()}   (old: 2023-11-23 to 2024-05-06)")
print(f"unique floorspaceid: {occ['floorspaceid'].nunique()}   (old: 5)")
print(f"unique deviceid: {occ['deviceid'].nunique()}   (old: 1)")

print("\n== ENV ==")
print(f"raw rows: {len(env_raw):,}   (old: 179,447)")
print(f"unique sensors: {env_raw['sensorid'].nunique()}   (old: 5)")

# a healthy sensor has about as many unique timestamps as readings
co2 = env_long[env_long["variable"] == "Carbon dioxide"]
stats = co2.groupby("sensorid").agg(readings=("time", "size"), unique_times=("time", "nunique"))
stats["pct_unique"] = (stats["unique_times"] / stats["readings"] * 100).round(1)
print("\nCO2 readings vs unique timestamps per sensor")
print("(old: only sensor ...869 was healthy, the other four were stuck near 0%)")
print(stats.to_string())
