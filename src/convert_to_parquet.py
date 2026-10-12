"""One-off conversion of the raw Excel files to fast parquet files."""
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # repo root
RAW = ROOT / "data" / "raw"                     # where the Excel files are
OUT = ROOT / "data" / "processed"               # where parquet copies go
OUT.mkdir(parents=True, exist_ok=True)

# (excel file, sheet position, output name). Sheet 0 = first sheet, 1 = second.
JOBS = [
    ("5EnvSensor_MayToDec2024_180kRows.xlsx", 0, "env"),
    ("Sensor ID and Locations.xlsx", 0, "env_loc"),
    ("Sensor ID and Locations.xlsx", 1, "occ_map"),
    ("5occupancySensor_MayToDec2024_9MRows.xlsx", 0, "occ"),  # biggest, so last
]

for fname, sheet, name in JOBS:
    start = time.time()
    print(f"Reading {fname} (sheet {sheet}) ...", flush=True)
    # dtype=str keeps long IDs and room labels like '1.40' exactly as written
    df = pd.read_excel(RAW / fname, sheet_name=sheet, dtype=str, engine="openpyxl")
    df.to_parquet(OUT / f"{name}.parquet")      # fast binary copy
    print(f"  saved {name}.parquet: {len(df):,} rows, {df.shape[1]} cols, "
          f"{time.time() - start:.0f}s")