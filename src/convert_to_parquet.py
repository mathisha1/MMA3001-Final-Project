"""Converts the raw files (CSV or Excel) to parquet.
Big CSVs are streamed in chunks so they never have to fit in memory all at once."""
import time
from pathlib import Path
import pandas as pd
import pyarrow as pa            # parquet engine, lets us write a file chunk by chunk
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent   # repo root
RAW = ROOT / "data" / "raw"                     # where the raw files are
OUT = ROOT / "data" / "processed"               # where parquet copies go
OUT.mkdir(parents=True, exist_ok=True)

CHUNK_ROWS = 500_000   # rows read per chunk. Lower this if you run out of memory


def csv_to_parquet(csv_path, out_path):
    """Stream a big CSV into one parquet file. Returns the total number of rows."""
    writer, total = None, 0
    # dtype=str keeps IDs and timestamps exactly as written (no silent rounding).
    # utf-8-sig removes the invisible marker Excel sometimes puts at the file start.
    for chunk in pd.read_csv(csv_path, dtype=str, chunksize=CHUNK_ROWS,
                             encoding="utf-8-sig"):
        if writer is None:
            # fix every column as text so all chunks share one schema
            schema = pa.schema([(c, pa.string()) for c in chunk.columns])
            writer = pq.ParquetWriter(out_path, schema)
        writer.write_table(pa.Table.from_pandas(chunk, schema=schema,
                                                preserve_index=False))
        total += len(chunk)
        print(f"    {total:,} rows so far", flush=True)
    writer.close()
    return total


# (file, sheet position or None for CSV, output name)
JOBS = [
    ("Sensor ID and Locations.xlsx", 0, "env_loc"),
    ("Sensor ID and Locations.xlsx", 1, "occ_map"),
    ("5EnvSensor_MayToDec2024_180kRows.csv", None, "env"),
    ("5occupancySensor_MayToDec2024_9MRows.csv", None, "occ"),   # biggest, so last
]

for fname, sheet, name in JOBS:
    start = time.time()
    print(f"Reading {fname} ...", flush=True)
    if sheet is None:
        n = csv_to_parquet(RAW / fname, OUT / f"{name}.parquet")
        print(f"  saved {name}.parquet: {n:,} rows in {time.time() - start:.0f}s")
    else:
        names = pd.ExcelFile(RAW / fname, engine="openpyxl").sheet_names
        print(f"  sheet names: {names}")   # check these are the sheets you expect
        df = pd.read_excel(RAW / fname, sheet_name=sheet, dtype=str, engine="openpyxl")
        df.to_parquet(OUT / f"{name}.parquet")
        print(f"  saved {name}.parquet: {len(df):,} rows, columns {list(df.columns)}")