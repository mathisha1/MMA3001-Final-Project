from pathlib import Path   # handles file paths in a way that works on any computer
import pandas as pd       # reads Excel files into tables

# __file__ is this script, .parent is src/, .parent.parent is the repo root.
# This way the script finds data/raw no matter where you run it from.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# show more of each cell, and all columns, when printing
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# list every Excel file in the raw folder
files = sorted(RAW_DIR.glob("*.xlsx"))
print(f"Found {len(files)} Excel files in {RAW_DIR}\n")

for f in files:
    print("=" * 70)
    print("FILE:", f.name)

    # find out how many sheets the workbook has (some files hide extra sheets)
    xl = pd.ExcelFile(f, engine="openpyxl")
    print("sheets:", xl.sheet_names)

    for sheet in xl.sheet_names:
        # nrows=5 reads only the first 5 rows, so big files load quickly
        df = pd.read_excel(f, sheet_name=sheet, nrows=5, engine="openpyxl")
        print(f"\n--- sheet '{sheet}' ---")
        print("columns:", list(df.columns))   # the exact column names, spelling and all
        print(df)                              # the first 5 rows