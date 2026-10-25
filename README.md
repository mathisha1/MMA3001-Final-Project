# MMA3001 project: forecasting hourly occupancy of campus workshop spaces

Monash Smart Infrastructure data (Dataset 2). Occupancy event logs are converted to hourly
series with a quantified numerical error, then forecast from calendar features with linear
regression, a decision tree and Gaussian process regression. Robustness is tested over time
and across spaces. The environmental sensor data was audited and could not be linked to the
occupancy spaces; that audit is part of the report.

## Data (not included in the repository)

Put these files in `data/raw/`:

- `5EnvSensor_MayToDec2024_180kRows.csv`
- `5occupancySensor_MayToDec2024_9MRows.csv`
- `Sensor ID and Locations.xlsx`

`data/term_calendar.csv` (university teaching, non-teaching and closure dates, copied from the
Monash key dates pages) is tracked.

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run order

Run from the repository root with the virtual environment active. Each script prints its
results and saves a copy in `outputs/`. Order matters: later scripts read files written by
earlier ones.

```
# 1. Data preparation and audit
python -m src.convert_to_parquet   # raw CSV/xlsx -> data/processed/*.parquet (run alone, about 1 min)
python -m src.audit_full           # needs step 1. Memory-safe audit of the full extract -> outputs/audit_full.txt, env_long.parquet
python -m src.event_checks         # needs step 1. What silence between occupancy events means -> outputs/event_semantics.txt

# 2. Hourly series and the numerical error of building it
python -m src.build_hourly         # needs step 1. Writes data/processed/hourly.parquet (used by every script below)
python -m src.convergence          # needs steps 1-2. Trapezoid/Simpson/Boole/Romberg error vs step size -> convergence.txt, convergence_table.csv

# 3. Baselines and models, validation period only
python -m src.baselines            # needs hourly.parquet. Mean and hour-of-week baselines
python -m src.baselines_term       # needs data/term_calendar.csv. Term-aware baselines
python -m src.drift_check          # level shift between years (train and validation only)
python -m src.models_basic         # linear regression and decision tree, tree depth chosen on validation
python -m src.linear_variants      # additive vs term-by-hour interaction linear model; sets LINEAR_PRIMARY
python -m src.robustness           # horizon, leave-one-space-out, drop the 2024-only space
python -m src.learning_curve       # does each model improve with more training data (validation only)
python -m src.gp_study             # Gaussian process accuracy, interval coverage and cost (slow, run alone)

# 4. Performance (run each alone, other jobs distort the timings)
python -m src.performance          # memory, profiling, per-hour scan vs cumulative integral, Cholesky scaling
python -m src.hardware_check       # matrix-multiply and copy speed against thread count

# 5. Final evaluation on the held-out test period
python -m src.final_test --dry-run # train -> validation; must reproduce the earlier numbers before the real run
python -m src.final_test           # train+validation -> test. Scored ONCE: refuses to run if outputs/final_test.txt exists
python -m src.figures              # needs final_test. Weekday profile, predicted vs actual, residuals, GP band
python -m src.residual_analysis    # needs final_test. Describes where the test errors are; refits nothing
python -m src.report_figures       # needs convergence_table.csv. Convergence, tree depth and cost figures
```

Slow steps: `gp_study` (about 10 to 40 minutes) and `final_test` (about 15 to 20 minutes,
mostly the Gaussian process fits).

## Early audit scripts (`legacy/`)

The first data extract had occupancy as an Excel file, which is cut off at Excel's limit of
1,048,575 rows, and environmental data that had to be unpacked from nested JSON. The scripts
in `legacy/` were written against that extract to decide whether the environmental sensors
could be linked to the occupancy spaces. They found they could not, and that finding is why
the project uses occupancy data only.

They are kept as the audit trail and are **not part of the final pipeline**. They were not
rerun on the full extract, and some assume the old files, so treat the saved outputs as the
evidence.

| Script | What it checked | Saved output |
|---|---|---|
| `peek_data.py` | Sheets, columns and first rows of each raw file | terminal only |
| `audit_data.py` | Row counts, JSON parsing, ID matching between the four tables | `outputs/data_audit.txt` |
| `diagnose_overlap.py` | Time ranges, repeated timestamps per env sensor, CO2 vs headcount correlation | `outputs/diagnose_overlap.txt` |
| `diagnose_link.py` | The same correlation after removing the weekly pattern | `outputs/diagnose_link.txt` |
| `headline_checks.py` | Compared the re-downloaded files with the first extract | terminal only |

Main findings: the env file has 5 sensors and 3 of them repeat a single timestamp; only 2 of
the 5 occupancy spaces appear in the mapping sheet (labelled incorrect data); no sensor
stood out from the others after the weekly pattern was removed. The audit of the full
occupancy extract is `src/audit_full.py`, which can be rerun.

## Tests and documentation

```
python -m pytest
python -m pdoc src -o docs -d numpy
```

## Layout

- `src/`: library modules and the scripts that produce each result
- `tests/`: pytest tests
- `data/`: `raw/` and `processed/` are not tracked
- `outputs/`: text reports from each script
- `docs/`: generated HTML documentation and `project_log.md`
- `legacy/`: early audit scripts kept as evidence (see above)

## Limitations

- Only five occupancy spaces, and only two can be matched to rooms (the mapping sheet is
  marked as incorrect), so spaces are identified by the first 8 characters of their ID.
- Headcount between events is assumed constant (a step function).
- The environmental sensors could not be linked to any occupancy space.

## AI use

See the AI-use reflection in the report.