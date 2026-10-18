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

```
python -m src.convert_to_parquet   # raw files -> data/processed/*.parquet
python -m src.audit_full           # data audit, writes env_long.parquet
python -m src.event_checks         # what silence between occupancy events means
python -m src.build_hourly         # hourly series three ways, error comparison
python -m src.convergence          # trapezoid / Simpson / Romberg convergence study
python -m src.baselines            # calendar baselines
python -m src.baselines_term       # term-aware baselines
python -m src.models_basic         # linear regression and decision tree
python -m src.robustness           # horizon, leave-one-space-out, drop-lone-space
python -m src.gp_study             # Gaussian process study (slow, run on its own)
```

Each script prints its results and saves a copy in `outputs/`. The scripts
`peek_data`, `audit_data`, `diagnose_overlap`, `diagnose_link` and `headline_checks` were used
on the first (truncated) extract and are kept as the audit trail.

## Tests and documentation

```
python -m pytest
python -m pdoc src.hourly src.integration src.features src.evaluation src.modeling -o docs
```

## Layout

- `src/`: library modules and the scripts that produce each result
- `tests/`: pytest tests
- `data/`: `raw/` and `processed/` are not tracked
- `outputs/`: text reports from each script
- `docs/`: generated HTML documentation and `project_log.md`

## Limitations

- Only five occupancy spaces, and only two can be matched to rooms (the mapping sheet is
  marked as incorrect), so spaces are identified by the first 8 characters of their ID.
- Headcount between events is assumed constant (a step function).
- The environmental sensors could not be linked to any occupancy space.

## AI use

See the AI-use reflection in the report.