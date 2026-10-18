# Project log

# PARAPHRASE AFTERWARDS

## Decisions and the evidence for them
- Dropped the environmental link: 3 of 5 env sensors repeat one timestamp, only 2 of 5
  occupancy spaces appear in the mapping sheet, and no sensor stood out after removing the
  weekly rhythm (outputs/audit_full.txt, outputs/diagnose_link.txt).
- Silence between occupancy events means empty: 87% of long gaps follow a NotOccupied event
  and 99.4% of event-free hours hold a value of 0 (outputs/event_semantics.txt,
  outputs/hourly_comparison.txt).
- Target is the exact time-weighted hourly mean. The plain mean of logged events is biased
  high by 0.42 people (outputs/hourly_comparison.txt).
- Higher-order integration rules do not help on this step-function data
  (outputs/convergence.txt).
- Split by time: train to 2024-12-31, validation 2025-01 to 2025-08, test from 2025-09,
  which is used once at the end.

## Where the AI was wrong or needed correcting
- Suggested methods that were not taught (Poisson regression, Kalman filter, random forest);
  I supplied the list of taught methods.
- Assumed headcount was small (0 to 5) from a sample; the full data reaches 72.
- Said env sensor 326 covered the occupancy period; its real data starts in Sep 2025.
- Predicted the plain-mean bias would be worst in busy afternoons; it peaks at arrival and
  departure hours.
- Predicted a coarse convergence slope near 0.5; measured 0.65 to 0.70.
- First slope fit included the h = 2 s row, where all timestamps fall on even seconds, so
  the slopes were misleading; refitted over h = 32 to 4 s.
- A drift-check script printed test-period averages; removed so exploration only uses
  train and validation.
- Scripts written for the truncated extract would have run out of memory on 9M rows.
- Environment problems: virtual environment broke after moving the folder out of OneDrive,
  so pytest runs as `python -m pytest`.

## What I verified myself
(fill in: which numbers I recomputed or checked, and how)