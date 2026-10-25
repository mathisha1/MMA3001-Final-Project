# Project log

## Decisions and the evidence for them
- Dropped the environmental sensors. Three of the five env sensors have frozen clocks
  (they repeat one or two timestamps for every reading), only 2 of the 5 occupancy spaces
  appear in the mapping sheet (which is labelled incorrect data), and after removing the
  weekly pattern no env sensor stood out from the others
  (outputs/audit_full.txt, outputs/diagnose_overlap.txt, outputs/diagnose_link.txt).
- Treated silence between occupancy events as an empty room. 87% of long gaps (over 10 min)
  follow a NotOccupied event, and 99.4% of hours with no events hold a last-logged value of 0
  (outputs/event_semantics.txt, outputs/hourly_comparison.txt).
- Used the exact time-weighted hourly mean as the target. The plain mean of logged events is
  biased high by 0.42 people on average (hours with events average 0.97 people)
  (outputs/hourly_comparison.txt).
- Higher-order integration rules (Simpson, Boole, Romberg) were no better than the trapezoid
  rule, which was the most accurate at every step size. Error fell at roughly first order
  because the headcount is a step function (outputs/convergence.txt). Timestamps all fall on
  even seconds, so slopes were fitted over h = 32 to 4 s to avoid the 2 s artefact.
- Time-based split: train to 2024-12-31, validation 2025-01 to 2025-08, test 2025-09 to
  2026-04. Models were tuned on validation, then refitted on train + validation and scored
  on the test period once.
- Built the term calendar (teaching, non-teaching, closure) by hand from the Monash key
  dates. The term feature improved the lookup baseline on validation (MAE 0.525 to 0.497
  overall, 1.033 to 0.953 on weekday daytime).
- Fitted the Gaussian process on random 2,000-hour subsamples (three seeds), because an exact
  GP on all training hours needs a 12.6 GB matrix and cost grows with n cubed.
- Set a rule before the test run: the linear variant with the lower validation RMSE becomes
  primary, and ties go to the simpler one. Additive won (1.067 against 1.075), so it stayed
  primary even though the interaction variant won 10 of 12 rolling origins and was better on
  test (outputs/linear_variants.txt, outputs/final_test.txt). I report both.
- Final test result: lookup was best or tied-best (MAE 0.455, RMSE 1.013, R2 0.305). Tree,
  linear and the GP did not reliably beat it, and the validation ranking reversed. I report
  the reversal as an open finding.

## Where the AI was wrong or needed correcting
- Suggested methods I had not been taught (Poisson regression, Kalman filter, random forest).
  I supplied the list of taught methods.
- Assumed headcount was small (0 to 5) from a sample. In the full data it reaches 72 people.
- Said env sensor 326 covered the occupancy period. Apart from one stray reading in Dec 2023,
  its real data starts in Sep 2025.
- Predicted the plain-mean bias would be largest in busy afternoons. It is largest around
  arrival and departure hours.
- Predicted a coarse convergence slope near 0.5. I measured 0.65 to 0.70.
- The first slope fit included the h = 2 s row, where all timestamps fall on even seconds, so
  the slopes were misleading. I refitted over h = 32 to 4 s.
- A drift-check script printed test-period averages. I removed that, so later exploration used
  only train and validation data.
- Scripts written for the truncated Excel extract would have run out of memory on 9M rows.
- Environment problems: the virtual environment broke when I moved the folder out of OneDrive,
  so pytest is run as `python -m pytest`.
- Predicted that linear and tree would beat lookup on the test period. They did not, and the
  validation ranking reversed.
- Guessed LPDDR5X memory (about 134 GB/s). It is DDR5-5600 on one module (about 45 GB/s peak).
- Quoted "94% of ceiling" from a noisy run. A repeat gave 105 to 312 GFLOP/s for the same
  matrix product, so I withdrew it.
- Assumed 8 FLOPs per cycle on the efficiency cores. Sources say 16 (4 x 128-bit FMA units),
  so the first peak estimate was too low.
- Gave figure edits that would have crashed (pasted into the wrong plot) and a zero line on a
  log axis.
- A bug in the GP summary section of final_test.py, caught before the run.
- Offered "lookup benefits most from more data" as the reason for the validation-to-test
  reversal. The learning curve did not support it, so I treat the reversal as an open finding.
- A wrapping recipe so ambiguous that load_events stayed outside main() in two scripts. The
  import test caught it and I fixed it.
- Described the five occupancy spaces as "workshop spaces" in the scope message, but the data
  does not identify what they are. The report calls them "monitored spaces".

## What I verified myself
- Found the Excel cap (1,048,575 rows) and re-downloaded the full CSV (9,091,732 rows).
- Found that all timestamps fall on even seconds, which explains the error collapse at h = 2 s.
- Unit tests: 22 at first, with hand-calculated expected values, then one import test per
  script (47 passing), then a brute-force check of the hourly integral (48).
- Fixed the wrapped scripts, reran them, and confirmed the key numbers still match
  (event-mean bias 0.422, tree depth 6, lookup validation MAE 0.497, convergence slopes).
- Ran a dry run that reproduced all seven validation numbers before the test period was used.
- Froze the configuration, committed it, and ran the test evaluation once.
- Checked each figure against its table.
- Measured the hardware: DDR5-5600 from Task Manager, repeated benchmarks, noted the run-to-run
  variation (105 to 312 GFLOP/s for the same matrix product).
- Spotted AI mistakes (nested folders, misplaced figure code, wrong file paths).
- Worked out the trapezoid test value by hand: 0.3359375 for x^2 on [0, 1] with 8 intervals.
- Theoretical peak = cores x clock x FLOPs per cycle. Lion Cove P-cores (6): 2 x 256-bit FMA =
  16 FLOPs/cycle. Skymont E-cores (8): 4 x 128-bit FMA = 16. Crestmont low-power cores (2): 8
  (inferred). That gives 395 GFLOP/s at base clocks and 1093 at max turbo, and about 672
  assuming 3.0 GHz sustained on P and E cores. Best measured matrix multiply was 312 GFLOP/s.
- Recomputed lookup's test MAE (0.455) and RMSE (1.013) from final_predictions.parquet.
- Not done yet: fresh-clone check and zipped clone (planned after submission).

## AI-use disclosure 
- Tool: Claude (Anthropic), in a chat interface.
- Used for: planning the project, writing most of the code, drafting explanations, suggesting
  checks, and interpreting outputs.
- Approximate contribution: AI wrote most of the code and first-draft
  explanations.
- How checked: tests with hand-worked values, the dry run, reproducing numbers myself, reading
  outputs and figures, and catching the errors listed above.
- My decisions: dropping the env side after the audit, the problem statement and its three
  parts, keeping the convergence and robustness questions, supplying the taught-methods list,
  building the term calendar, getting the scope approved, setting the time split, and deciding
  when to stop.
- Errors and limits: see "Where the AI was wrong".
- Effect on my understanding: Using AI for the code saved time and progressed the completion of the report, I used it to improve my understanding of the content. 