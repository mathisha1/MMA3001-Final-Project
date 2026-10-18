"""Step 3: how fast does the error of sampled integration shrink as the step h shrinks?

For random windows of 4096 s we compare the exact mean headcount (from the event log)
with trapezoid, Simpson and Romberg estimates built from evenly spaced samples."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.hourly import load_events, cumulative_integral, integral_up_to
from src.integration import sample_step, romberg_table

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

WINDOW_S = 4096.0         # 2**12 seconds, so halving gives whole seconds down to 1 s
K = 12                    # finest grid has 2**12 intervals, so h = 1 s
N_WINDOWS = 1500          # random windows per space
SPACES = ["0f3f37f8", "cca1ac3e"]   # a busy space, and the one with the largest headcounts
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
rng = np.random.default_rng(42)     # fixed seed, so the run is repeatable

report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


def rms(x):
    """Root mean square."""
    return float(np.sqrt(np.mean(np.square(x))))


occ = load_events(PROC / "occ.parquet")
hourly = pd.read_parquet(PROC / "hourly.parquet")
hourly["start_s"] = (hourly["hour_utc"] - EPOCH).dt.total_seconds()

for sp in SPACES:
    g = occ[occ["space"] == sp]
    t = g["t"].to_numpy("float64")
    v = g["headcount"].to_numpy("float64")
    cum = cumulative_integral(t, v)

    # random window starts at hour boundaries, from hours that contain at least one event
    h = hourly[(hourly["space"] == sp) & (hourly["n_events"] > 0)]
    h = h[h["start_s"] + WINDOW_S <= t[-1]]
    starts = rng.choice(h["start_s"].to_numpy(), size=min(N_WINDOWS, len(h)), replace=False)

    # exact mean headcount in each window, in people
    exact = (integral_up_to(starts + WINDOW_S, t, v, cum)
             - integral_up_to(starts, t, v, cum)) / WINDOW_S

    # sample the staircase on the finest grid once; coarser grids reuse these samples
    t0 = time.perf_counter()
    grid = np.linspace(0.0, WINDOW_S, 2 ** K + 1)
    f = sample_step(t, v, starts[:, None] + grid[None, :])
    sample_s = time.perf_counter() - t0
    R = romberg_table(f, WINDOW_S)

    # error of each estimate, in people
    err = {(k, j): R[k][j] / WINDOW_S - exact for k in range(K + 1) for j in range(min(k, 3) + 1)}

    log(f"\n{'=' * 70}\nSPACE {sp}: {len(starts)} windows of {WINDOW_S:.0f} s, "
        f"mean exact headcount {exact.mean():.2f}, sampling took {sample_s:.1f} s")
    for label, mask in [("ALL WINDOWS", np.ones(len(exact), bool)),
                        ("BUSY WINDOWS (mean headcount >= 1)", exact >= 1.0)]:
        rows = []
        for k in range(K + 1):
            row = {"k": k, "n": 2 ** k, "h_s": WINDOW_S / 2 ** k,
                   "trap_bias": err[(k, 0)][mask].mean()}
            for j, name in enumerate(["trap_rms", "simpson_rms", "boole_rms", "romberg3_rms"]):
                row[name] = rms(err[(k, j)][mask]) if k >= j else np.nan
            rows.append(row)
        tab = pd.DataFrame(rows)
        log(f"\n-- {label}: {mask.sum()} windows. RMS error of the window mean, in people --")
        log(tab.round(4).to_string(index=False))

        # fitted log-log slope: error ~ h**slope. Coarse and fine ranges of h
        for name in ["trap_rms", "simpson_rms", "boole_rms"]:
            slopes = []
            for lo, hi in [(2, 6), (7, 10)]:               # coarse: h 1024..64 s, fine: h 32..4 s
                sub = tab[(tab["k"] >= lo) & (tab["k"] <= hi)]
                slopes.append(np.polyfit(np.log(sub["h_s"]), np.log(sub[name]), 1)[0])
            log(f"slope of {name}: coarse (h 1024 to 64 s) {slopes[0]:.2f}, fine (h 32 to 4 s) {slopes[1]:.2f}")

log("\nNotes: textbook slopes would be 2 (trapezoid), 4 (Simpson), 6 (Boole) for smooth functions.")
log("Timestamps are whole seconds, so h = 1 s resolves every jump exactly.")
(OUTDIR / "convergence.txt").write_text("\n".join(report), encoding="utf-8")
print("\nsaved outputs/convergence.txt")