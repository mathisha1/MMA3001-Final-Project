"""Step 10: performance study.

A. memory: compact event table versus reading the needed columns as text
B. where the loading time goes (cProfile)
C. time to build all hourly series
D. exact cumulative-integral method versus a per-hour scan: time, scaling, result check
E. empirical ceilings of this machine: matrix multiply GFLOP/s and copy bandwidth
F. Cholesky factorisation scaling on random positive definite matrices
G. arithmetic intensity (hand count) against the measured ceilings

Run it on its own: other jobs would distort the timings."""
import cProfile
import io
import pstats
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.linalg import cholesky
from src.hourly import HOUR_S, hourly_series, hourly_time_weighted, load_events

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "processed" / "occ.parquet"
OUTDIR = ROOT / "outputs"
report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


def best_of(fn, repeats):
    """Smallest wall time in seconds over several calls of fn()."""
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def naive_hourly(t, v, starts):
    """Mean headcount per hour by scanning all events once per hour (O(N x H))."""
    out = np.empty(len(starts))
    for k, a in enumerate(starts):
        b = a + HOUR_S
        inside = (t > a) & (t < b)                           # scans every event
        i0 = np.searchsorted(t, a, side="right") - 1         # event in force at the start
        times = np.concatenate(([a], t[inside], [b]))
        vals = np.concatenate(([v[i0]], v[inside]))
        out[k] = np.sum(vals * np.diff(times)) / HOUR_S
    return out


def main():
    """Run parts A to G and save outputs/performance.txt."""
    pd.set_option("display.width", 200)
    OUTDIR.mkdir(exist_ok=True)

    # ---- A. memory of the compact table versus text columns
    pf = pq.ParquetFile(PATH)
    n_rows = pf.metadata.num_rows
    batch = next(pf.iter_batches(batch_size=1_000_000,
                                 columns=["floorspaceid", "headcount", "collecteddate"]))
    text_mb = batch.to_pandas().memory_usage(deep=True).sum() / 1e6
    log("== A. MEMORY ==")
    log(f"3 needed columns as text: {text_mb:.0f} MB per million rows, about "
        f"{text_mb * n_rows / 1e9:.1f} GB for all {n_rows:,} rows (extrapolated from the first million)")

    # ---- B. profile of loading
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    occ = load_events(PATH)
    pr.disable()
    load_s = time.perf_counter() - t0
    compact_mb = occ.memory_usage(deep=True).sum() / 1e6
    log(f"compact table actually used (space category, time float64, headcount float32): "
        f"{compact_mb:.0f} MB for {len(occ):,} rows")
    log(f"\n== B. LOADING: {load_s:.0f} s wall time under the profiler (profiler adds some overhead) ==")
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumulative").print_stats(12)
    log(buf.getvalue())

    # ---- C. build all hourly series
    t0 = time.perf_counter()
    tables = [hourly_series(g["t"].to_numpy("float64"), g["headcount"].to_numpy("float64"), str(sp))
              for sp, g in occ.groupby("space", observed=True)]
    build_s = time.perf_counter() - t0
    n_hours = sum(len(t) for t in tables)
    log(f"== C. BUILD HOURLY SERIES: {n_hours:,} hours from {len(occ):,} events in {build_s:.2f} s ==")

    # ---- D. exact cumulative method versus per-hour scan
    g = occ[occ["space"] == "0f3f37f8"].head(200_000)
    t, v = g["t"].to_numpy("float64"), g["headcount"].to_numpy("float64")
    first = np.ceil(t[0] / HOUR_S) * HOUR_S
    max_h = int((t[-1] - first) // HOUR_S) - 1
    sizes = [x for x in (50, 100, 200, 400, 800) if x <= max_h]
    n_ev = len(t)
    rows = []
    for H in sizes:
        starts = first + HOUR_S * np.arange(H)
        t_naive = best_of(lambda: naive_hourly(t, v, starts), 3)
        t_vec = best_of(lambda: hourly_time_weighted(t, v, starts), 20)
        same = bool(np.allclose(naive_hourly(t, v, starts), hourly_time_weighted(t, v, starts),
                                atol=1e-6, rtol=0))
        rows.append({"hours": H, "naive_ms": t_naive * 1e3, "cumulative_ms": t_vec * 1e3,
                     "speedup": t_naive / t_vec, "same_result": same,
                     "naive_comparisons": 2 * n_ev * H, "cumulative_flops": 3 * n_ev + 8 * H})
    log(f"\n== D. HOURLY MEAN: PER-HOUR SCAN VERSUS CUMULATIVE INTEGRAL ({n_ev:,} events) ==")
    log(pd.DataFrame(rows).round(3).to_string(index=False))
    log("naive: every hour scans every event, so work grows with events x hours. "
        "cumulative: one pass over the events (about 3 FLOPs each) plus a binary search per hour.")

    # ---- E. empirical ceilings
    rng = np.random.default_rng(0)
    n = 4000
    A, B = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    gemm_gflops = 2 * n ** 3 / best_of(lambda: A @ B, 3) / 1e9
    del A, B
    big = np.ones(50_000_000)
    dst = np.empty_like(big)
    bw_gbs = 2 * big.nbytes / best_of(lambda: np.copyto(dst, big), 5) / 1e9
    del big, dst
    log("\n== E. EMPIRICAL CEILINGS OF THIS MACHINE (NumPy, all cores) ==")
    log(f"matrix multiply, n = {n}: {gemm_gflops:.0f} GFLOP/s")
    log(f"copy of 400 MB: {bw_gbs:.1f} GB/s moved (read + write)")
    log(f"machine balance (GFLOP/s per GB/s): {gemm_gflops / bw_gbs:.1f} FLOP/byte")

    # ---- F. Cholesky scaling
    rows, prev = [], None
    for n in (1000, 2000, 4000, 8000, 12000):
        try:
            M = rng.standard_normal((n, n))
            M = (M + M.T) / 2 + n * np.eye(n)                # symmetric, diagonally dominant
            sec = best_of(lambda: cholesky(M, lower=True, check_finite=False), 3)
        except MemoryError:
            log(f"n = {n}: not enough memory, skipped")
            break
        del M
        rows.append({"n": n, "matrix_MB": 8 * n * n / 1e6, "seconds": sec,
                     "GFLOP_per_s": (n ** 3 / 3) / sec / 1e9,
                     "local_exponent": np.nan if prev is None
                     else np.log(sec / prev[1]) / np.log(n / prev[0])})
        prev = (n, sec)
    ch = pd.DataFrame(rows)
    log("\n== F. CHOLESKY FACTORISATION, RANDOM SPD MATRICES (FLOPs counted as n^3 / 3) ==")
    log(ch.round(3).to_string(index=False))
    big_n = ch[ch["n"] >= 4000]
    if len(big_n) >= 3:
        slope = np.polyfit(np.log(big_n["n"]), np.log(big_n["seconds"]), 1)[0]
        log(f"log-log slope for n >= 4000: {slope:.2f} (cubic = 3)")

    # ---- G. arithmetic intensity by hand count
    log("\n== G. ARITHMETIC INTENSITY (hand count) ==")
    ai_int = (3 * n_ev) / (56 * n_ev)
    log(f"cumulative integral: about 3 FLOPs per event; if np.diff, the multiply and cumsum each "
        f"move their arrays through memory that is 56 bytes per event, so AI = {ai_int:.3f} FLOP/byte "
        f"(pessimistic, arrays this small mostly stay in cache). Machine balance is "
        f"{gemm_gflops / bw_gbs:.1f}, so it is memory-bound in principle.")
    for n in (2000, 8000, 39713):
        log(f"Cholesky n = {n}: {n ** 3 / 3:.2e} FLOPs, matrix {8 * n * n / 1e9:.2f} GB, "
            f"ideal AI = (n^3/3) / (8 n^2) = n/24 = {n / 24:.0f} FLOP/byte (upper bound, "
            f"assumes the matrix moves once), far above machine balance, so compute-bound")
    if len(ch):
        log(f"largest Cholesky reached {ch['GFLOP_per_s'].iloc[-1]:.0f} GFLOP/s, "
            f"{ch['GFLOP_per_s'].iloc[-1] / gemm_gflops:.0%} of the matrix-multiply ceiling")

    (OUTDIR / "performance.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nsaved outputs/performance.txt")


if __name__ == "__main__":
    main()