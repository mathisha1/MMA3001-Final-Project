"""Step 10b: measured speed of this machine against thread count.

Matrix multiply gives a practical FLOP/s ceiling and chunked copies give a memory
bandwidth ceiling, each with several thread counts. Run it on its own, with nothing else
open, because other jobs would distort the timings."""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

OUTDIR = Path(__file__).resolve().parent.parent / "outputs"
THREADS = [1, 2, 4, 6, 8, 16]
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


def copy_with_threads(dst, src, k):
    """Copy src into dst using k threads, each copying one contiguous chunk."""
    edges = np.linspace(0, len(src), k + 1).astype(int)

    def work(i):
        np.copyto(dst[edges[i]:edges[i + 1]], src[edges[i]:edges[i + 1]])

    with ThreadPoolExecutor(max_workers=k) as pool:
        list(pool.map(work, range(k)))


def main():
    """Measure matrix-multiply and copy speed for each thread count and save the report."""
    pd.set_option("display.width", 200)
    OUTDIR.mkdir(exist_ok=True)
    log("== MATH LIBRARY IN USE ==")
    for lib in threadpool_info():
        log(f"{lib.get('user_api')}: {lib.get('internal_api')}, "
            f"{lib.get('num_threads')} threads, version {lib.get('version')}")

    n = 4000
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    rows = []
    for k in THREADS:
        with threadpool_limits(limits=k, user_api="blas"):
            sec = best_of(lambda: A @ B, 3)
        rows.append({"threads": k, "seconds": sec, "GFLOP_per_s": 2 * n ** 3 / sec / 1e9})
        print(f"  matmul, {k} threads done", flush=True)
    mm = pd.DataFrame(rows)
    mm["speedup_vs_1"] = mm["GFLOP_per_s"] / mm["GFLOP_per_s"].iloc[0]
    log(f"\n== MATRIX MULTIPLY, n = {n} (2 n^3 FLOPs per product) ==")
    log(mm.round(2).to_string(index=False))
    del A, B

    src = np.ones(50_000_000)
    dst = np.empty_like(src)
    rows = []
    for k in THREADS:
        sec = best_of(lambda: copy_with_threads(dst, src, k), 5)
        rows.append({"threads": k, "seconds": sec,
                     "GB_per_s": 2 * src.nbytes / sec / 1e9})   # read + write
        print(f"  copy, {k} threads done", flush=True)
    bw = pd.DataFrame(rows)
    bw["speedup_vs_1"] = bw["GB_per_s"] / bw["GB_per_s"].iloc[0]
    log("\n== COPY BANDWIDTH, 400 MB (counts bytes read plus bytes written) ==")
    log(bw.round(2).to_string(index=False))
    log(f"\nmachine balance with the best of each: {mm['GFLOP_per_s'].max():.0f} GFLOP/s / "
        f"{bw['GB_per_s'].max():.0f} GB/s = {mm['GFLOP_per_s'].max() / bw['GB_per_s'].max():.1f} FLOP/byte")

    (OUTDIR / "hardware_check.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nsaved outputs/hardware_check.txt")


if __name__ == "__main__":
    main()