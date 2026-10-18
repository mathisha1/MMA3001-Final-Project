"""Step 8: Gaussian process regression on calendar features: accuracy, uncertainty and cost.

An exact GP needs an n x n matrix, so it is fitted on random subsamples of the training
period. Everything is scored on the validation period; the test period is never touched."""
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cholesky
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from src.evaluation import TARGET, metrics, weekday_day
from src.features import split_by_time
from src.modeling import gp_features, load_prepared, model_zoo

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
SIZES = [250, 500, 1000, 2000]          # subsample sizes for the sweep
EXTRA_SEEDS = [1, 2]                    # repeat the largest size with other random subsamples
BENCH_SIZES = [500, 1000, 2000, 4000, 8000]   # sizes for the factorisation benchmark
report = []


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


def make_kernel(d):
    """Constant x RBF (one length scale per input, so 'ARD') + white noise.

    Inputs are all of order 1, so length scales are bounded to 0.01..100. A length scale at
    100 means that input has no detectable effect."""
    return (ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 10.0)))


def fit_gp(X, y, seed):
    """Fit a GP with one run of the optimiser and count convergence warnings.

    Returns (gp, fit_seconds, optimiser_failures, parameters_at_a_bound)."""
    gp = GaussianProcessRegressor(kernel=make_kernel(X.shape[1]), normalize_y=True,
                                  n_restarts_optimizer=0, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        gp.fit(X, y)
        fit_s = time.perf_counter() - t0
    msgs = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    at_bound = sum("bound" in m for m in msgs)       # "optimal value ... close to the bound"
    return gp, fit_s, len(msgs) - at_bound, at_bound


def predict_gp(gp, X, chunk=4000):
    """Predictive mean and standard deviation, in chunks to limit memory.

    The white-noise term is part of the kernel, so the standard deviation should describe a new
    observed hour, not just the underlying mean (my reading of scikit-learn; the coverage
    check below tests it)."""
    mu, sd = [], []
    for i in range(0, len(X), chunk):
        m, s = gp.predict(X[i:i + chunk], return_std=True)
        mu.append(m)
        sd.append(s)
    return np.concatenate(mu), np.concatenate(sd)


def evaluate_gp(gp, Xv, yv, day):
    """Accuracy, 95% interval coverage and timing of a fitted GP on validation data."""
    t0 = time.perf_counter()
    mu, sd = predict_gp(gp, Xv)
    predict_s = time.perf_counter() - t0
    p = np.clip(mu, 0, None)
    m, md = metrics(yv, p), metrics(yv[day], p[day])
    inside = np.abs(yv - mu) <= 1.96 * sd
    return {"predict_s": predict_s, "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
            "MAE_day": md["MAE"], "coverage95": inside.mean(),
            "coverage95_day": inside[day].mean(), "mean_sd": sd.mean()}


def main():
    """Run the subsample sweep, the same-n comparison and the factorisation benchmark."""
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    h = load_prepared()
    parts = split_by_time(h)
    train = parts["train"].reset_index(drop=True)      # parts["test"] is never used
    val = parts["val"].reset_index(drop=True)
    spaces = sorted(h["space"].unique())
    Xdf = gp_features(train, spaces)
    names = list(Xdf.columns)
    Xtr, ytr = Xdf.to_numpy(), train[TARGET].to_numpy()
    Xva, yva = gp_features(val, spaces).to_numpy(), val[TARGET].to_numpy()
    day = weekday_day(val)
    zoo = model_zoo(True)

    log(f"training hours available: {len(train):,}; validation hours: {len(val):,}; "
        f"GP inputs: {len(names)} ({', '.join(names)})")
    rows_gp, rows_cmp, kept = [], [], None
    runs = [(n, 0) for n in SIZES] + [(SIZES[-1], s) for s in EXTRA_SEEDS]
    for n, seed in runs:
        idx = np.sort(np.random.default_rng(seed).choice(len(train), size=n, replace=False))
        gp, fit_s, n_fail, n_bound = fit_gp(Xtr[idx], ytr[idx], seed)
        ev = evaluate_gp(gp, Xva, yva, day)
        rows_gp.append({"n": n, "seed": seed, "fit_s": fit_s, "predict_s": ev["predict_s"],
                        "K_MB": n * n * 8 / 1e6, "logML": gp.log_marginal_likelihood_value_,
                        "opt_not_converged": n_fail, "param_at_bound": n_bound,
                        "MAE": ev["MAE"], "RMSE": ev["RMSE"], "R2": ev["R2"],
                        "MAE_day": ev["MAE_day"], "coverage95": ev["coverage95"],
                        "coverage95_day": ev["coverage95_day"], "mean_sd": ev["mean_sd"]})
        rows_cmp.append({"n": n, "seed": seed, "model": "gp", "MAE": ev["MAE"],
                         "RMSE": ev["RMSE"], "R2": ev["R2"], "MAE_day": ev["MAE_day"]})
        sub = train.iloc[idx]                           # the other models get the same hours
        for name, fn in zoo.items():
            p = fn(sub, val)
            m, md = metrics(yva, p), metrics(yva[day], p[day])
            rows_cmp.append({"n": n, "seed": seed, "model": name, "MAE": m["MAE"],
                             "RMSE": m["RMSE"], "R2": m["R2"], "MAE_day": md["MAE"]})
        if (n, seed) == (SIZES[-1], 0):
            kept = gp
        print(f"  done n={n}, seed={seed}: fit {fit_s:.1f} s, validation MAE {ev['MAE']:.3f}",
              flush=True)
    for name, fn in zoo.items():                        # the same models on ALL training hours
        p = fn(train, val)
        m, md = metrics(yva, p), metrics(yva[day], p[day])
        rows_cmp.append({"n": len(train), "seed": 0, "model": name, "MAE": m["MAE"],
                         "RMSE": m["RMSE"], "R2": m["R2"], "MAE_day": md["MAE"]})

    log("\n== A. GP FIT AND VALIDATION, BY SUBSAMPLE SIZE ==")
    log("opt_not_converged = optimiser warnings; param_at_bound = hyperparameters stuck at a "
        "bound; coverage95 = share of validation hours inside the 95% band (ideal 0.95)")
    log(pd.DataFrame(rows_gp).round(3).to_string(index=False))

    cmp = pd.DataFrame(rows_cmp)
    log("\n== B. SAME-n COMPARISON (every model fitted on the same subsample; last column = all "
        "training hours, GP excluded) ==")
    for col in ("MAE", "RMSE", "MAE_day"):
        log(f"\n{col} (seed 0):")
        log(cmp[cmp["seed"] == 0].pivot_table(index="model", columns="n", values=col)
            .round(3).to_string())
    log(f"\nn = {SIZES[-1]}, three random subsamples (mean and standard deviation):")
    log(cmp[cmp["n"] == SIZES[-1]].groupby("model")[["MAE", "RMSE", "MAE_day"]]
        .agg(["mean", "std"]).round(3).to_string())

    log(f"\n== C. LEARNED KERNEL (n = {SIZES[-1]}, seed 0) ==")
    k = kept.kernel_
    log(f"amplitude {k.k1.k1.constant_value:.3f}; noise level {k.k2.noise_level:.3f} "
        "(both in units of the standardised target variance)")
    log("length scale per input (large = input barely matters; 100 = at the upper bound):")
    log(pd.Series(k.k1.k2.length_scale, index=names).round(2).to_string())

    log("\n== D. COST OF ONE EXACT FACTORISATION (kernel from section C, fixed) ==")
    rng = np.random.default_rng(123)
    out = []
    for n in BENCH_SIZES:
        Xs = Xtr[rng.choice(len(train), size=n, replace=False)]
        t0 = time.perf_counter()
        K = k(Xs)
        build_s = time.perf_counter() - t0
        best = np.inf
        try:
            for _ in range(3):
                t0 = time.perf_counter()
                cholesky(K, lower=True, check_finite=False)
                best = min(best, time.perf_counter() - t0)
        except LinAlgError:
            best = np.nan
        del K
        out.append({"n": n, "K_MB": n * n * 8 / 1e6, "build_K_s": build_s,
                    "cholesky_s": best, "GFLOP_per_s": (n ** 3 / 3) / best / 1e9})
        print(f"  benchmark n={n}: cholesky {best:.3f} s", flush=True)
    bench = pd.DataFrame(out)
    log("FLOPs for a Cholesky factorisation are about n^3 / 3, and GFLOP_per_s uses that count")
    log(bench.round(3).to_string(index=False))
    ok = bench[(bench["n"] >= 1000) & bench["cholesky_s"].notna()]
    if len(ok) >= 3:
        slope = np.polyfit(np.log(ok["n"]), np.log(ok["cholesky_s"]), 1)[0]
        log(f"log-log slope of factorisation time against n (n >= 1000): {slope:.2f} (cubic = 3)")
    speed = bench["GFLOP_per_s"].dropna()
    if len(speed):
        n_full = len(train)
        log(f"all {n_full:,} training hours: the matrix alone would need "
            f"{n_full ** 2 * 8 / 1e9:.1f} GB, and one factorisation would take about "
            f"{(n_full ** 3 / 3) / (speed.iloc[-1] * 1e9) / 60:.0f} minutes at the measured "
            f"{speed.iloc[-1]:.0f} GFLOP/s (extrapolated, not measured)")
    fit_row = [r for r in rows_gp if r["n"] == SIZES[-1] and r["seed"] == 0][0]
    chol_row = bench[bench["n"] == 2000]["cholesky_s"]
    if len(chol_row) and chol_row.iloc[0] > 0:
        log(f"fit time at n = 2000 is {fit_row['fit_s'] / chol_row.iloc[0]:.0f} times one "
            "factorisation, because the optimiser evaluates the likelihood many times")

    (OUTDIR / "gp_study.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nsaved outputs/gp_study.txt")


if __name__ == "__main__":
    main()