"""Step 7: robustness of the calendar models.

A. Does error grow with forecast horizon? Rolling origins inside train + validation only.
B. Does a model work for a space it has never seen? Leave one space out.
C. Does the space that exists only in 2024 help or hurt?
The test period is never touched."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation import TARGET, metrics, weekday_day
from src.features import LOCAL_TZ, VAL_END, split_by_time
from src.modeling import load_prepared, model_zoo

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
report = []

# MAE and RMSE printed by step 6 (outputs/models_basic.txt), to check we reproduce it
STEP6 = {"training mean": (0.775, 1.210), "lookup": (0.497, 1.072),
         "linear": (0.559, 1.067), "tree": (0.495, 1.044)}


def log(msg=""):
    """Print a line and keep it for the saved report."""
    print(msg, flush=True)
    report.append(str(msg))


def main():
    """Run checks 0 and parts A, B and C, and save outputs/robustness.txt."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    h = load_prepared()
    parts = split_by_time(h)
    train, val = parts["train"], parts["val"]      # parts["test"] is deliberately never used
    zoo = model_zoo(use_space=True)

    # ---- check 0: do these wrappers reproduce the numbers from step 6?
    log("== CHECK 0: DO THESE MODELS REPRODUCE STEP 6 ON VALIDATION? ==")
    ok = True
    for name, fn in zoo.items():
        m = metrics(val[TARGET], fn(train, val))
        want = STEP6[name]
        match = abs(m["MAE"] - want[0]) < 0.0015 and abs(m["RMSE"] - want[1]) < 0.0015
        ok = ok and match
        log(f"{name}: MAE {m['MAE']:.3f} (step 6: {want[0]:.3f}), "
            f"RMSE {m['RMSE']:.3f} (step 6: {want[1]:.3f}) {'OK' if match else 'MISMATCH'}")
    log("reproduction passed" if ok else "reproduction FAILED: stop and check before using the rest")

    # ---- A: rolling origins. Train on everything before the origin, forecast the next 3 months.
    log("\n== A. ERROR VERSUS FORECAST HORIZON (rolling origin, train + validation data only) ==")
    pool = h[h["local_time"] < VAL_END]
    assert pool["local_time"].max() < VAL_END           # guard: the test period is excluded
    origins = ([pd.Timestamp(2024, m, 1, tz=LOCAL_TZ) for m in range(7, 13)]
               + [pd.Timestamp(2025, m, 1, tz=LOCAL_TZ) for m in range(1, 7)])
    rows = []
    for origin in origins:
        tr = pool[pool["local_time"] < origin]
        te = pool[(pool["local_time"] >= origin)
                  & (pool["local_time"] < origin + pd.DateOffset(months=3))]
        horizon = np.zeros(len(te), int)                 # 1, 2 or 3 months ahead
        for k in (1, 2, 3):
            horizon[(te["local_time"] >= origin + pd.DateOffset(months=k - 1)).to_numpy()] = k
        subsets = (("all hours", np.ones(len(te), bool)), ("weekday 9-5", weekday_day(te)))
        for name, fn in zoo.items():
            p = fn(tr, te)
            for k in (1, 2, 3):
                for subset, extra in subsets:
                    mask = (horizon == k) & extra
                    if mask.sum() < 100:
                        continue
                    m = metrics(te[TARGET][mask], p[mask])
                    rows.append({"origin": origin.strftime("%Y-%m"), "horizon": k,
                                 "subset": subset, "model": name, "MAE": m["MAE"],
                                 "RMSE": m["RMSE"], "mean_actual": te[TARGET][mask].mean()})
    res = pd.DataFrame(rows)
    base = (res[res["model"] == "training mean"][["origin", "horizon", "subset", "MAE"]]
            .rename(columns={"MAE": "MAE_base"}))
    res = res.merge(base, on=["origin", "horizon", "subset"], how="left")
    res["MAE_ratio"] = res["MAE"] / res["MAE_base"]
    log(f"{len(origins)} origins from {origins[0]:%Y-%m} to {origins[-1]:%Y-%m}; each origin is "
        "scored on the following 3 months. MAE_ratio is the model's MAE divided by the "
        "training-mean model's MAE in the same window, which removes seasonal level effects.")
    for subset in ("all hours", "weekday 9-5"):
        s = (res[res["subset"] == subset].groupby(["model", "horizon"])
             .agg(origins=("MAE", "size"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
                  RMSE=("RMSE", "mean"), MAE_ratio=("MAE_ratio", "mean"),
                  mean_actual=("mean_actual", "mean")).round(3))
        log(f"\n-- {subset} --")
        log(s.to_string())

    # ---- B: leave one space out
    log("\n== B. LEAVE ONE SPACE OUT (train period -> validation period) ==")
    log("own space: the model saw this space in training. held out: it never saw this space "
        "and has no space feature. bias = mean predicted minus mean actual (people).")
    own, held = model_zoo(True), model_zoo(False)
    rows = []
    for space in sorted(val["space"].unique()):
        te = val[val["space"] == space]
        day = weekday_day(te)
        for name in own:
            for setting, fn, tr in (("own space", own[name], train),
                                    ("held out", held[name], train[train["space"] != space])):
                p = fn(tr, te)
                m = metrics(te[TARGET], p)
                md = metrics(te[TARGET][day], p[day])
                rows.append({"space": space, "model": name, "setting": setting,
                             "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                             "bias": float(p.mean() - te[TARGET].mean()), "MAE_day": md["MAE"]})
    res = pd.DataFrame(rows)
    log(res.sort_values(["space", "model", "setting"]).round(3).to_string(index=False))
    log("\naverage over the four spaces:")
    log(res.groupby(["model", "setting"])[["MAE", "RMSE", "MAE_day", "bias"]].mean()
        .round(3).to_string())

    # ---- C: drop spaces that never appear in validation
    lone = sorted(set(train["space"]) - set(val["space"]))
    log(f"\n== C. DROP SPACES ABSENT FROM VALIDATION {lone} FROM TRAINING ==")
    day = weekday_day(val)
    rows = []
    for setting, tr in (("all training spaces", train),
                        ("without those spaces", train[~train["space"].isin(lone)])):
        for name, fn in zoo.items():
            p = fn(tr, val)
            m, md = metrics(val[TARGET], p), metrics(val[TARGET][day], p[day])
            rows.append({"training data": setting, "model": name, "train_hours": len(tr),
                         "MAE": m["MAE"], "RMSE": m["RMSE"],
                         "MAE_day": md["MAE"], "RMSE_day": md["RMSE"]})
    log(pd.DataFrame(rows).round(3).to_string(index=False))

    (OUTDIR / "robustness.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nsaved outputs/robustness.txt")


if __name__ == "__main__":
    main()