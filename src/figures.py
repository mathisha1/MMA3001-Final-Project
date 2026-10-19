"""Figures from the saved predictions. Usage: python -m src.figures [--dry-run]"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")                 # draw straight to files, no window
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["lookup", "linear", "tree", "GP seed 0"]


def main():
    """Draw the four figures from final_predictions[_dry].parquet."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    suffix = "_dry" if ap.parse_args().dry_run else ""
    df = pd.read_parquet(ROOT / "data" / "processed" / f"final_predictions{suffix}.parquet")
    figdir = ROOT / "outputs" / f"figures{suffix}"
    figdir.mkdir(parents=True, exist_ok=True)
    df["hour"] = df["local_time"].dt.hour
    df["dow"] = df["local_time"].dt.dayofweek

    # 1. weekday profile: average actual and predicted headcount by hour
    g = df[df["dow"] < 5].groupby("hour")[["actual"] + MODELS].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(g.index, g["actual"], color="black", linewidth=2.5, label="actual")
    for m in MODELS:
        ax.plot(g.index, g[m], label=m)
    # the other two GP subsamples, dotted in the same colour as GP seed 0
    gp_colour = ax.lines[-1].get_color()          # last line plotted so far is GP seed 0
    wk = df[df["dow"] < 5].groupby("hour")[["GP seed 1", "GP seed 2"]].mean()
    for m in wk.columns:
        ax.plot(wk.index, wk[m], color=gp_colour, linestyle=":", linewidth=1.2, label=m)
    ax.set_xlabel("hour of day (Melbourne time)")
    ax.set_ylabel("mean headcount (people)")
    ax.set_title("Weekdays: average actual and predicted headcount by hour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "weekday_profile.png", dpi=200)
    plt.close(fig)

    # 2. predicted versus actual, density of hours
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, m in zip(axes, ["tree", "GP seed 0"]):
        hb = ax.hexbin(df["actual"], df[m], gridsize=40, extent=(0, 10, 0, 10),
                       bins="log", mincnt=1)
        ax.plot([0, 10], [0, 10], "r--", linewidth=1)
        ax.set_title(m)
        ax.set_xlabel("actual (people)")
    # one overall title, written once after the loop
    fig.suptitle(f"{(df['actual'] > 10).mean():.1%} of hours with actual above 10 people "
                 "are not shown", fontsize=9)
    axes[0].set_ylabel("predicted (people)")
    fig.colorbar(hb, ax=axes, label="hours (log scale)")
    fig.savefig(figdir / "predicted_vs_actual.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. residual distributions (clipped at +/- 6 people so the bulk is visible)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in MODELS:
        ax.hist((df["actual"] - df[m]).clip(-6, 6), bins=60, histtype="step", label=m)
    ax.set_yscale("log")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")   # zero error
    ax.set_xlabel("actual minus predicted (people), clipped at +/- 6")
    ax.set_ylabel("hours (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "residuals.png", dpi=200)
    plt.close(fig)

    # 4. GP band for a typical week: the space with most hours, the week whose total
    #    headcount is closest to the median (chosen by a fixed rule, not by looks)
    sp = df["space"].value_counts().idxmax()
    d = df[df["space"] == sp].sort_values("local_time").copy()
    iso = d["local_time"].dt.isocalendar()
    d["wk"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    tot = d.groupby("wk")["actual"].agg(["sum", "size"])
    tot = tot[tot["size"] >= 160]
    pick = (tot["sum"] - tot["sum"].median()).abs().idxmin()
    w = d[d["wk"] == pick]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(w["local_time"], w["GP seed 0 mu"] - 1.96 * w["GP seed 0 sd"],
                    w["GP seed 0 mu"] + 1.96 * w["GP seed 0 sd"], alpha=0.25,
                    label="GP 95% band (seed 0)")
    ax.plot(w["local_time"], w["GP seed 0 mu"], label="GP mean (seed 0)")
    ax.plot(w["local_time"], w["actual"], color="black", linewidth=1, label="actual")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")   # people cannot go below zero
    ax.set_ylabel("headcount (people)")
    ax.set_title(f"Space {sp}, ISO week {pick % 100} of {pick // 100}: GP prediction and 95% band")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(figdir / "gp_band_week.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures in {figdir}")


if __name__ == "__main__":
    main()