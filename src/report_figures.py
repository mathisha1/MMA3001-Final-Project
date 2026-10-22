"""Report figures: convergence study, tree overfitting, cost of the Gaussian process."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from src.evaluation import TARGET, metrics
from src.features import split_by_time
from src.modeling import load_prepared, tree_features

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "outputs" / "figures_report"


def convergence_figure():
    """Log-log error against step size, from outputs/convergence_table.csv."""
    tab = pd.read_csv(ROOT / "outputs" / "convergence_table.csv")
    tab = tab[tab["subset"] == "ALL WINDOWS"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (sp, g) in zip(axes, tab.groupby("space")):
        g = g.sort_values("h_s")
        for col, label in [("trap_rms", "trapezoid"), ("simpson_rms", "Simpson"),
                           ("boole_rms", "Boole"), ("romberg3_rms", "Romberg level 3")]:
            ax.loglog(g["h_s"], g[col], marker="o", markersize=3, label=label)
        ref = g.loc[g["h_s"] == 32, "trap_rms"].iloc[0]
        hh = np.array([4.0, 1024.0])
        ax.loglog(hh, ref * (hh / 32) ** 1, "k:", label="slope 1")
        ax.loglog(hh, ref * (hh / 32) ** 2, "k--", label="slope 2")
        ax.axvline(2, color="grey", linewidth=0.6)     # timestamps fall on even seconds
        ax.set_ylim(5e-5, 3)
        ax.set_title(f"space {sp}")
        ax.set_xlabel("step h (s); grey line = 2 s timestamp resolution")
    axes[0].set_ylabel("RMS error of window mean (people)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "convergence.png", dpi=200)
    plt.close(fig)


def tree_figure():
    """Training and validation RMSE against tree depth (minimum leaf 50)."""
    h = load_prepared()
    parts = split_by_time(h)
    train, val = parts["train"], parts["val"]          # the test period is not used
    spaces = sorted(h["space"].unique())
    Xtr, Xva = tree_features(train, spaces), tree_features(val, spaces)
    rows = []
    for depth in [2, 3, 4, 6, 8, 10, 14]:
        m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=50, random_state=0)
        m.fit(Xtr, train[TARGET])
        rows.append({"depth": depth, "train": metrics(train[TARGET], m.predict(Xtr))["RMSE"],
                     "val": metrics(val[TARGET], m.predict(Xva))["RMSE"]})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    print("check: depth 6 should give train RMSE 1.283 and validation RMSE 1.044 (outputs/models_basic.txt)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res["depth"], res["train"], marker="o", label="training period")
    ax.plot(res["depth"], res["val"], marker="o", label="validation period")
    ax.axvline(6, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("tree depth (minimum leaf 50)")
    ax.set_ylabel("RMSE (people)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "tree_depth.png", dpi=200)
    plt.close(fig)


def cost_figure():
    """Measured time against n for GP fits and for one Cholesky factorisation."""
    # copied by hand from outputs/gp_study.txt section A (seed 0) and outputs/final_test.txt section I
    val_n, val_s = [250, 500, 1000, 2000], [1.967, 4.766, 27.064, 82.198]
    fin_n, fin_s = [2000, 2000, 2000, 3000], [115.791, 75.767, 731.471, 583.514]
    # copied by hand from outputs/performance.txt section F
    ch_n, ch_s = np.array([1000, 2000, 4000, 8000, 12000]), np.array([0.034, 0.091, 0.281, 1.070, 2.774])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(val_n, val_s, "o-", label="GP fit, validation stage")
    ax.loglog(fin_n, fin_s, "s", label="GP fit, final run (same n varies)")
    ax.loglog(ch_n, ch_s, "^-", label="one Cholesky factorisation")
    nn = np.array([250.0, 12000.0])
    ax.loglog(nn, ch_s[3] * (nn / 8000) ** 3, "k--", linewidth=0.8, label="slope 3 reference")
    ax.set_xlabel("n (training hours)")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "gp_cost.png", dpi=200)
    plt.close(fig)


def main():
    """Draw all report figures into outputs/figures_report."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    if (ROOT / "outputs" / "convergence_table.csv").exists():
        convergence_figure()
    else:
        print("convergence_table.csv not found: patch and rerun src.convergence first")
    tree_figure()
    cost_figure()
    print(f"saved figures in {FIGDIR}")


if __name__ == "__main__":
    main()