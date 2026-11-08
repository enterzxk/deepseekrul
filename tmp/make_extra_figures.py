"""Extra paper figures: error vs observation age, error by lifetime quartile, measurement conditions."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper_figures"; OUT.mkdir(exist_ok=True)


def load(base, subset, arm, seed=101):
    b = []
    for k in range(5):
        p = Path(base) / subset / str(k) / f"{arm}_{seed}.csv"
        if not p.is_file():
            return None
        b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b, ignore_index=True)


def nasa(e):
    e = np.asarray(e, float)
    return np.where(e > 0, np.expm1(e / 10.0), np.expm1(-e / 13.0))


# 1. error vs observation age
fig, ax = plt.subplots(figsize=(5.2, 3.2))
bins = [(50, 100), (100, 150), (150, 200), (200, 400)]
for subset in ["FD002", "FD004"]:
    for arm, ls in [("M0", "--"), ("FULL", "-")]:
        d = load("runs", subset, arm)
        if d is None:
            continue
        d["err"] = d.y_pred - d.y_true
        xs, ys = [], []
        for lo, hi in bins:
            s = d[(d.end_cycle >= lo) & (d.end_cycle < hi)]
            if len(s) < 50:
                continue
            xs.append((lo + hi) / 2); ys.append(float(np.sqrt((s.err ** 2).mean())))
        ax.plot(xs, ys, ls, marker="o", label=f"{subset} {arm}", alpha=.85)
ax.set_xlabel("observed cycles at the window end"); ax.set_ylabel("RMSE (cycles)")
ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_error_by_age.png", dpi=200); fig.savefig(OUT / "fig_error_by_age.pdf")
plt.close(fig)

# 2. error by lifetime quartile
fig, ax = plt.subplots(figsize=(5.2, 3.2))
labels = ["Q1 short", "Q2", "Q3", "Q4 long"]
width = 0.35
for i, subset in enumerate(["FD002", "FD004"]):
    for j, arm in enumerate(["M0", "FULL"]):
        d = load("runs", subset, arm)
        if d is None:
            continue
        d["err"] = d.y_pred - d.y_true
        life = d.groupby("unit").end_cycle.max()
        q = pd.qcut(life, 4, labels=False)
        vals = []
        for k in range(4):
            u = q.index[q == k]
            s = d[d.unit.isin(u)]
            vals.append(float(np.sqrt((s.err ** 2).mean())) if len(s) else np.nan)
        ax.bar(np.arange(4) + (i * 2 + j) * width - 1.5 * width, vals, width=width,
               label=f"{subset} {arm}")
ax.set_xticks(range(4)); ax.set_xticklabels(labels)
ax.set_ylabel("RMSE (cycles)"); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_error_by_life_quartile.png", dpi=200)
fig.savefig(OUT / "fig_error_by_life_quartile.pdf"); plt.close(fig)

# 3. measurement conditions
mc = HERE / "MEASUREMENT_CONDITIONS.csv"
if mc.is_file():
    d = pd.read_csv(mc)
    conds = [c for c in ["clean", "mc1", "mc2"] if c in set(d.condition)]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    arms = sorted(d.arm.unique())
    w = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        vals = [float(d[(d.arm == arm) & (d.condition == c)].rmse.mean()) for c in conds]
        ax.bar(np.arange(len(conds)) + i * w - 0.4 + w / 2, vals, width=w, label=arm)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds)
    ax.set_ylabel("pooled RMSE (cycles)"); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_measurement_conditions.png", dpi=200)
    fig.savefig(OUT / "fig_measurement_conditions.pdf"); plt.close(fig)
    print("measurement figure written")
else:
    print("no MEASUREMENT_CONDITIONS.csv yet")
print("extra figures written to", OUT)
