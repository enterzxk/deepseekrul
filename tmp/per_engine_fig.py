"""Per-engine scatter figure: M0 vs FULL per-engine RMSE (win/loss pattern)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path('/home/user/rulb/tim_20261105/full_20261105')
OUT = HERE / "paper_figures"; OUT.mkdir(exist_ok=True)
def load(subset, arm):
    b = []
    for k in range(5):
        p = HERE / 'runs' / subset / str(k) / f'{arm}_101.csv'
        if not p.is_file(): return None
        b.append(pd.read_csv(p))
    return pd.concat(b, ignore_index=True)
def eng(d):
    e = d.y_pred - d.y_true
    return np.sqrt(pd.DataFrame({'u': d.unit, 'e2': e**2}).groupby('u').e2.mean())
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
for ax, subset in zip(axes, ['FD002', 'FD004']):
    m = load(subset, 'M0'); f = load(subset, 'FULL')
    if m is None or f is None: continue
    em, ef = eng(m), eng(f)
    idx = em.index.intersection(ef.index)
    lim = max(em.loc[idx].max(), ef.loc[idx].max()) * 1.05
    ax.scatter(em.loc[idx], ef.loc[idx], s=14, alpha=.7)
    ax.plot([0, lim], [0, lim], 'k--', lw=.8)
    better = (ef.loc[idx] < em.loc[idx]).mean()
    ax.set_xlabel('per-engine RMSE, M0 (cycles)'); ax.set_ylabel('per-engine RMSE, FULL (cycles)')
    ax.set_title(f'{subset}: FULL better on {100*better:.0f}% of engines', fontsize=10)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "fig_per_engine_scatter.png", dpi=200)
fig.savefig(OUT / "fig_per_engine_scatter.pdf")
print("written", OUT / "fig_per_engine_scatter.png")
