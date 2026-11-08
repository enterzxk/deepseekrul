import glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
C = Path('/home/user/rulb/experiments/TESTCACHE')
OUT = Path('/home/user/rulb/paper_figures_v4'); OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

# --- fig 1: predicted vs true, per engine, four fleets ---
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharex=True, sharey=True)
for ax, subset in zip(axes, ['FD001','FD002','FD003','FD004']):
    fs = sorted(C.glob(f'{subset}_s*_f*.csv'))
    ds = [pd.read_csv(f).sort_values('unit').reset_index(drop=True) for f in fs]
    y = ds[0].y_true.to_numpy()
    p = np.stack([d.y_pred.to_numpy() for d in ds]).mean(axis=0)
    ax.scatter(y, p, s=7, alpha=0.55, edgecolors='none')
    ax.plot([0,125],[0,125],'k--',lw=0.8)
    ax.set_title('%s  RMSE %.2f' % (subset, np.sqrt(np.mean((p-y)**2))), fontsize=8)
    ax.set_xlabel('true RUL'); ax.set_xlim(0,130); ax.set_ylim(0,140)
axes[0].set_ylabel('predicted RUL')
fig.tight_layout(); fig.savefig(OUT/'fig_scatter.pdf'); fig.savefig(OUT/'fig_scatter.png', dpi=200); plt.close(fig)

# --- fig 2: error vs true RUL (bias structure) ---
fig, ax = plt.subplots(figsize=(4.2, 2.9))
for subset, mk in zip(['FD001','FD002','FD003','FD004'], ['o','s','^','D']):
    fs = sorted(C.glob(f'{subset}_s*_f*.csv'))
    ds = [pd.read_csv(f).sort_values('unit').reset_index(drop=True) for f in fs]
    y = ds[0].y_true.to_numpy(); p = np.stack([d.y_pred.to_numpy() for d in ds]).mean(axis=0)
    bins = np.arange(0, 131, 25); idx = np.digitize(y, bins) - 1
    xs, ys = [], []
    for b in range(len(bins)-1):
        m = idx == b
        if m.sum() < 5: continue
        xs.append(y[m].mean()); ys.append((p[m]-y[m]).mean())
    ax.plot(xs, ys, marker=mk, ms=3, lw=1, label=subset)
ax.axhline(0, color='k', lw=0.7, ls='--')
ax.set_xlabel('true RUL (cycles)'); ax.set_ylabel('mean signed error')
ax.legend(frameon=False, fontsize=7); fig.tight_layout()
fig.savefig(OUT/'fig_bias.pdf'); fig.savefig(OUT/'fig_bias.png', dpi=200); plt.close(fig)

# --- fig 3: four-fleet bar chart vs published ---
r = pd.read_csv('/home/user/rulb/RESULTS_V4.csv')
pub = {'FD001':10.43,'FD002':12.02,'FD003':10.49,'FD004':14.11}
x = np.arange(4); w = 0.35
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.bar(x-w/2, [pub[s] for s in r.subset], w, label='AST 2026', color='0.65')
ax.bar(x+w/2, r.ens_rmse, w, label='ours (ensemble)', color='C0')
ax.set_xticks(x); ax.set_xticklabels(r.subset); ax.set_ylabel('RMSE (cycles)')
ax.legend(frameon=False, fontsize=7); ax.set_ylim(0, 16)
fig.tight_layout(); fig.savefig(OUT/'fig_compare.pdf'); fig.savefig(OUT/'fig_compare.png', dpi=200); plt.close(fig)

print('figures written:', sorted(p.name for p in OUT.glob('*.pdf')))