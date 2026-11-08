from pathlib import Path
p=Path('make_extra_figures.py'); s=p.read_text()
old = s[s.index('# 3. measurement conditions'):]
new = '''# 3. measurement conditions
mc = HERE / "MEASUREMENT_CONDITIONS.csv"
if mc.is_file():
    d = pd.read_csv(mc)
    conds = [c for c in ["clean", "mc1", "mc2"] if c in set(d.condition)]
    arms = sorted(d.arm.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    w = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        vals = [float(d[(d.arm == arm) & (d.condition == c)].rmse.mean()) for c in conds]
        axes[0].bar(np.arange(len(conds)) + i * w - 0.4 + w / 2, vals, width=w, label=arm)
        axes[1].bar(np.arange(len(conds)) + i * w - 0.4 + w / 2,
                    [v / vals[0] for v in vals], width=w, label=arm)
    axes[0].set_xticks(range(len(conds))); axes[0].set_xticklabels(conds)
    axes[0].set_ylabel("pooled RMSE (cycles)"); axes[0].set_yscale("log")
    axes[1].set_xticks(range(len(conds))); axes[1].set_xticklabels(conds)
    axes[1].set_ylabel("RMSE relative to clean"); axes[1].axhline(1.0, color="k", lw=.8)
    for ax in axes:
        ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_measurement_conditions.png", dpi=200)
    fig.savefig(OUT / "fig_measurement_conditions.pdf"); plt.close(fig)
    print("measurement figure written")
else:
    print("no MEASUREMENT_CONDITIONS.csv yet")
print("extra figures written to", OUT)
'''
s = s.replace(old, new)
p.write_text(s); print('patched')
