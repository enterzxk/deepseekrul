
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import glob, numpy as np, pandas as pd
def per_engine_mse(tag):
    fs = sorted(glob.glob(f"experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv"))
    dfs = [pd.read_csv(f).set_index("unit_id") for f in fs]
    units = dfs[0].index.to_numpy()
    # per engine: squared error per seed, averaged over seeds (proper)
    mse = np.stack([(d.loc[units, "signed_error"].to_numpy() ** 2) for d in dfs]).mean(0)
    return units, mse, [d.loc[units, "signed_error"].to_numpy() for d in dfs]
ua, ma, ea = per_engine_mse("M1_global")
ub, mb, eb = per_engine_mse("M6_tfA")
assert np.array_equal(ua, ub)
n = len(ua)
print(f"engines={n} seeds={len(ea)}")
print(f"engine-level RMSE: M1={np.sqrt(ma.mean()):.3f}  M6={np.sqrt(mb.mean()):.3f}  rel={(np.sqrt(mb.mean())-np.sqrt(ma.mean()))/np.sqrt(mb.mean())*100:.1f}%")
# row-pooled
ra = np.sqrt(np.mean(np.concatenate([e**2 for e in ea])))
rb = np.sqrt(np.mean(np.concatenate([e**2 for e in eb])))
print(f"row-pooled RMSE:   M1={ra:.3f}  M6={rb:.3f}  rel={(rb-ra)/rb*100:.1f}%")
rng = np.random.default_rng(20261022)
idx = rng.integers(0, n, size=(20000, n))
d = np.sqrt(ma[idx].mean(1)) - np.sqrt(mb[idx].mean(1))
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"paired bootstrap diff mean={d.mean():.3f} CI=[{lo:.3f},{hi:.3f}] excl0={bool(lo>0 or hi<0)} P(M1 better)={float((d<0).mean()):.3f}")
# per-seed engine-level rmse
for k,f in enumerate(sorted(glob.glob("experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv"))):
    d1 = pd.read_csv(f); print(f"  M1 seed{k}: engine_rmse={np.sqrt((d1.signed_error**2).mean()):.3f}")
for k,f in enumerate(sorted(glob.glob("experiments/order03/M6_tfA/FD002/*/matched_cutoff_predictions.csv"))):
    d1 = pd.read_csv(f); print(f"  M6 seed{k}: engine_rmse={np.sqrt((d1.signed_error**2).mean()):.3f}")
EOF
