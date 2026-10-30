
cd /home/yndx/zxk/bearing-rul-pi
pids=$(pgrep -f "[s]up.sh" | sort -n); first=$(echo "$pids" | head -1)
for p in $pids; do [ "$p" != "$first" ] && kill -9 $p 2>/dev/null; done
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
def load(tag):
    fs = sorted(glob.glob(f"experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv"))
    dfs = [pd.read_csv(f).set_index("unit_id") for f in fs]
    units = dfs[0].index.to_numpy()
    err = np.stack([d.loc[units, "signed_error"].to_numpy() for d in dfs]).mean(0)
    return units, err
ua, ea = load("M1_global")
ub, eb = load("M6_tfA")
assert np.array_equal(ua, ub)
n = len(ua)
rng = np.random.default_rng(20261022)
sa, sb = ea**2, eb**2
idx = rng.integers(0, n, size=(20000, n))
ra = np.sqrt(sa[idx].mean(1)); rb = np.sqrt(sb[idx].mean(1))
d = ra - rb
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"engines={n}")
print(f"M1 rmse={np.sqrt(sa.mean()):.3f}  M6 rmse={np.sqrt(sb.mean()):.3f}  rel_improve={(np.sqrt(sb.mean())-np.sqrt(sa.mean()))/np.sqrt(sb.mean())*100:.1f}%")
print(f"paired engine bootstrap: mean_diff={d.mean():.3f} CI=[{lo:.3f},{hi:.3f}] excludes_zero={bool(lo>0 or hi<0)}")
print(f"P(M1 better)={float((d<0).mean()):.3f}")
EOF
