
cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, numpy as np, pandas as pd
def boot(a_tag, b_tag, sub):
    A = pd.concat([pd.read_csv(f).assign(rep=f) for f in glob.glob(f"experiments/order03/{a_tag}/{sub}/*/matched_cutoff_predictions.csv")])
    B = pd.concat([pd.read_csv(f).assign(rep=f) for f in glob.glob(f"experiments/order03/{b_tag}/{sub}/*/matched_cutoff_predictions.csv")])
    ea = A.groupby("unit_id").signed_error.apply(lambda s: (s**2).mean())
    eb = B.groupby("unit_id").signed_error.apply(lambda s: (s**2).mean())
    idx = ea.index.intersection(eb.index); ea, eb = ea.loc[idx].to_numpy(), eb.loc[idx].to_numpy()
    rng = np.random.default_rng(20261022); n=len(idx); d=[]
    for _ in range(20000):
        t = rng.integers(0,n,n); d.append(np.sqrt(ea[t].mean())-np.sqrt(eb[t].mean()))
    d=np.array(d); lo,hi=np.percentile(d,[2.5,97.5])
    print(f"{a_tag} vs {b_tag} [{sub}] engines={n} diff={d.mean():+.3f} CI=[{lo:+.3f},{hi:+.3f}] excl0={bool(lo>0 or hi<0)} P={float((d<0).mean()):.3f}")
for sub in ("FD001","FD002"):
    boot("M1_global","M6_tfA",sub)
    boot("M1_global","M0_raw_tf",sub)
    boot("M6_tfA","M0_raw_tf",sub)
EOF
