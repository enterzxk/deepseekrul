import glob, numpy as np, pandas as pd
def load(root, tag):
    fs = glob.glob(f"{root}/{tag}/FD002/*/protocol_c_predictions.csv")
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    return {u: g.set_index("abs_cut").signed_error.to_dict() for u,g in d.groupby("unit_id")}
def effect(root, a, b, B=2000, seed=1):
    ea, eb = load(root,a), load(root,b)
    if not ea or not eb: return None
    units = sorted(set(ea)&set(eb)); rng = np.random.default_rng(seed); d=[]
    for _ in range(B):
        va=[];vb=[]
        for u in units:
            c = rng.choice(list(ea[u].keys()))
            va.append(ea[u][c]); vb.append(eb[u].get(c, np.nan))
        va=np.array(va); vb=np.array(vb); m=~np.isnan(vb)
        d.append(np.sqrt((va[m]**2).mean())-np.sqrt((vb[m]**2).mean()))
    d=np.array(d)
    return float(np.median(d)), float((d<0).mean())
for a,b in (("M1_global","M0_raw_tf"), ("M6_tfA","M0_raw_tf")):
    print(f"--- {a} vs {b} ---")
    meds=[]
    for name, root in (("split7","experiments/order03"), ("split8","experiments/split8"), ("split9","experiments/split9")):
        r = effect(root,a,b)
        if r:
            print(f"  {name}: median={r[0]:+.3f} P(better)={r[1]:.3f}")
            meds.append(r[0])
    if meds:
        print(f"  mean={np.mean(meds):+.3f} sd={np.std(meds):+.3f} sign_flip={bool(min(meds)<0<max(meds))}")
