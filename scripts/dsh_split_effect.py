import glob, numpy as np, pandas as pd
def load(root, tag):
    fs = glob.glob(f"{root}/{tag}/FD002/*/protocol_c_predictions.csv")
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    return {u: g.set_index("abs_cut").signed_error.to_dict() for u,g in d.groupby("unit_id")}
def effect(root, a="M1_global", b="M6_tfA", B=2000, seed=1):
    ea, eb = load(root,a), load(root,b)
    if not ea or not eb: return None
    units = sorted(set(ea)&set(eb))
    rng = np.random.default_rng(seed); d=[]
    for _ in range(B):
        va=[];vb=[]
        for u in units:
            c = rng.choice(list(ea[u].keys()))
            va.append(ea[u][c]); vb.append(eb[u].get(c, np.nan))
        va=np.array(va); vb=np.array(vb); m=~np.isnan(vb)
        d.append(np.sqrt((va[m]**2).mean())-np.sqrt((vb[m]**2).mean()))
    d=np.array(d)
    return dict(engines=len(units), median=float(np.median(d)), p_better=float((d<0).mean()),
                lo=float(np.percentile(d,2.5)), hi=float(np.percentile(d,97.5)))
rows=[]
for name, root in (("split7","experiments/order03"), ("split8","experiments/split8"), ("split9","experiments/split9")):
    r = effect(root)
    if r:
        rows.append({"split":name, **r})
        print(f"{name}: P(M1 better)={r['p_better']:.3f} median={r['median']:+.3f} CI=[{r['lo']:+.3f},{r['hi']:+.3f}] engines={r['engines']}")
df = pd.DataFrame(rows)
if len(df):
    print()
    print(f"effect across splits: mean={df['median'].mean():+.3f} sd={df['median'].std():+.3f} range=[{df['median'].min():+.3f},{df['median'].max():+.3f}]")
    print(f"P(better) range: [{df['p_better'].min():.3f},{df['p_better'].max():.3f}]  sign flips: {bool((df['median']>0).any() and (df['median']<0).any())}")
    df.to_csv("results/dsh/effect_across_splits.csv", index=False)
