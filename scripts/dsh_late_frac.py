"""Late-fraction comparison: a binary, heavy-tail-free metric."""
import glob
import numpy as np, pandas as pd
ROOTS = {'split7':'order03','split8':'split8','split9':'split9'}
def late_per_engine(root, tag):
    fs = glob.glob(f'experiments/{root}/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    g = d.groupby('unit_id').agg(y=('y_true','mean'), p=('y_pred','mean'))
    return (g.p > g.y).astype(float)
for a_tag, b_tag in (('M1_global','M6_tfA'), ('M1_global','M0_raw_tf')):
    print(f'=== {a_tag} vs {b_tag}: late fraction, paired per engine')
    for name, root in ROOTS.items():
        a, b = late_per_engine(root, a_tag), late_per_engine(root, b_tag)
        if a is None or b is None:
            print(f'  {name}: missing'); continue
        j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
        diff = j.b - j.a   # positive: b later than a
        n = len(j); md = diff.mean()
        rng = np.random.default_rng(20261022)
        bs = [diff.to_numpy()[rng.integers(0, n, n)].mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        sig = bool(lo > 0 or hi < 0)
        print(f'  {name}: n={n} late_a={j.a.mean():.3f} late_b={j.b.mean():.3f} diff={md:+.3f} CI=[{lo:+.3f},{hi:+.3f}] sig={sig}')
