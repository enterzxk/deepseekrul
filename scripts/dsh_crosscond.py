"""Cross-condition consistency: FD001 (single condition) vs FD002 (six)."""
import glob
import numpy as np, pandas as pd
def per_engine(sub, tag, kind='matched_cutoff_predictions.csv'):
    fs = glob.glob(f'experiments/order03/{tag}/{sub}/*/{kind}')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float)**2
    return d.groupby('unit_id').sq.mean()
print(f"{'comparison':22s} {'subset':6s} {'n':>3s} {'A':>6s} {'B':>6s} {'diff':>7s} {'dir':>4s}")
for a_tag, b_tag in (('M1_global','M6_tfA'), ('M6_tfA','M0_raw_tf'), ('M1_global','M0_raw_tf')):
    for sub in ('FD001','FD002'):
        a, b = per_engine(sub, a_tag), per_engine(sub, b_tag)
        if a is None or b is None:
            print(f'{a_tag} vs {b_tag:10s} {sub}  missing'); continue
        j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
        ra, rb = np.sqrt(j.a.mean()), np.sqrt(j.b.mean())
        d = ra - rb
        print(f'{a_tag+chr(32)+chr(118)+chr(115)+chr(32)+b_tag:22s} {sub:6s} {len(j):3d} {ra:6.2f} {rb:6.2f} {d:+7.2f} {"A<B" if d<0 else "A>B":>4s}')
