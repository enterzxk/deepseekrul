"""Protocol B and pooled vs macro, for the three baselines."""
import glob
import numpy as np, pandas as pd
def load(tag, kind):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/{kind}')
    if not fs: return None
    return pd.concat([pd.read_csv(f) for f in fs])
def pooled_macro(d):
    e = d.signed_error.to_numpy(float)
    per = d.groupby('unit_id').signed_error.apply(lambda s: np.sqrt((s**2).mean()))
    return float(np.sqrt((e**2).mean())), float(per.mean())
print('=== Protocol B (life fractions) ===')
print(f"{'cell':14s} {'f=0.5':>7s} {'f=0.7':>7s} {'f=0.9':>7s}  {'pooled':>7s} {'macro':>7s}")
for tag in ('M1_global','M6_tfA','M0_raw_tf'):
    d = load(tag, 'pseudo_test_predictions.csv')
    if d is None: print(tag,'missing'); continue
    row = []
    for f in (0.5, 0.7, 0.9):
        s = d[np.isclose(d.cutoff, f)]
        row.append(np.sqrt((s.signed_error**2).mean()) if len(s) else float('nan'))
    p, m = pooled_macro(d)
    print(f'{tag:14s} {row[0]:7.2f} {row[1]:7.2f} {row[2]:7.2f}  {p:7.2f} {m:7.2f}')
print()
print('=== matched-cutoff: pooled vs macro ===')
for tag in ('M1_global','M6_tfA','M0_raw_tf'):
    d = load(tag, 'matched_cutoff_predictions.csv')
    if d is None: continue
    p, m = pooled_macro(d)
    print(f'{tag:14s} pooled={p:6.2f}  macro={m:6.2f}')
