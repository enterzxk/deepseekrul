"""Protocol B (life fractions) and pooled vs macro, from existing predictions."""
import glob
import numpy as np, pandas as pd
def load(tag, kind):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/{kind}')
    if not fs: return None
    return pd.concat([pd.read_csv(f) for f in fs])
print('file kinds present:', sorted({f.split('/')[-1] for f in glob.glob('experiments/order03/M1_global/FD002/*/*.csv')}))
for kind in ('pseudo_test_predictions.csv',):
    d = load('M1_global', kind)
    if d is None: print(kind, 'missing'); continue
    print(kind, 'cols:', list(d.columns))
    print(d.head(3).to_string())
