"""Per-seed RMSE spread within a cell vs the cross-cell gap."""
import glob
import numpy as np, pandas as pd
for tag in ('M1_global','M4_racdr','M6_tfA'):
    vals = {}
    for f in glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv'):
        seed = f.split('/')[-2].split('_s')[-1]
        d = pd.read_csv(f)
        vals[seed] = float(np.sqrt((d.signed_error.astype(float)**2).mean()))
    if not vals: continue
    v = np.array(list(vals.values()))
    print(f'{tag:12s} seeds=' + ' '.join(f'{k}:{vv:.2f}' for k, vv in sorted(vals.items())) +
          f'  | mean={v.mean():.2f} sd={v.std(ddof=1):.2f}')
