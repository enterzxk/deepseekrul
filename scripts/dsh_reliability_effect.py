"""What does use_reliability actually do? M4_racdr vs M1_global."""
import glob, hashlib
import numpy as np, pandas as pd
def per_engine(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float)**2
    return d.groupby('unit_id').sq.mean(), d
for tag in ('M1_global','M4_racdr','M4c_noid','H2_fixed1'):
    r = per_engine(tag)
    if r is None: print(tag,'missing'); continue
    pe, d = r
    h = hashlib.md5(pd.util.hash_pandas_object(d.y_pred).values.tobytes()).hexdigest()[:8]
    print(f'{tag:12s} RMSE={np.sqrt(pe.mean()):6.2f}  pred_md5={h}')
print()
a = pd.concat([pd.read_csv(f) for f in glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')])
b = pd.concat([pd.read_csv(f) for f in glob.glob('experiments/order03/M4_racdr/FD002/*/matched_cutoff_predictions.csv')])
j = a.merge(b, on='unit_id', suffixes=('_m1','_m4'))
print(f'M1 vs M4_racdr: corr={j.y_pred_m1.corr(j.y_pred_m4):.4f} | mean|diff|={np.abs(j.y_pred_m1-j.y_pred_m4).mean():.2f}')
