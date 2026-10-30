"""Effect of available history length (cutoff cycle) on error."""
import glob
import numpy as np, pandas as pd
def load(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/protocol_c_predictions.csv')
    if not fs: return None
    return pd.concat([pd.read_csv(f) for f in fs])
print('cutoff  n     RMSE   MAE   late(M1)   trueRUL_med')
for tag in ('M1_global',):
    d = load(tag)
    for c in sorted(d.abs_cut.unique()):
        s = d[np.isclose(d.abs_cut, c)]
        e = s.signed_error.to_numpy(float)
        print(f'{c:6.0f} {len(s):4d}  {np.sqrt((e**2).mean()):6.2f} {np.abs(e).mean():5.2f}  {(e>0).mean():8.3f}   {s.y_true.median():8.1f}')
print()
print('same for M6_tfA:')
d = load('M6_tfA')
for c in sorted(d.abs_cut.unique()):
    s = d[np.isclose(d.abs_cut, c)]
    e = s.signed_error.to_numpy(float)
    print(f'{c:6.0f} {np.sqrt((e**2).mean()):6.2f}')
