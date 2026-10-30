"""H2 verdict (correct paths)."""
import glob
import numpy as np, pandas as pd
def per_engine(pattern):
    fs = glob.glob(pattern)
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float) ** 2
    return d.groupby('unit_id').sq.mean()
PAT = {
  'split7': ('experiments/order03/FD002/base_s*/matched_cutoff_predictions.csv', 'experiments/order03/M1_global/FD002/A_s*/matched_cutoff_predictions.csv'),
}
for root in ('split8','split9'):
    for fname, label in (('matched_cutoff_predictions.csv','matched'), ('protocol_c_predictions.csv','protocolC')):
        a = per_engine(f'experiments/{root}/FD002/base_s*/{fname}')
        b = per_engine(f'experiments/{root}/M1_global/FD002/A_s*/{fname}')
        if a is None or b is None:
            print(f'{root} {label}: missing (base={a is not None}, m1={b is not None})'); continue
        j = pd.concat([a.rename('m7'), b.rename('m1')], axis=1).dropna()
        x, y = j.m7.to_numpy(), j.m1.to_numpy()
        rng = np.random.default_rng(20261022); d = []
        for _ in range(4000):
            t = rng.integers(0, len(x), len(x))
            d.append(np.sqrt(x[t].mean()) - np.sqrt(y[t].mean()))
        d = np.asarray(d); sig = bool(np.percentile(d,2.5)>0 or np.percentile(d,97.5)<0)
        print(f'{root} {label:9s} n={len(j)} M7={np.sqrt(x.mean()):.2f} M1={np.sqrt(y.mean()):.2f} diff={np.sqrt(x.mean())-np.sqrt(y.mean()):+.2f} CI=[{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}] sig={sig}')
