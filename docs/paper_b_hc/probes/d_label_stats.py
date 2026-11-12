"""Stage D label statistics: how much of the data sits on the RUL cap.

D4's censored loss and D3's prediction horizon K both depend on two facts that
must be measured, not assumed:

  * what fraction of train/val windows have r_t >= rul_cap (right-censored), and
  * how long the longest training engine is (K's cost driver).

No model, no torch: this only builds the frozen data layer.

Usage: python docs/paper_b_hc/probes/d_label_stats.py
"""
import json
from pathlib import Path

import numpy as np

from paper_b.hc import data_hc
from paper_b.hc import config_hc as C

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
DATA = str(ROOT / 'data/CMAPSS')


def main():
    cap = C.RUL_CAP
    out = []
    print('%-6s %8s %8s %7s %8s %8s %7s %6s %6s'
          % ('subset', 'n_train', 'cap_tr', 'frac_tr', 'n_val', 'cap_val',
             'frac_val', 'maxlen', 'p95'))
    for s in C.SUBSETS:
        ds = data_hc.build(s, DATA, C.SPAN, C.VAL_CUTS, cycle_source='raw')
        tr, va = ds['train'], ds['val']
        ytr = np.asarray(tr['y'], np.float32)
        yva = np.asarray(va['y'], np.float32)
        ctr = np.asarray(tr['cycle_raw'], np.float32)
        # engine length = max cycle per engine, from the raw-cycle axis
        units = ds['tr'].groupby('unit', sort=True).size().to_numpy()
        lens = units.astype(np.float64)
        rec = dict(subset=s, n_train=int(len(ytr)),
                   cap_train=int((ytr >= cap - 1e-6).sum()),
                   frac_train=float((ytr >= cap - 1e-6).mean()),
                   n_val=int(len(yva)),
                   cap_val=int((yva >= cap - 1e-6).sum()),
                   frac_val=float((yva >= cap - 1e-6).mean()),
                   max_len=float(lens.max()),
                   p95_len=float(np.percentile(lens, 95)),
                   mean_len=float(lens.mean()),
                   max_cycle_raw=float(ctr.max()))
        out.append(rec)
        print('%-6s %8d %8d %7.3f %8d %8d %7.3f %6.0f %6.0f'
              % (s, rec['n_train'], rec['cap_train'], rec['frac_train'],
                 rec['n_val'], rec['cap_val'], rec['frac_val'],
                 rec['max_len'], rec['p95_len']))
    p = ROOT / 'experiments/hc/tmp'
    p.mkdir(parents=True, exist_ok=True)
    (p / 'd_label_stats.json').write_text(json.dumps(out, indent=1))
    print('\nwrote %s' % (p / 'd_label_stats.json'))


if __name__ == '__main__':
    main()
