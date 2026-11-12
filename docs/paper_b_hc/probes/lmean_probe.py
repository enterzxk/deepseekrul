"""L_mean of the frozen TRAINING split, plus the resulting b0. Read-only probe."""
import math
import sys

from paper_b.training.data import _prepare_split

for s in ['FD001', 'FD002', 'FD003', 'FD004']:
    d = _prepare_split(s, 'data/CMAPSS', 7, 0, 0.2, 6)
    tr = d['tr'].groupby('unit').size().to_numpy()
    va = d['va'].groupby('unit').size().to_numpy()
    m = float(tr.mean())
    b0 = math.log(math.expm1(1.0 / m))
    print('%s train_n=%d Lmean=%.4f max=%d p95=%.1f b0=%.6f '
          'val_n=%d valLmean=%.1f valmax=%d'
          % (s, len(tr), m, int(tr.max()), float(__import__('numpy').percentile(tr, 95)),
             b0, len(va), float(va.mean()), int(va.max())))
    sys.stdout.flush()
