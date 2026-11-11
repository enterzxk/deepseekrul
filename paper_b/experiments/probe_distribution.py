"""Compare official-test RUL distribution with the validation-cut distribution."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_b.training.data import load_train, load_test, split_units

CUTS = (0.4, 0.55, 0.7, 0.85, 0.95)
for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
    tr = load_train(Path('data/CMAPSS'), subset)
    _, rul = load_test(Path('data/CMAPSS'), subset)
    test_rul = np.minimum(np.array(list(rul.values())), 125)
    units = np.sort(tr.unit.unique())
    _, val = split_units(units, 7, 0, 0.2)
    vr = []
    for u in val:
        n = int((tr.unit == u).sum())
        for f in CUTS:
            c = int(min(n, max(1, round(f * n))))
            vr.append(min(max(n - c, 0), 125))
    vr = np.array(vr)
    print(f'{subset}: test n={len(test_rul)} mean={test_rul.mean():.1f} q25/50/75='
          f'{np.percentile(test_rul,25):.0f}/{np.percentile(test_rul,50):.0f}/{np.percentile(test_rul,75):.0f} | '
          f'val n={len(vr)} mean={vr.mean():.1f} q25/50/75='
          f'{np.percentile(vr,25):.0f}/{np.percentile(vr,50):.0f}/{np.percentile(vr,75):.0f}')
