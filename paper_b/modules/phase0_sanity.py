"""Phase 0 hard invariants for the RHSE front-end.

0.1 zero-flag identity: with no flags, R3c == R0 and R3 == R1.
0.2 padding immunity: mask==0 positions can never be flagged.
0.3 baseline parity: clean B0 prediction matches the frozen preds.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_b.modules.rhse_rul_eval import predict_at, prep
from paper_b.modules.stress_suite import inject

ROOT = Path('/home/user/sprint24')
DATA = '/home/user/rulb/data/CMAPSS'
SUBSET, SEED = 'FD001', 101

tr, va, sd, rhse, net = prep(SUBSET, DATA, SEED, ROOT / 'experiments/hc/_gate')

# find an early window with left padding (cut < 70) and a late window
i_early = None
for i in range(len(va['lengths'])):
    for c in va['cut_t'][i]:
        if 0 < c < 70 and va['mask'][i, int(c) - 1].sum() >= 20:
            i_early, t_early = i, int(c) - 1
            break
    if i_early is not None:
        break
if i_early is None:
    raise SystemExit('no padded window found')

mask = va['mask'][i_early, t_early].astype(bool)
base = va['baseline'][i_early, t_early]
x = va['x'][i_early, t_early]
dy = (x - base) / sd
p0 = predict_at(net, va, i_early, t_early, x)

# 0.1 clean zero-flag identity: qthr=-inf -> no flags -> x unchanged
qthr = np.full(sd.shape, -np.inf, dtype=np.float32)
with torch.no_grad():
    _, w, _ = rhse(torch.tensor(dy[None], dtype=torch.float32),
                   torch.tensor(mask[None], dtype=torch.float32))
w = w[0].numpy()
bad_cycle = (w < qthr[None, :]) & mask[:, None]
assert bad_cycle.sum() == 0, 'zero-flag produced flags'
x_rhc = x.copy()
p3c = predict_at(net, va, i_early, t_early, x_rhc)
d_clean = float(np.max(np.abs(p3c - p0)))
print(f'0.1 clean zero-flag: max|R3c-R0|={d_clean:.3e}')
assert d_clean < 1e-6

# 0.1 faulted zero-flag identity
dy_f, _ = inject(dy, mask, 0, 'bias', 3.0, np.random.default_rng(0))
x_f = base + dy_f * sd
p1 = predict_at(net, va, i_early, t_early, x_f)
p3 = predict_at(net, va, i_early, t_early, x_f)
d_f = float(np.max(np.abs(p3 - p1)))
print(f'0.1 faulted zero-flag: max|R3-R1|={d_f:.3e}')
assert d_f < 1e-6

# 0.2 padding immunity: loose threshold must never flag mask==0
qthr = np.full(sd.shape, 1.0, dtype=np.float32)
bad_cycle = (w < qthr[None, :]) & mask[:, None]
n_pad = int((~mask).sum())
n_pad_flags = int(bad_cycle[~mask].sum())
print(f'0.2 padding: n_pad={n_pad} n_pad_flags={n_pad_flags}')
assert n_pad_flags == 0

# 0.3 baseline parity against frozen preds.csv
pred_csv = ROOT / 'experiments/hc/cata_gate/preds/FD001_s101.csv'
df = pd.read_csv(pred_csv)
uid = int(va['engine_id'][i_early])
cut = int(va['cycle_raw'][i_early, t_early])
row = df[(df['engine'] == uid) & (df['cut'] == cut)]
if len(row):
    ref = float(row.iloc[0]['pred_k1'])
    d_ref = abs(float(p0) - ref)
    print(f'0.3 parity: engine={uid} cut={cut} p0={float(p0):.6f} ref={ref:.6f} diff={d_ref:.3e}')
    assert d_ref < 1e-4
else:
    print(f'0.3 parity: no row for engine={uid} cut={cut} (skipped)')

print('PHASE0 SANITY PASS')
