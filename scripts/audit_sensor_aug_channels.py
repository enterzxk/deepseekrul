"""MDE of the sensor-aug Stage-1 MISSING and NOISE channels, matching the report's estimand.

The report's noise-channel number (G2 +8.73%, G5 +7.02%) is NOT a mean of per-cell
percentage gains. It is:

    per (arm, subset): average RMSE over seeds AND over the three levels
                       (beta = 0.25/0.5/1, or k = 1/2/3 for missing)
    per subset:        gain = (G0_avg - arm_avg) / G0_avg
    macro:             mean of the two subset gains

Verified: this reproduces the report exactly (G1 +5.326, G2 +8.734, G5 +7.018) and
the published `noise123` aggregates (G0 FD001 15.180, FD004 16.984).

For a paired-by-seed test we take the per-seed analogue: within a seed, average RMSE
over the three levels inside each subset, take the subset gain, then average the two
subsets. The macro is then tested over the three seeds with the exact noncentral-t
80% coefficient.

Bars: missing k=1/2/3 macro >= 10pp; noise beta=.25/.5/1 macro >= 10pp.

Read-only: consumes stage1_cells.csv.
"""
import os

import numpy as np
import pandas as pd

from mde_coef import mde_coef

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, 'tmp', 'audit_sensor_aug', 'stage1_cells.csv')
SUBSETS = ('FD001', 'FD004')
LEVELS = {
    'MISSING': ('miss1_rmse', 'miss2_rmse', 'miss3_rmse'),
    'NOISE': ('noise0.25_rmse', 'noise0.5_rmse', 'noise1_rmse'),
}
BAR = 10.0

d = pd.read_csv(CSV)
seeds = sorted(d[d.group == 'G0'].seed.unique())
coef, tcrit = mde_coef(len(seeds) - 1)


def avg_rmse(arm, sub, seed, cols):
    r = d[(d.group == arm) & (d.subset == sub) & (d.seed == seed)]
    if r.empty:
        return None
    return float(r[list(cols)].to_numpy().mean())


for label, cols in LEVELS.items():
    print('=' * 82)
    print('%s channel (levels averaged inside each subset, then subset gain, then macro)'
          % label)
    print('bar = %.1fpp   n_seed = %d   df = %d   exact coef = %.4f   t_crit = %.4f'
          % (BAR, len(seeds), len(seeds) - 1, coef, tcrit))
    print('=' * 82)
    print('%-4s %10s %10s %10s %8s %6s %12s %10s %10s' % (
        'arm', 'macro%', 'sd', 'se', '|t|', 'sig?', 'MACRO MDE80%', 'MDE/BAR', 'macro/BAR'))
    for arm in ('G1', 'G2', 'G3', 'G4', 'G5'):
        per = []
        for s in seeds:
            g = []
            for sub in SUBSETS:
                b = avg_rmse('G0', sub, s, cols)
                a = avg_rmse(arm, sub, s, cols)
                if b is None or a is None:
                    continue
                g.append(100.0 * (b - a) / b)
            per.append(float(np.mean(g)))
        per = np.array(per)
        m = float(per.mean())
        se = float(per.std(ddof=1) / np.sqrt(len(per)))
        mde = coef * se
        print('%-4s %10.3f %10.3f %10.3f %8.2f %6s %12.3f %10.2f %10.2f' % (
            arm, m, per.std(ddof=1), se, abs(m / se),
            'YES' if abs(m / se) > tcrit else 'no', mde, mde / BAR, m / BAR))
    print()

print('Reading: "sig?" asks whether the arm differs from G0 under that corruption.')
print('It says nothing about reaching the 10pp practical bar -- a significant +5%')
print('is still half the bar. The macro/BAR column is the one the gate uses.')

# ---------------------------------------------------------------------------
# Per-subset view of the noise channel: is FD004's small gain a power problem or
# a real null? The macro MDE cannot answer that, so compute it per subset.
# ---------------------------------------------------------------------------
cols = LEVELS['NOISE']
print()
print('=' * 82)
print('NOISE channel, PER SUBSET (is the small FD004 gain undetectable, or absent?)')
print('=' * 82)
print('%-4s %-6s %10s %10s %8s %6s %10s %12s' % (
    'arm', 'subset', 'gain%', 'se%', '|t|', 'sig?', 'MDE80%', 'gain/MDE'))
for arm in ('G1', 'G2', 'G5'):
    for sub in SUBSETS:
        per = []
        for s in seeds:
            b = avg_rmse('G0', sub, s, cols)
            a = avg_rmse(arm, sub, s, cols)
            if b is None or a is None:
                continue
            per.append(100.0 * (b - a) / b)
        per = np.array(per)
        m = float(per.mean())
        se = float(per.std(ddof=1) / np.sqrt(len(per)))
        mde = coef * se
        print('%-4s %-6s %10.3f %10.3f %8.2f %6s %10.3f %12.2f' % (
            arm, sub, m, se, abs(m / se),
            'YES' if abs(m / se) > tcrit else 'no', mde, m / mde))

