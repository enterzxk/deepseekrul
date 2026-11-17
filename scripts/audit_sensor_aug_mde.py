"""MDE of the sensor-aug Stage-1 screen, computed on its own pairing structure.

The screen is 6 arms (G0 baseline, G1-G5) x 2 subsets (FD001/FD004) x 3 seeds
(101/202/303), judged by "macro gain >= +1.0%" on the clean channel.

MDE is computed on the paired-by-seed macro (mean of the two per-subset gains),
because the pre-registered bar is a macro quantity. The exact 80%-power coefficient
for the noncentral t at df = n_seed-1 = 2 is 5.6535 (scripts/mde_coef.py).

Read-only: consumes reports/table_a_clean.csv, trains nothing.
"""
import os
import re

import numpy as np
import pandas as pd

from mde_coef import mde_coef

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, 'tmp', 'audit_sensor_aug', 'table_a_clean.csv')
BAR = 1.0                      # pre-registered macro bar, percentage points
SUBSETS = ('FD001', 'FD004')


def per_seed(s):
    return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r's(\d+):([\d.]+)', s)}


t = pd.read_csv(CSV)
rows = {(r.group, r.subset): per_seed(r.per_seed) for _, r in t.iterrows()}
seeds = sorted(rows[('G0', 'FD001')])

coef, tcrit = mde_coef(len(seeds) - 1)
print('sensor-aug Stage 1 (clean channel), n_seed = %d, df = %d' % (len(seeds), len(seeds) - 1))
print('exact 80%%-power coefficient = %.4f   t_crit = %.4f   bar = %.1fpp'
      % (coef, tcrit, BAR))
print()
print('%-4s %-6s %10s %10s %10s %10s %8s %6s' % (
    'arm', 'subset', 'gain%', 'se%', 'MDE80%', '|t|', 'sig?', 'gain>MDE'))
mdes = []
for arm in ('G1', 'G2', 'G3', 'G4', 'G5'):
    for sub in SUBSETS:
        b = rows[('G0', sub)]
        a = rows[(arm, sub)]
        ks = sorted(set(b) & set(a))
        d = np.array([b[k] - a[k] for k in ks])
        base = float(np.mean([b[k] for k in ks]))
        se = d.std(ddof=1) / np.sqrt(len(ks))
        g = 100 * d.mean() / base
        se_pct = 100 * se / base
        mde = coef * se_pct
        mdes.append(mde)
        print('%-4s %-6s %10.3f %10.3f %10.3f %8.2f %6s %8s' % (
            arm, sub, g, se_pct, mde, abs(d.mean() / se),
            'YES' if abs(d.mean() / se) > tcrit else 'no',
            'YES' if g > mde else 'no'))
print()
print('per-subset MDE80 range: %.2f%% .. %.2f%%' % (min(mdes), max(mdes)))
print()
print('MACRO (mean of the two per-subset gains, paired by seed):')
print('%-4s %10s %10s %10s %8s %6s %10s' % (
    'arm', 'macro%', 'sd', 'se', '|t|', 'sig?', 'MACRO MDE80%'))
for arm in ('G1', 'G2', 'G3', 'G4', 'G5'):
    per = []
    for s in seeds:
        g = [100.0 * (rows[('G0', sub)][s] - rows[(arm, sub)][s]) / rows[('G0', sub)][s]
             for sub in SUBSETS]
        per.append(float(np.mean(g)))
    per = np.array(per)
    m = float(per.mean())
    se = float(per.std(ddof=1) / np.sqrt(len(per)))
    mde = coef * se
    print('%-4s %10.3f %10.3f %10.3f %8.2f %6s %10.3f  (bar %.1fpp -> %.1fx below)'
          % (arm, m, per.std(ddof=1), se, abs(m / se),
             'YES' if abs(m / se) > tcrit else 'no', mde, BAR, mde / BAR))
