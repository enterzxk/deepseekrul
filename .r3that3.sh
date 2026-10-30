#!/bin/bash
cat > /home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_that3.py <<'PYEOF'
import numpy as np, pandas as pd
r = pd.read_csv('/home/yndx/zxk/bearing-rul-pi/results/dsh/review_r3/that_absolute.csv')
r['at_lower'] = np.isclose(r.that, 1.2*r.cutoff)
r['at_upper'] = np.isclose(r.that, 400.0)
print(f"{'cut':>4s} {'n':>4s} {'nan':>5s} {'lower':>6s} {'upper':>6s} {'free':>5s}")
for c, g in r.groupby('cutoff'):
    free = g[g.ok & ~g.at_lower & ~g.at_upper]
    print(f'{c:4d} {len(g):4d} {int(g.that.isna().sum()):5d} {int(g.at_lower.sum()):6d} {int(g.at_upper.sum()):6d} {len(free):5d}')
free = r[r.ok & ~r.at_lower & ~r.at_upper]
print(); print('free (unclipped, finite) cases:', len(free))
if len(free) > 3:
    cc = float(np.corrcoef(free.that, free.life)[0,1]); print('  corr(T_hat, T) = %.3f  R2 = %.3f' % (cc, cc*cc))
    print('  median T_hat/T = %.2f' % float(np.median(free.that/free.life)))
else:
    print('  too few unclipped cases to correlate')
print()
print('slope distribution (sign/magnitude):')
print(r.groupby('cutoff').slope.describe()[['count','mean','std','min','50%','max']].round(5).to_string())
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && /home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_that3.py
