"""Bootstrap at cut 0.95: LTAG vs K1-Calib / LS-ctx."""
import glob
from pathlib import Path

import pandas as pd

from paper_b.eval.metrics_v2 import paired_engine_bootstrap

for f in sorted(glob.glob('results/ltagv2_p1_FD00*.preds.csv')):
    subset = Path(f).name.split('_')[2].replace('.preds.csv', '')
    df = pd.read_csv(f)
    df['rank'] = df.groupby('engine')['cut'].rank(method='dense').astype(int) - 1
    sub = df[df['rank'] == 4]
    print(f'\n## {subset} cut=0.95 n={len(sub)}')
    for name, col in [('K1-Calib', 'pred_calib'), ('LS-ctx', 'pred_ls')]:
        b = paired_engine_bootstrap(sub[col], sub['pred_ltag'], sub['y'], sub['engine'])
        print(f'{name} vs LTAG: Δ={b["delta"]:+.3f} CI=[{b["ci_lo"]:+.3f},{b["ci_hi"]:+.3f}] '
              f'p={b["p_one_sided"]:.3f}')
