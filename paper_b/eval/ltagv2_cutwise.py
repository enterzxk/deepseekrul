"""Cut-wise validation RMSE: does the LS/K1-Calib advantage hold near end-of-life?"""
import glob
from pathlib import Path

import pandas as pd

from paper_b.eval.metrics_v2 import engine_mean_rmse

for f in sorted(glob.glob('results/ltagv2_p1_FD00*.preds.csv')):
    subset = Path(f).name.split('_')[2].replace('.preds.csv', '')
    df = pd.read_csv(f)
    df['rank'] = df.groupby('engine')['cut'].rank(method='dense').astype(int) - 1
    fracs = [0.4, 0.55, 0.7, 0.85, 0.95]
    print(f'\n## {subset}')
    print('| cut_frac | K1 | LTAG | K1-Calib | LS-ctx | LS−LTAG |')
    print('|---:|---:|---:|---:|---:|---:|')
    for rank in sorted(df['rank'].unique()):
        sub = df[df['rank'] == rank]
        vals = {c: engine_mean_rmse(sub.y, sub[c], sub.engine)
                for c in ['p_k1', 'pred_ltag', 'pred_calib', 'pred_ls']}
        lab = fracs[rank] if rank < len(fracs) else rank
        print(f"| {lab} | {vals['p_k1']:.3f} | {vals['pred_ltag']:.3f} | "
              f"{vals['pred_calib']:.3f} | {vals['pred_ls']:.3f} | "
              f"{vals['pred_ls'] - vals['pred_ltag']:+.3f} |")
