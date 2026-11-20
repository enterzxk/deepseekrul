"""Per-engine win-rate of LS-ctx vs LTAG/K1 (validation)."""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from paper_b.eval.metrics_v2 import per_engine_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results')
    a = ap.parse_args()
    print('| subset | n | LS<LTAG | LS<K1 | mean Δ(LS−LTAG) | median Δ |')
    print('|---|---:|---:|---:|---:|---:|')
    for f in sorted(glob.glob(str(Path(a.dir) / 'ltagv2_p1_FD00*.preds.csv'))):
        subset = Path(f).name.split('_')[2].replace('.preds.csv', '')
        df = pd.read_csv(f)
        e_ls = per_engine_rmse(df.y, df.pred_ls, df.engine)
        e_lt = per_engine_rmse(df.y, df.pred_ltag, df.engine)
        e_k1 = per_engine_rmse(df.y, df.p_k1, df.engine)
        keys = sorted(e_ls)
        d = np.array([e_ls[k] - e_lt[k] for k in keys])
        print(f'| {subset} | {len(keys)} | {int((d < 0).sum())} | '
              f'{int(sum(e_ls[k] < e_k1[k] for k in keys))} | '
              f'{d.mean():+.3f} | {np.median(d):+.3f} |')


if __name__ == '__main__':
    main()
