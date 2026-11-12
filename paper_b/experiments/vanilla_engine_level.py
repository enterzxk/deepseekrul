"""Recompute engine-level validation RMSE for the Vanilla arms from val_pred.csv.

Uniform with a8_engine_level.csv (per-engine RMSE first, then unweighted mean over
engines).  Reads only saved predictions; never retrains.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def engine_level(csv_path):
    d = pd.read_csv(csv_path)
    per = []
    for _, g in d.groupby('unit'):
        e = g.y_pred.to_numpy() - g.y_true.to_numpy()
        per.append(float(np.sqrt(np.mean(e ** 2))))
    per = np.asarray(per)
    return float(per.mean()), float(per.std(ddof=1)), int(len(per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='paper_b/results/raw')
    ap.add_argument('--tags', default='VAN_raw_L70,VAN_dsp_cycle_L70,VAN_raw_cycle_L70,VAN_dsp_L70')
    ap.add_argument('--out', default='paper_b/results/vanilla_engine_level.csv')
    a = ap.parse_args()
    rows = []
    for tag in a.tags.split(','):
        for p in sorted(Path(a.root).glob(f'{tag}/*/s*/val_pred.csv')):
            subset = p.parts[-3]
            seed = int(p.parts[-2][1:])
            mean, sd, n = engine_level(p)
            m = json.loads((p.parent / 'metrics.json').read_text())
            rows.append(dict(tag=tag, subset=subset, seed=seed, n_engines=n,
                             val_eng=mean, val_eng_sd=sd,
                             val_pool=m.get('val_rmse'),
                             params=m.get('params'), best_ep=m.get('best_ep')))
    df = pd.DataFrame(rows).sort_values(['tag', 'subset', 'seed'])
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))
    print('wrote', a.out, len(df), 'rows')


if __name__ == '__main__':
    main()
