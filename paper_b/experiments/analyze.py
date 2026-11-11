"""Aggregate Paper B run metrics into seed-level and subset-level tables."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='paper_b/results/raw')
    ap.add_argument('--out', default='paper_b/results/all_runs.csv')
    a = ap.parse_args()
    rows = []
    for mp in Path(a.root).glob('*/*/s*/metrics.json'):
        m = json.loads(mp.read_text())
        cfgp = mp.parent / 'config.json'
        cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
        tp = mp.parent / 'test_metrics.json'
        tm = json.loads(tp.read_text()) if tp.exists() else {}
        rows.append(dict(tag=cfg.get('tag', mp.parts[-4]), subset=cfg.get('subset', mp.parts[-3]),
                         seed=int(cfg.get('seed', mp.parts[-2][1:])),
                         span=cfg.get('span'), model=cfg.get('model'),
                         ms_mode=cfg.get('ms_mode'), sa_mode=cfg.get('sa_mode'),
                         hi_source=cfg.get('hi_source'), hi_mono=cfg.get('hi_mono'),
                         use_cycle=cfg.get('use_cycle'), input_mode=cfg.get('input_mode'),
                         val_rmse=m.get('val_engine_rmse'), val_mae=m.get('mae'),
                         val_bias=m.get('bias'), params=m.get('params'),
                         best_ep=m.get('best_ep'), elapsed_s=m.get('elapsed_s'),
                         test_rmse=tm.get('rmse'), test_nasa=tm.get('nasa_sum'),
                         test_bias=tm.get('bias')))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print('wrote', a.out, len(df), 'runs')
    if len(df):
        g = df.groupby(['tag', 'subset']).agg(
            n=('seed', 'count'), val_mean=('val_rmse', 'mean'), val_std=('val_rmse', 'std'),
            test_mean=('test_rmse', 'mean'), test_std=('test_rmse', 'std')).reset_index()
        g.to_csv(str(a.out).replace('.csv', '_summary.csv'), index=False)
        print(g.to_string(index=False))


if __name__ == '__main__':
    main()
