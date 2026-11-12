"""Retrain one A8-base cell into /tmp and run the inference-time module ablation.

The frozen artifacts on disk do not include model.pt, so the dual-span / cycle
ablation needs a checkpoint.  The config is taken verbatim from the A8 grid.
"""
import json
from pathlib import Path

from ..training.train import train_one

GRID = Path('paper_b/experiments/grids/ablation8.json')
OUT = '/tmp/audit_runs'


def main():
    runs = json.loads(GRID.read_text())
    cfg = next(r for r in runs
               if r['tag'] == 'A8_base_L70' and r['subset'] == 'FD001' and r['seed'] == 101)
    cfg = {**cfg, 'data_dir': 'data/CMAPSS'}
    print('config', json.dumps({k: cfg[k] for k in
                                ('tag', 'subset', 'seed', 'ms_mode', 'sa_mode',
                                 'hi_source', 'use_cycle', 'span')}))
    m = train_one(cfg, device='cuda:0', out_root=OUT)
    print('trained', m['val_engine_mean_rmse'], 'ep', m['best_ep'])


if __name__ == '__main__':
    main()
