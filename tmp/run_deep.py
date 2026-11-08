"""Runner for the deep body (multi-scale conv + extra layer + attention pooling)."""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent; ROOT = PARENT.parent
sys.path[:0] = [str(HERE), str(PARENT), str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT)]
import run_fold as base
import run_ida as ida
import run_full_fold3 as r3
from deep_body import DeepBody

def build_deep(arm, window, d_model, layers, extra_layers, dropout=0.15):
    body = ida.CandidateModel(num_sensors=14, window_size=window, d_model=d_model, num_layers=layers,
                              dropout=dropout, use_baseline=True, use_ema_features=True,
                              use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)
    deep = DeepBody(body, d_model, dropout=dropout, heads=4, extra_layers=extra_layers)
    if arm in ('M0',):
        return ida.PriorWrapper(deep, 'M0', d_model=d_model)
    if arm in ('FULL','OPS'):
        return ida.PriorWrapper(deep, 'M1', d_model=d_model)
    raise ValueError(arm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset'); ap.add_argument('fold', type=int)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--arms', default='M0,FULL')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--extra-layers', type=int, default=1)
    ap.add_argument('--tag', default='deep')
    a = ap.parse_args()
    dest = HERE / f'runs_deep_w{a.window}_{a.tag}' / a.subset / str(a.fold)
    dest.mkdir(parents=True, exist_ok=True)
    folds, df = ida.outer_folds(a.subset)
    df = df.sort_values(['unit','cycle']).reset_index(drop=True)
    train, valid = folds[a.fold]
    tables = base.prior_tables(df, a.subset, train, valid, a.fold)
    tr, va = ida.prep_fold(df, a.subset, a.fold, train, valid, window=a.window)
    xtr, ytr, etr, mtr = tr[:4]; xva, yva, eva, mva = va[:4]
    lives = df.groupby('unit').cycle.max()
    meta = pd.DataFrame(mva, columns=['unit','end_cycle'])
    meta['short'] = meta.unit.map(lives) <= np.quantile(lives.loc[sorted(train)], .25)
    meta['true_life'] = meta.unit.map(lives)
    for arm in [x for x in a.arms.split(',') if x]:
        net = build_deep(arm, a.window, a.d_model, a.layers, a.extra_layers)
        print(arm, 'params', sum(p.numel() for p in net.parameters()), flush=True)
        ida.set_seed(a.seed); net = net.cuda()
        tx = [xtr, ytr, etr, base.prior_values(mtr, tables['FULL'])]
        vx = [xva, yva, eva, base.prior_values(mva, tables['FULL'])]
        target = dest / f'{arm}_{a.seed}.csv'
        wrote = r3.train_arm(arm, net, tx, vx, target, meta, a.seed, None, None, a.epochs)
        print('completed', a.subset, a.fold, arm, 'deep', 'trained' if wrote else 'cached', flush=True)
    (dest / f'done_{a.seed}.json').write_text(json.dumps(dict(subset=a.subset, fold=a.fold, seed=a.seed, window=a.window, d_model=a.d_model, layers=a.layers, extra_layers=a.extra_layers)))

if __name__ == '__main__':
    main()