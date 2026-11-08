"""Span control: build 100-cycle windows, then subsample every other position to 50
positions. Same tensor size / parameter count as the 50-cycle baseline, twice the
observation span. Isolates the effect of span from the effect of positions."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent; ROOT = PARENT.parent
sys.path[:0] = [str(PARENT), str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT)]
import run_fold as base
import run_ida as ida
import run_full_fold3 as r3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset'); ap.add_argument('fold', type=int)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--build-window', type=int, default=100)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--arms', default='M0,FULL')
    ap.add_argument('--epochs', type=int, default=80)
    a = ap.parse_args()
    pos = a.build_window // a.stride
    dest = HERE / f'runs_span{a.build_window}_s{a.stride}' / a.subset / str(a.fold)
    dest.mkdir(parents=True, exist_ok=True)
    folds, df = ida.outer_folds(a.subset)
    df = df.sort_values(['unit','cycle']).reset_index(drop=True)
    train, valid = folds[a.fold]
    tables = base.prior_tables(df, a.subset, train, valid, a.fold)
    tr, va = ida.prep_fold(df, a.subset, a.fold, train, valid, window=a.build_window)
    xtr, ytr, etr, mtr = tr[0][:, ::a.stride, :], tr[1], tr[2], tr[3]
    ctr = None if tr[4] is None else tr[4][:, ::a.stride, :]
    xva, yva, eva, mva = va[0][:, ::a.stride, :], va[1], va[2], va[3]
    cva = None if va[4] is None else va[4][:, ::a.stride, :]
    print('positions', xtr.shape, 'span', a.build_window, flush=True)
    lives = df.groupby('unit').cycle.max()
    meta = pd.DataFrame(mva, columns=['unit','end_cycle'])
    meta['short'] = meta.unit.map(lives) <= np.quantile(lives.loc[sorted(train)], .25)
    meta['true_life'] = meta.unit.map(lives)
    for arm in [x for x in a.arms.split(',') if x]:
        net = r3.build_net(arm, pos); ida.set_seed(a.seed); net = net.cuda()
        tx = [xtr, ytr, etr, base.prior_values(mtr, tables['FULL'])]
        vx = [xva, yva, eva, base.prior_values(mva, tables['FULL'])]
        target = dest / f'{arm}_{a.seed}.csv'
        wrote = r3.train_arm(arm, net, tx, vx, target, meta, a.seed, None, None, a.epochs)
        print('completed', a.subset, a.fold, arm, 'span', a.build_window, 'pos', pos, 'trained' if wrote else 'cached', flush=True)
    (dest / f'done_{a.seed}.json').write_text(json.dumps(dict(subset=a.subset, fold=a.fold, build_window=a.build_window, stride=a.stride, pos=pos)))

if __name__ == '__main__':
    main()