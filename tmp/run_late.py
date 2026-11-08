"""Late-window specialist at the SAME 50-cycle window: separates specialisation
from observation length. Trains only on windows with end_cycle >= cutoff."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent; ROOT = PARENT.parent
sys.path[:0] = [str(PARENT), str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT)]
import run_fold as base
import run_ida as ida
import run_full_fold3 as r3

def filt(data, cutoff):
    x, y, e, m, c = data
    keep = m[:, 1] >= cutoff
    return (x[keep], y[keep], e[keep], m[keep], None if c is None else c[keep])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset'); ap.add_argument('fold', type=int)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--cutoff', type=int, default=100)
    ap.add_argument('--arms', default='M0')
    ap.add_argument('--epochs', type=int, default=80)
    a = ap.parse_args()
    dest = HERE / f'runs_late_c{a.cutoff}_w{a.window}' / a.subset / str(a.fold); dest.mkdir(parents=True, exist_ok=True)
    folds, df = ida.outer_folds(a.subset)
    df = df.sort_values(['unit','cycle']).reset_index(drop=True)
    train, valid = folds[a.fold]
    tables = base.prior_tables(df, a.subset, train, valid, a.fold)
    tr, va = ida.prep_fold(df, a.subset, a.fold, train, valid, window=a.window)
    tr = filt(tr, a.cutoff); va = filt(va, a.cutoff)
    xtr, ytr, etr, mtr = tr[:4]; xva, yva, eva, mva = va[:4]
    lives = df.groupby('unit').cycle.max()
    meta = pd.DataFrame(mva, columns=['unit','end_cycle'])
    meta['short'] = meta.unit.map(lives) <= np.quantile(lives.loc[sorted(train)], .25)
    meta['true_life'] = meta.unit.map(lives)
    for arm in [x for x in a.arms.split(',') if x]:
        net = r3.build_net(arm, a.window); ida.set_seed(a.seed); net = net.cuda()
        tx = [xtr, ytr, etr, base.prior_values(mtr, tables['FULL'])]
        vx = [xva, yva, eva, base.prior_values(mva, tables['FULL'])]
        target = dest / f'{arm}_{a.seed}.csv'
        wrote = r3.train_arm(arm, net, tx, vx, target, meta, a.seed, None, None, a.epochs)
        print('completed', a.subset, a.fold, arm, 'late-spec w', a.window, 'trained' if wrote else 'cached', flush=True)
    (dest / f'done_{a.seed}.json').write_text(json.dumps(dict(subset=a.subset, fold=a.fold, cutoff=a.cutoff, window=a.window)))

if __name__ == '__main__':
    main()