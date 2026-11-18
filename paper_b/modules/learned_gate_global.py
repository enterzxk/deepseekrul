"""Global learned gate trained across all four subsets.

Why a global gate: the per-subset gates are trained on one regime family each.
On FD001 the per-subset gate is conservative and loses to the fixed median; a
global gate sees multi-regime and single-regime training engines together and
can learn the regime-conditional policy from the measured `regime_dist`, not
from the subset label. The subset id is included as a feature only for
diagnosis; the headline comparison also reports a no-subset-id variant.

Training is on TRAINING engines of all subsets (GroupKFold by subset+engine);
validation engines are never used for fitting or early stopping.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from paper_b.modules.learned_gate import (FEATS, Gate, canon, engine_mean_rmse,
                                          train_gate)

SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
# ``cycle_frac`` and ``n`` are NOT available at test time: the test engine's
# failure time is unknown. The gate must use only observable features. The
# current cycle ``cut`` and the observed window length are observable, so they
# replace the lifetime-normalised features.
FEATS_G = [f for f in FEATS if f != 'cycle_frac']


def load_glob(pattern):
    frames = []
    for p in sorted(glob.glob(pattern)):
        name = Path(p).name
        subset = name.split('_')[0]
        seed = int(name.split('_s')[1].split('.')[0])
        df = canon(pd.read_csv(p))
        df['subset'] = subset
        df['seed'] = seed
        frames.append(df)
    if not frames:
        raise SystemExit('no files matched %s' % pattern)
    return pd.concat(frames, ignore_index=True)


def design(df, use_subset_id=True):
    X = df[FEATS_G].to_numpy(float)
    extra = []
    if use_subset_id:
        for s in SUBSETS:
            extra.append((df['subset'] == s).to_numpy(float))
    cycle = df['cut'].to_numpy(float)
    extra.append(cycle / 250.0)
    extra.append(np.minimum(cycle, 70.0) / 70.0)
    extra.append((df['regime_dist'].to_numpy(float) > 0.1).astype(float))
    return np.column_stack([X] + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-glob', default=str(Path.home() / 'sprint24/experiments/hc/cata_gate/train/*.csv'))
    ap.add_argument('--val-glob', default=str(Path.home() / 'sprint24/experiments/hc/cata_gate/preds/*.csv'))
    ap.add_argument('--out', default=str(Path.home() / 'sprint24/experiments/hc/cata_gate/global.json'))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--no-subset-id', action='store_true')
    ap.add_argument('--random-gate', action='store_true',
                    help='placebo: replace the learned gate with a Bernoulli draw '
                         'at the same mean rate')
    ap.add_argument('--random-seed', type=int, default=12345)
    ap.add_argument('--save-models', default=None,
                    help='torch.save the fold models + feature scaler for a frozen test')
    a = ap.parse_args()

    tr = load_glob(a.train_glob)
    va = load_glob(a.val_glob)
    use_id = not a.no_subset_id
    Xtr = design(tr, use_id)
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    p_med_tr = tr['p_med'].to_numpy(float)
    p_k1_tr = tr['p_k1'].to_numpy(float)
    y_tr = tr['y_cap'].to_numpy(float)
    groups = (tr['subset'] + '_' + tr['engine'].astype(str)).to_numpy()

    gkf = GroupKFold(n_splits=a.folds)
    models = []
    oof_g = np.zeros(len(tr))
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, groups=groups)):
        m = train_gate(Xtr[ti], p_med_tr[ti], p_k1_tr[ti], y_tr[ti],
                       hidden=a.hidden, epochs=a.epochs, lr=a.lr, wd=a.wd,
                       seed=a.seed + fold)
        oof_g[vi] = m(torch.tensor(Xtr[vi], dtype=torch.float32)).detach().numpy()
        models.append(m)
    oof_pred = oof_g * p_med_tr + (1.0 - oof_g) * p_k1_tr
    out = dict(train_rows=int(len(tr)), val_rows=int(len(va)),
               use_subset_id=use_id,
               oof=dict(k1=engine_mean_rmse(tr, p_k1_tr),
                        median=engine_mean_rmse(tr, p_med_tr),
                        learned=engine_mean_rmse(tr, oof_pred)),
               per_subset={})
    for subset in SUBSETS:
        sub = tr[tr['subset'] == subset]
        m = sub.index
        out['per_subset'][subset] = dict(
            oof_k1=engine_mean_rmse(sub, p_k1_tr[m]),
            oof_median=engine_mean_rmse(sub, p_med_tr[m]),
            oof_learned=engine_mean_rmse(sub, oof_pred[m]),
            oof_gate_mean=float(oof_g[m].mean()))

    for subset in SUBSETS:
        vs = va[va['subset'] == subset].reset_index(drop=True)
        Xva = design(vs, use_id)
        Xva = (Xva - mu) / sd
        gv = np.mean([m(torch.tensor(Xva, dtype=torch.float32)).detach().numpy()
                      for m in models], axis=0)
        if a.random_gate:
            rng = np.random.default_rng(a.random_seed)
            rate = float(gv.mean())
            gv = (rng.random(len(gv)) <= rate).astype(float)
        p_med = vs['p_med'].to_numpy(float)
        p_k1 = vs['p_k1'].to_numpy(float)
        pred = gv * p_med + (1.0 - gv) * p_k1
        rows = []
        for name, pr in [('k1', p_k1), ('median', p_med), ('learned', pred)]:
            rows.append(dict(config=name, engine_mean_rmse=engine_mean_rmse(vs, pr)))
        per_seed = []
        for seed, g in vs.groupby('seed'):
            pos = g.index.to_numpy()
            per_seed.append(dict(
                seed=int(seed),
                k1=engine_mean_rmse(g, p_k1[pos]),
                median=engine_mean_rmse(g, p_med[pos]),
                learned=engine_mean_rmse(g, pred[pos]),
                gate_mean=float(gv[pos].mean())))
        out['per_subset'].setdefault(subset, {}).update(
            val=rows, val_gate_mean=float(gv.mean()),
            val_frac_gt_half=float((gv > 0.5).mean()),
            n_val=int(len(vs)), n_engines=int(vs['engine'].nunique()),
            per_seed=per_seed)
        print('%s val k1=%.4f median=%.4f learned=%.4f delta=%+.4f gate=%.3f'
              % (subset, rows[0]['engine_mean_rmse'], rows[1]['engine_mean_rmse'],
                 rows[2]['engine_mean_rmse'],
                 rows[0]['engine_mean_rmse'] - rows[2]['engine_mean_rmse'],
                 gv.mean()))
        for r in per_seed:
            print('  s%d k1=%.4f med=%.4f learned=%.4f d=%+.4f g=%.3f'
                  % (r['seed'], r['k1'], r['median'], r['learned'],
                     r['k1'] - r['learned'], r['gate_mean']))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    if a.save_models:
        torch.save(dict(models=[m.state_dict() for m in models],
                        mu=mu, sd=sd, feats=FEATS_G, use_subset_id=use_id),
                   a.save_models)
        print('saved models to', a.save_models)


if __name__ == '__main__':
    main()
