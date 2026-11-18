"""SHI-LTAG: LTAG gate with the SMHI sensor-HI expert as an extra feature.

Same protocol as learned_gate_global.py: training engines only, GroupKFold by
(subset, engine), no test labels. The only change is the feature set: the SMHI
prediction and its differences from K=1 / median are added. Validation only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from paper_b.modules.learned_gate import Gate, engine_mean_rmse, train_gate
from paper_b.modules.learned_gate_global import FEATS_G, SUBSETS, load_glob

FEATS_SHI = FEATS_G + ['p_smhi', 'p_smhi_minus_k1', 'p_smhi_minus_med']


def design(df, use_subset_id=True):
    X = df[FEATS_SHI].to_numpy(float)
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
    ap.add_argument('--train-glob', default=str(Path.home() / 'sprint24/experiments/hc/cata_gate/train_shi/*.csv'))
    ap.add_argument('--val-glob', default=str(Path.home() / 'sprint24/experiments/hc/cata_gate/preds_shi/*.csv'))
    ap.add_argument('--out', default=str(Path.home() / 'sprint24/experiments/hc/cata_gate/shi_ltag.json'))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    tr = load_glob(a.train_glob)
    va = load_glob(a.val_glob)
    Xtr = design(tr, False)
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
    oof = np.zeros(len(tr))
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, groups=groups)):
        m = train_gate(Xtr[ti], p_med_tr[ti], p_k1_tr[ti], y_tr[ti],
                       hidden=a.hidden, epochs=a.epochs, lr=a.lr, wd=a.wd,
                       seed=a.seed + fold)
        oof[vi] = m(torch.tensor(Xtr[vi], dtype=torch.float32)).detach().numpy()
        models.append(m)
    oof_pred = oof * p_med_tr + (1 - oof) * p_k1_tr
    out = dict(oof=dict(k1=engine_mean_rmse(tr, p_k1_tr),
                        median=engine_mean_rmse(tr, p_med_tr),
                        learned=engine_mean_rmse(tr, oof_pred)),
               per_subset={})
    for subset in SUBSETS:
        vs = va[va['subset'] == subset].reset_index(drop=True)
        Xva = (design(vs, False) - mu) / sd
        gv = np.mean([m(torch.tensor(Xva, dtype=torch.float32)).detach().numpy()
                      for m in models], axis=0)
        p_med = vs['p_med'].to_numpy(float)
        p_k1 = vs['p_k1'].to_numpy(float)
        pred = gv * p_med + (1 - gv) * p_k1
        per_seed = []
        for seed, g in vs.groupby('seed'):
            pos = g.index.to_numpy()
            per_seed.append(dict(seed=int(seed),
                                 k1=engine_mean_rmse(g, p_k1[pos]),
                                 median=engine_mean_rmse(g, p_med[pos]),
                                 learned=engine_mean_rmse(g, pred[pos]),
                                 gate_mean=float(gv[pos].mean())))
        out['per_subset'][subset] = dict(
            k1=engine_mean_rmse(vs, p_k1),
            median=engine_mean_rmse(vs, p_med),
            learned=engine_mean_rmse(vs, pred),
            delta=engine_mean_rmse(vs, p_k1) - engine_mean_rmse(vs, pred),
            gate_mean=float(gv.mean()), per_seed=per_seed)
        print('%s k1=%.4f median=%.4f shi-ltag=%.4f delta=%+.4f gate=%.3f'
              % (subset, out['per_subset'][subset]['k1'],
                 out['per_subset'][subset]['median'],
                 out['per_subset'][subset]['learned'],
                 out['per_subset'][subset]['delta'], gv.mean()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
