"""LTAG v3 Phase 1b: post-hoc G1.3 ensemble and Linear-Stack feature ablation.

All features observable (no cycle_frac / true lifetime). Test sealed.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from paper_b.eval.metrics_v2 import engine_mean_rmse, mae, nasa_mean, pooled_rmse
from paper_b.modules.learned_gate import FEATS, canon, train_gate

CTX = ['regime_dist', 'cap', 'cut_norm', 'hist_norm']
FEATS_OBS = [f for f in FEATS if f != 'cycle_frac'] + ['cut_norm', 'hist_norm']
MW = ['p_med', 'p_mean', 'p_std', 'stat', 'shift', 'p_med_minus_k1', 'p_mean_minus_k1']


def load(root, kind):
    frames = []
    for p in sorted(glob.glob(str(Path(root) / kind / '*.csv'))):
        name = Path(p).name
        subset = name.split('_')[0]
        seed = int(name.split('_s')[1].split('.')[0])
        if seed not in (101, 202, 303):
            continue
        df = canon(pd.read_csv(p))
        df['subset'] = subset
        df['seed'] = seed
        df['cut_norm'] = df['cut'] / 250.0
        df['hist_norm'] = np.minimum(df['cut'], 70.0) / 70.0
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ens(df):
    num = [c for c in df.columns if c not in ('subset', 'seed')]
    g = df.groupby(['subset', 'engine', 'cut'], as_index=False)[num].mean()
    return g


def met(df, pred, name):
    y = df['y_cap'].to_numpy(float)
    return dict(arm=name, engine_mean=engine_mean_rmse(y, pred, df['engine']),
                pooled=pooled_rmse(y, pred), mae=mae(y, pred), nasa=nasa_mean(y, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/home/user/sprint24/experiments/hc/cata_gate')
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--out', default='/home/user/sprint24/results/ltagv2_p1b.json')
    a = ap.parse_args()
    tr = load(a.root, 'train')
    va = load(a.root, 'preds')
    va = va[va.subset == a.subset].copy()
    tre, vae = ens(tr), ens(va)
    groups = tre['engine'].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=5)
    p_k1_tr, p_med_tr = tre['p_k1'].to_numpy(float), tre['p_med'].to_numpy(float)
    p_k1_va, p_med_va = vae['p_k1'].to_numpy(float), vae['p_med'].to_numpy(float)
    y_tr = tre['y_cap'].to_numpy(float)

    def ridge(cols):
        Ztr = tre[cols].to_numpy(float)
        Zva = vae[cols].to_numpy(float)
        m = Ridge(alpha=1.0).fit(Ztr, y_tr)
        return m.predict(Zva)

    rows = [met(vae, p_k1_va, 'Ens3-K1'), met(vae, p_med_va, 'Ens3-Median')]
    # Ensemble LTAG: retrain gate on ensemble train features
    Xtr = tre[FEATS_OBS].to_numpy(float)
    Xva = vae[FEATS_OBS].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    preds = []
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, groups=groups)):
        m = train_gate(Xtr[ti], p_med_tr[ti], p_k1_tr[ti], y_tr[ti], seed=7 + fold)
        g = m(torch.tensor(Xva, dtype=torch.float32)).detach().numpy()
        preds.append(g * p_med_va + (1 - g) * p_k1_va)
    rows.append(met(vae, np.mean(preds, 0), 'Ens3-LTAG'))

    rows.append(met(vae, ridge(['p_k1', 'p_med']), 'LS-2'))
    rows.append(met(vae, ridge(['p_k1', 'p_med'] + CTX), 'LS-ctx'))
    rows.append(met(vae, ridge(['p_k1', 'p_med'] + FEATS_OBS), 'LS-all'))
    res = dict(subset=a.subset, n_train=int(len(tre)), n_val=int(len(vae)), table=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
