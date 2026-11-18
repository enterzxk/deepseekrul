"""Evaluate the frozen global learned gate on the official test features.

The gate models and feature scaler are loaded from the checkpoint written by
``learned_gate_global.py --save-models``. No test label is used for fitting or
for choosing the gate; labels are used only to score the predictions.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_b.modules.learned_gate import Gate, canon
from paper_b.modules.learned_gate_global import design


def rmse(pred, y):
    p = np.clip(np.asarray(pred, float), 0.0, 125.0)
    return float(np.sqrt(np.mean((p - np.asarray(y, float)) ** 2)))


def nasa(pred, y):
    d = np.clip(np.asarray(pred, float), 0.0, 125.0) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0,
                                np.exp(d / 10.0) - 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-glob', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    ck = torch.load(a.models, map_location='cpu', weights_only=False)
    feats = ck['feats']
    mu, sd = ck['mu'], ck['sd']
    models = []
    for sdct in ck['models']:
        m = Gate(len(feats) + 2 + (4 if ck['use_subset_id'] else 0) + 1,
                 hidden=16)
        m.load_state_dict(sdct)
        m.eval()
        models.append(m)

    out = {}
    for p in sorted(glob.glob(a.features_glob)):
        df = canon(pd.read_csv(p))
        if 'subset' not in df.columns:
            df['subset'] = Path(p).name.split('_')[0]
        X = design(df, ck['use_subset_id'])
        X = (X - mu) / sd
        g = np.mean([m(torch.tensor(X, dtype=torch.float32)).detach().numpy()
                     for m in models], axis=0)
        p_k1 = df['p_k1'].to_numpy(float)
        p_med = df['p_med'].to_numpy(float)
        pred = g * p_med + (1.0 - g) * p_k1
        y = df['y_cap'].to_numpy(float)
        rec = dict(n=int(len(df)), gate_mean=float(g.mean()),
                   k1=dict(rmse=rmse(p_k1, y), nasa=nasa(p_k1, y),
                           mae=float(np.mean(np.abs(np.clip(p_k1, 0, 125) - y))),
                           bias=float(np.mean(np.clip(p_k1, 0, 125) - y))),
                   median=dict(rmse=rmse(p_med, y), nasa=nasa(p_med, y)),
                   learned=dict(rmse=rmse(pred, y), nasa=nasa(pred, y),
                                mae=float(np.mean(np.abs(np.clip(pred, 0, 125) - y))),
                                bias=float(np.mean(np.clip(pred, 0, 125) - y))))
        out[Path(p).stem] = rec
        print('%-10s n=%3d k1=%.4f med=%.4f learned=%.4f delta=%+.4f g=%.3f'
              % (Path(p).stem, rec['n'], rec['k1']['rmse'], rec['median']['rmse'],
                 rec['learned']['rmse'], rec['k1']['rmse'] - rec['learned']['rmse'],
                 rec['gate_mean']))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
