"""Evaluate a vanilla-transformer ablation checkpoint on the official C-MAPSS test set."""
import argparse, sys
import numpy as np, pandas as pd, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'candidates'))
from dataset import (load_cmapss_train, load_cmapss_files,
                     OperatingConditionScaler, SELECTED_SENSOR_COLUMNS)
from candidates.dspt_vanilla import build, MAX_SPAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--split-seed', type=int, default=7)
    ap.add_argument('--span', type=int, default=70)
    ap.add_argument('--rul-cap', type=float, default=125.0)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--ms', action='store_true')
    ap.add_argument('--sa', action='store_true')
    ap.add_argument('--hi', action='store_true')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    raw = load_cmapss_train(ROOT / 'data' / 'CMAPSS', a.subset)
    units = np.sort(raw['unit'].unique())
    rng = np.random.default_rng(a.split_seed + 20261020)
    sh = rng.permutation(units)
    n_val = max(1, int(round(units.size * 0.2)))
    val_units = set(int(x) for x in sh[a.fold * n_val:(a.fold + 1) * n_val])
    train_units = [int(x) for x in units if int(x) not in val_units]
    tr = raw[raw.unit.isin(train_units)].sort_values(['unit', 'cycle']).reset_index(drop=True)

    cols = list(SELECTED_SENSOR_COLUMNS)
    scaler = OperatingConditionScaler(cols, n_clusters=6, random_state=a.split_seed)
    scaler.fit_transform(tr)

    _tr, test, ytrue = load_cmapss_files(ROOT / 'data' / 'CMAPSS', a.subset)
    te = scaler.transform(test)
    tf = test.sort_values(['unit', 'cycle']).reset_index(drop=True)

    xs, ys, ms = [], [], []
    for uid, g in tf.groupby('unit', sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(te[pos], dtype=np.float32)
        m = min(len(seq), a.span)
        x = np.zeros((a.span, seq.shape[1]), np.float32)
        x[a.span - m:] = seq[len(seq) - m:]
        mask = np.zeros(a.span, np.float32); mask[a.span - m:] = 1.0
        xs.append(x); ms.append(mask)
    X = np.stack(xs); MK = np.stack(ms)

    net = build(14, a.d_model, a.layers, a.dropout, max_span=a.span,
                use_ms=a.ms, use_sa=a.sa, use_hi=a.hi).cuda()
    net.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=True))
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            o = net(torch.from_numpy(X[i:i+256]).cuda(), mask=torch.from_numpy(MK[i:i+256]).cuda())
            preds.append(o['prediction'].reshape(-1).cpu().numpy())
    p = np.concatenate(preds)
    p = np.clip(p, 0.0, a.rul_cap)
    yt = np.minimum(np.asarray(ytrue, dtype=np.float64), a.rul_cap)
    pd.DataFrame({'unit': np.arange(1, len(p)+1), 'y_true': yt, 'y_pred': p}).to_csv(a.out, index=False)
    rmse = float(np.sqrt(np.mean((p - yt)**2)))
    print('DONE %s rmse=%.4f n=%d' % (a.subset, rmse, len(p)), flush=True)


if __name__ == '__main__':
    main()
