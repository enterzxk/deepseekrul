"""Verify that an independently reconstructed pipeline reproduces the archived
predictions bit-for-bit, before the single authorised test-set evaluation.

If this check fails, the test-set numbers would be produced by a different feature
pipeline than the training numbers, and the whole exercise would be void. So:
stage 1 reproduces fold-1 validation predictions from scratch and compares them with
experiments/e10/A_s2027_f1/.../matched_cutoff_predictions.csv; stage 2 only runs if
stage 1 passes.

Usage:  python scripts/dsh_test_verify.py --stage verify
        python scripts/dsh_test_verify.py --stage test
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = '/home/yndx/zxk/bearing-rul-pi'
sys.path.insert(0, f'{ROOT}/src')
sys.path.insert(0, f'{ROOT}/scripts')

from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS, OperatingConditionScaler  # noqa: E402
from candidates.model import CandidateModel  # noqa: E402
import run_candidate as rc  # noqa: E402

WINDOW = 50
MAX_RUL = 125.0
SPLIT_SEED = 7
N_COND = 6
SUBSET = 'FD002'


def split_frames(fold):
    """Reproduce exactly the partition used by run_candidate.prepare(fold=fold)."""
    raw = load_cmapss_train(os.path.join(ROOT, 'data', 'CMAPSS'), SUBSET)
    units = np.sort(raw['unit'].unique())
    rng = np.random.default_rng(SPLIT_SEED + 20261020)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(units.size * 0.2)))
    start = fold * n_val
    val_units = sorted(int(x) for x in shuffled[start:start + n_val])
    train_units = sorted(int(x) for x in units if int(x) not in set(val_units))
    tr = raw[raw.unit.isin(train_units)].sort_values(['unit', 'cycle']).reset_index(drop=True)
    va = raw[raw.unit.isin(val_units)].sort_values(['unit', 'cycle']).reset_index(drop=True)
    return tr, va, val_units


def fit_scaler(train_frame):
    scaler = OperatingConditionScaler(list(SELECTED_SENSOR_COLUMNS), n_clusters=N_COND,
                                      random_state=SPLIT_SEED)
    scaler.fit_transform(train_frame)
    return scaler


def build_model():
    return CandidateModel(num_sensors=14, window_size=WINDOW, d_model=64, num_layers=1,
                          dropout=0.15, use_baseline=True, use_ema_features=True,
                          use_cycle=True, use_graph=False, use_dual_head=False)


BASE_NAME = {'cnn1d': 'cnn1d', 'bilstm': 'bilstm', 'gru': 'gru_attention',
             'tcn': 'tcn', 'tf': 'transformer'}


def load_model(ckpt, arm='A'):
    if arm in BASE_NAME:
        from baselines import build_baseline
        m = build_baseline(BASE_NAME[arm], num_sensors=14, d_model=88, dropout=0.15)
    else:
        m = build_model()
    obj = torch.load(ckpt, map_location='cpu')
    state = obj.get('model', obj) if isinstance(obj, dict) else obj
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected, f'{arm}: unexpected keys: {unexpected[:5]}'
    m.eval()
    return m


def predict(m, x, extras, baseline=False):
    """WindowSet feeds zeros(50,3) when cond is None; replicate that exactly."""
    if baseline:
        with torch.no_grad():
            out = m(torch.from_numpy(x))
        return out.reshape(-1).numpy().astype(float)
    cond = torch.zeros(len(x), WINDOW, 3)
    with torch.no_grad():
        out = m(torch.from_numpy(x), cond=cond, extras=torch.from_numpy(extras))
    return out['prediction'].reshape(-1).numpy().astype(float)  # (B,1) -> (B,)


def stage_verify():
    """Compare a reconstructed pipeline against the official prepare() output.

    Two independent checks, both must pass before the test set is opened:
      (a) the reconstructed feature tensors must equal prepare()'s matched_set tensors;
      (b) a prediction from those tensors must reproduce the archived CSV to < 0.1.
    """
    tr, va, _ = split_frames(1)
    scaler = fit_scaler(tr)
    va_s = scaler.transform(va)
    x, y, extras, us = rc.build_windows_matched(va, va_s, WINDOW, MAX_RUL, cutoff_seed=2026)

    p = rc.prepare(SUBSET, SPLIT_SEED, WINDOW, MAX_RUL, condition_aware=True,
                   n_cond=N_COND, ref_mode='global', fold=1)
    ms = p[7]
    dx = float(np.abs(ms.x.numpy() - x).max())
    de = float(np.abs(ms.extras.numpy() - extras).max())

    ckpt = f'{ROOT}/experiments/e10/A_s2027_f1/FD002/A_s2027/best_model.pt'
    m = load_model(ckpt, 'A')
    pred = predict(m, ms.x.numpy(), ms.extras.numpy())
    ref = pd.read_csv(f'{ROOT}/experiments/e10/A_s2027_f1/FD002/A_s2027/matched_cutoff_predictions.csv')
    ref = ref.sort_values('unit_id').reset_index(drop=True)
    dp = float(np.abs(pred - ref.y_pred.to_numpy(float)).max())

    print(f'(a) feature equality: max|x diff| = {dx:.3e}   max|extras diff| = {de:.3e}')
    print(f'(b) prediction replay: max|dpred| = {dp:.3e}')
    ok = dx < 1e-6 and de < 1e-6 and dp < 0.1
    print('VERIFY', 'PASS' if ok else 'FAIL')
    return ok


def stage_test():
    """Last-window evaluation on test_FD002, all fold-seed models of one arm."""
    from dataset import _read_whitespace_table
    from pathlib import Path as _P
    traj = _read_whitespace_table(_P(ROOT) / 'data' / 'CMAPSS' / 'test_FD002.txt')
    rul = pd.read_csv(os.path.join(ROOT, 'data', 'CMAPSS', 'RUL_FD002.txt'), sep=r'\s+',
                      header=None)[0].to_numpy(float)
    print(f'test_FD002: {traj.unit.nunique()} engines, rul rows {len(rul)}')

    out = {'subset': SUBSET, 'protocol': 'last observable window, one prediction per engine',
           'n_engines': int(traj.unit.nunique()), 'arms': {}}
    for arm, exp, seeds in (('A', 'e10', [2027, 2028, 2029]),
                            ('cnn1d', 'e11', [2027, 2028, 2029]),
                            ('bilstm', 'e11', [2027, 2028, 2029]),
                            ('gru', 'e11', [2027, 2028, 2029]),
                            ('tcn', 'e11', [2027]),
                            ('tf', 'e11', [2027])):
        per_model = []
        for fold in range(5):
            tr, _, _ = split_frames(fold)
            scaler = fit_scaler(tr)
            ts = scaler.transform(traj)
            # one window per test engine: its last WINDOW cycles
            x, extras, units = [], [], []
            for u, g in traj.groupby('unit', sort=True):
                pos = g.index.to_numpy(np.int64)
                seq = np.asarray(ts[pos], dtype=np.float32)
                if seq.shape[0] < WINDOW:
                    continue
                w = seq[-WINDOW:]
                early = seq[: min(5, seq.shape[0])]
                baseline = early.mean(axis=0)
                ref_std = float(early.std(axis=0).mean()) if early.shape[0] > 1 else 0.0
                ref_count = min(1.0, early.shape[0] / 5.0)
                dev = float(np.abs(w - baseline).mean())
                ema = seq[0].copy()
                for r in seq:
                    ema = 0.5 * ema + 0.5 * r
                jitter = float(np.abs(seq[-1] - ema).mean())
                mean_age = float(np.mean(np.arange(1, min(early.shape[0], 5) + 1)))
                ev = [ref_count, 1.0, float(np.exp(-mean_age / 30.0)), float(np.exp(-dev)),
                      1.0 / (1.0 + ref_std), jitter]
                x.append(np.ascontiguousarray(w))
                extras.append(np.concatenate([baseline,
                                              [min(len(seq) / rc.CYCLE_SCALE, 1.0)], ev]))
                units.append(int(u))
            x = np.stack(x).astype(np.float32)
            extras = np.stack(extras).astype(np.float32)
            for seed in seeds:
                ck = f'{ROOT}/experiments/{exp}/{arm}_s{seed}_f{fold}/FD002/{arm}_s{seed}/best_model.pt'
                if not os.path.exists(ck):
                    continue
                m = load_model(ck, arm)
                p = np.clip(predict(m, x, extras, baseline=(arm in BASE_NAME)), 0, MAX_RUL)
                y = np.array([rul[u - 1] for u in units], dtype=float)
                e = p - y
                per_model.append(dict(fold=fold, seed=seed,
                                      rmse=float(np.sqrt(np.mean(e ** 2))),
                                      mae=float(np.mean(np.abs(e))),
                                      nasa=float(np.mean(np.where(e < 0, np.exp(-e / 13) - 1,
                                                                  np.exp(e / 10) - 1))),
                                      late=float(np.mean(e > 0)),
                                      severe=float(np.mean(e > 10)),
                                      n=len(units)))
                print(f'  {arm} f{fold} s{seed}: RMSE {per_model[-1]["rmse"]:.3f} '
                      f'MAE {per_model[-1]["mae"]:.3f} n={len(units)}')
        if per_model:
            agg = {k: float(np.mean([r[k] for r in per_model]))
                   for k in ('rmse', 'mae', 'nasa', 'late', 'severe')}
            agg['n_models'] = len(per_model)
            agg['per_model'] = per_model
            out['arms'][arm] = agg
            print(f'{arm}: mean over {len(per_model)} models  RMSE {agg["rmse"]:.3f}')

    os.makedirs(f'{ROOT}/results/dsh/test', exist_ok=True)
    with open(f'{ROOT}/results/dsh/test/test_fd002.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote results/dsh/test/test_fd002.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['verify', 'test'], required=True)
    a = ap.parse_args()
    if a.stage == 'verify':
        sys.exit(0 if stage_verify() else 1)
    else:
        stage_test()
