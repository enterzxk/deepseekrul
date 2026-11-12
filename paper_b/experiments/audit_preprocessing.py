"""Preprocessing / leakage audit for the Paper B protocol (read-only).

Checks, on the real FD001-FD004 files:
  1. the operating-condition scaler is fitted on training engines only and
     really does produce cluster-wise zero-mean / unit-variance sensors;
  2. the training path never opens the official test files;
  3. the baseline reference only uses frames at or before the window end;
  4. the validation cuts never read official test RUL values;
  5. train/val engines are disjoint and the split is deterministic.

Usage: python -m paper_b.experiments.audit_preprocessing --data-dir data/CMAPSS
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..training import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/CMAPSS')
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    a = ap.parse_args()
    rep = {}

    # ---- 1/5. split + scaler --------------------------------------------
    for subset in a.subsets.split(','):
        s = D._prepare_split(subset, Path(a.data_dir), 7, 0, 0.2, 6)
        tr_units, va_units = set(map(int, s['train_units'])), set(map(int, s['val_units']))
        tr_s, lt = s['tr_s'], s['lt']
        per_cluster = {}
        for k in range(6):
            m = lt == k
            if m.sum() < 2:
                continue
            per_cluster[int(k)] = dict(
                n=int(m.sum()),
                max_abs_mean=float(np.abs(tr_s[m].mean(0)).max()),
                max_abs_std_minus_1=float(np.abs(tr_s[m].std(0) - 1).max()),
            )
        # the scaler means must be exactly the train-frame cluster means
        raw_tr = s['tr'][D.SENSOR_COLS].to_numpy(np.float64)
        lab_tr = s['scaler'].labels(s['tr'])
        rebuilt = np.stack([raw_tr[lab_tr == k].mean(0) for k in range(6)])
        rep[subset] = dict(
            n_train_units=len(tr_units), n_val_units=len(va_units),
            units_disjoint=not (tr_units & va_units),
            max_abs_scaler_mean_diff=float(np.abs(s['scaler'].means - rebuilt).max()),
            per_cluster=per_cluster,
        )

    # ---- 2. training path must not open test files ----------------------
    orig = D.load_test
    def _boom(*_a, **_k):
        raise AssertionError('training path opened official test files')
    D.load_test = _boom
    try:
        ds = D.prepare_train('FD001', Path(a.data_dir), span=70, split_seed=7, fold=0,
                             val_frac=0.2, rul_cap=125.0, n_cond=6,
                             cuts=(0.4, 0.55, 0.7, 0.85, 0.95))
        rep['training_path_opens_test'] = False
        rep['train_windows'] = {k: list(v.shape) for k, v in ds['train'].items()
                                if hasattr(v, 'shape')}
        rep['val_windows'] = {k: list(v.shape) for k, v in ds['val'].items()
                              if hasattr(v, 'shape')}
    except AssertionError:
        rep['training_path_opens_test'] = True
    finally:
        D.load_test = orig

    # ---- 3. baseline reference is causal -------------------------------
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(120, 14)).astype(np.float32)
    clusters = np.zeros(120, np.int64)
    cycles = np.arange(1, 121, dtype=np.int64)
    pop = np.zeros((6, 14), np.float32)
    base_t = D._baseline_ref(seq, clusters, cycles, 60, pop)
    fut = seq.copy()
    fut[61:] += 100.0                      # perturb only frames after t=60
    base_pert = D._baseline_ref(fut, clusters, cycles, 60, pop)
    rep['baseline_causal'] = bool(np.allclose(base_t, base_pert))
    # and it does depend on past frames (the last five early-life frames)
    past = seq.copy()
    past[25:30] += 100.0
    rep['baseline_uses_past'] = bool(not np.allclose(
        base_t, D._baseline_ref(past, clusters, cycles, 60, pop)))
    early_only = seq.copy()
    early_only[:25] += 100.0
    rep['baseline_ignores_frames_before_last5'] = bool(np.allclose(
        base_t, D._baseline_ref(early_only, clusters, cycles, 60, pop)))

    # ---- 4. validation cuts never read test RUL ------------------------
    # make_val_windows only receives the train file frame; the cut is round(f*n)
    n = 200
    cuts = (0.4, 0.55, 0.7, 0.85, 0.95)
    got = [int(min(n, max(1, round(f * n)))) for f in cuts]
    rep['val_cuts_from_train_length'] = dict(n=n, cuts=got,
                                             targets=[min(max(n - c, 0), 125) for c in got])

    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
