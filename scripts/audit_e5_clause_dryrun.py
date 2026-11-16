"""Dry run of the H1 additional clause BEFORE any E5 arm result exists.

Builds an lr4 and an lr14 model from the current code (no checkpoint needed),
makes the branch non-trivial by randomising the zero-initialised last layer,
runs one validation pass, and evaluates exactly the clause-5..14 comparison
that scripts/audit_e5_mech.py performs.

This is a plumbing check, not a result: the numbers come from random weights.

Usage:
  python3 scripts/audit_e5_clause_dryrun.py --data-dir /path/CMAPSS
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from paper_b.training.data import prepare_train          # noqa: E402
from paper_b.training.train import _build                 # noqa: E402
import audit_e5_mech as mech                              # noqa: E402

BASE = dict(
    model='dspt', span=70, epochs=40, patience=12, batch=512, lr=1e-3,
    weight_decay=1e-4, d_model=64, layers=2, dropout=0.2, mode='dual',
    ms_mode='full', sa_mode='basic', hi_source='hf', hi_mono='correct',
    hi_lambda=0.5, hi_mono_coef=0.1, use_cycle=True, use_pad_mask=True,
    input_mode='raw', aux_dual=True, instnorm=False, rul_cap=125, fp32=True,
    split_seed=17, fold=0, val_frac=0.2, n_cond=6,
    cuts=[0.4, 0.55, 0.7, 0.85, 0.95],
    legacy_rul=False, global_hi=False, legacy_ms_pad=False, legacy_ema=False,
)


def eigenvalues_for(lr_mode, subset, va, dev, seed=101):
    cfg = {**BASE, 'subset': subset, 'seed': seed, 'lr_mode': lr_mode}
    torch.manual_seed(seed)
    net = _build(cfg).to(dev)
    # the branch's last layer is zero-initialised by design; randomise it so
    # the clause is exercised with a live branch instead of an exact identity
    g = net.lr.g
    torch.nn.init.normal_(g[-1].weight, std=0.02)
    torch.nn.init.normal_(g[-1].bias, std=0.02)
    net.eval()
    H, _, rel = mech.capture(net, va, dev, cfg)
    C = mech.chat(H)
    lam = torch.linalg.eigvalsh(C.double()).clamp_min(0).flip(0)
    return np.asarray([float(v) for v in lam]), net, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--subset', default='FD001')
    a = ap.parse_args()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds = prepare_train(a.subset, Path(a.data_dir), span=BASE['span'], split_seed=17,
                       fold=0, val_frac=0.2, rul_cap=BASE['rul_cap'],
                       n_cond=BASE['n_cond'], cuts=tuple(BASE['cuts']))
    va = ds['val']

    e4, net4, rel4 = eigenvalues_for('lr4', a.subset, va, dev)
    e14, net14, rel14 = eigenvalues_for('lr14', a.subset, va, dev)
    print('lr4  params', sum(p.numel() for p in net4.parameters()),
          'raw_u', tuple(net4.lr.raw_u.shape))
    print('lr14 params', sum(p.numel() for p in net14.parameters()),
          'raw_u', tuple(net14.lr.raw_u.shape))

    # exactly the clause-5..14 comparison from audit_e5_mech.py, on one seed
    t4, t14 = e4[4:], e14[4:]
    diff = float(np.abs(t4 - t14).max())
    print('\nclause 5..14 (single seed dry run)')
    print('  max|eig_lr4 - eig_lr14| = %.3e' % diff)
    print('  dtype of eigenvalues   = float64')
    print('  clause would read: distinguishable =', diff > 0.0,
          '(single seed -> no sd yet; real run uses 3 seeds)')
    print('\nDRY RUN OK: the clause path executes end to end before any E5 arm result.')


if __name__ == '__main__':
    main()
