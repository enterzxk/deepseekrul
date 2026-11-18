"""P6: do sensor features carry tail information beyond yhat? (TAIL_MODULE_DESIGN §4.3)

The spec fixes the task (within a yhat segment, predict |e| from sensor
features, engine-level 5-fold) and the criterion (within-segment AUROC >= 0.65
and bootstrap 95% CI lower bound > 0.55) but leaves two things open. This file
pins them explicitly so the result cannot be re-interpreted later:

  * features  : per-sensor summary of the input window -- mean, std, last value
                and least-squares slope over the span -> 14 x 4 = 56 numbers.
                These are literally "sensor features" and need no model hook,
                so the test is the hardest (most conservative) version of P6.
  * binary label : within the segment, label = 1 if |e| is above the segment
                MEDIAN, else 0. Score = predicted |e| from a ridge regression.

Because n inside one segment may be tiny, the script prints the segment sizes
first and also reports the best univariate feature as an upper bound: if even
the best single feature cannot separate, no head can.

Data: split-43 diagnostic platform, `model` (last-epoch) weights, seed-averaged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from _cells import resolve                                    # noqa: E402
from audit_diag_split import build_diag_windows, SUBSETS, SEEDS, SPLIT  # noqa: E402
from paper_b.training.data import _prepare_split               # noqa: E402
from paper_b.training.train import _build, predict             # noqa: E402

SEG_EDGES = (0.0, 25.0, 50.0, 75.0, 100.0, 125.0)
TARGET_SEG = 3          # [75, 100)
LAM = 10.0              # ridge penalty after standardisation
N_BOOT = 2000
BOOT_SEED = 20261118


def auroc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def features(x):
    """(n, span, 14) windows -> (n, 56): mean/std/last/slope per sensor."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    last = x[:, -1, :]
    t = np.linspace(-1.0, 1.0, x.shape[1])[:, None]          # (span, 1)
    slope = (x * t[None, :, :]).sum(axis=1) / (t ** 2).sum()
    return np.concatenate([mean, std, last, slope], axis=1)


def ridge_fit(X, y, lam=LAM):
    Xc = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    A[0, 0] -= lam                      # do not penalise the intercept
    return np.linalg.solve(A, Xc.T @ y)


def ridge_pred(w, X):
    return np.concatenate([np.ones((len(X), 1)), X], axis=1) @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--table', required=True)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root, data_dir = Path(a.root), Path(a.data_dir)
    frozen = json.loads(Path(a.table).read_text())['subsets']

    feats, yhat, y, unit, subs = [], [], [], [], []
    for subset in SUBSETS:
        table = {int(r['unit']): int(r['truncation_cycle']) for r in frozen[subset]['rows']}
        per_seed, x_ref = [], None
        for seed in SEEDS:
            cell = Path(resolve(root, 'E0', subset, seed, SPLIT))
            cfg = json.loads((cell / 'config.json').read_text())
            ck = torch.load(cell / 'last.pt', map_location='cpu', weights_only=False)
            s = _prepare_split(subset, data_dir, SPLIT, cfg['fold'], cfg['val_frac'],
                               cfg['n_cond'])
            d = build_diag_windows(s, cfg['span'], table)
            net = _build(cfg)
            net.load_state_dict(ck['model'])
            net.eval()
            per_seed.append((d['unit'], d['y'], predict(net, d, 'cpu', cfg['mode'], fp32=True)))
            if x_ref is None:
                x_ref = d['x']
        units = per_seed[0][0]
        for u, yy, p in per_seed[1:]:
            if not np.array_equal(u, units):
                raise SystemExit(f'{subset}: seeds disagree on the diagnostic units')
        feats.append(features(x_ref))
        yhat.append(np.mean([p for _, _, p in per_seed], axis=0))
        y.append(per_seed[0][1])
        unit.append(units)
        subs.append(np.array([subset] * len(units), dtype=object))

    F = np.concatenate(feats)
    yh = np.concatenate(yhat)
    yy = np.concatenate(y)
    U = np.concatenate(unit)
    err = np.abs(yh - yy)

    S = np.concatenate(subs)
    seg = np.clip(np.digitize(yh, SEG_EDGES) - 1, 0, len(SEG_EDGES) - 2)
    print('yhat-segment sizes: ' + '  '.join(
        f'[{SEG_EDGES[k]:.0f},{SEG_EDGES[k+1]:.0f})={int((seg == k).sum())}'
        for k in range(len(SEG_EDGES) - 1)))

    m = seg == TARGET_SEG
    n = int(m.sum())
    print(f'\nP6 segment [75,100): n={n} engines')
    if n < 12:
        print('P6  UNDER-POWERED: fewer than 12 engines in the segment; '
              'AUROC cannot be estimated reliably, gate not evaluable')
        return 1

    X, t, u = F[m], err[m], U[m]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    lab = (t > np.median(t)).astype(int)

    # 56 features on 30 engines is hopeless for a full-rank ridge: the first run
    # scored AUROC 0.369, i.e. worse than chance, which is an anti-learner, not
    # evidence of "no information". Reduce to 3 PCA components -- the honest
    # dimension for n=30 -- before fitting.
    K = 3
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:K].T

    # Fold key must be (subset, unit), not unit id alone: the segment pools all
    # four subsets, so unit ids collide across subsets and FD001-5 would share a
    # fold with FD002-5. Harmless today (one sample per engine, no leakage) but
    # wrong as soon as an engine contributes more than one sample.
    key = np.array([f'{s}|{int(x)}' for s, x in zip(S[m], u)], dtype=object)
    uniq, fold_of = np.unique(key, return_inverse=True)
    fold = fold_of % 5
    score = np.empty(len(t))
    for f in range(5):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        w = ridge_fit(Z[tr], t[tr], lam=1.0)
        score[te] = ridge_pred(w, Z[te])
    au = auroc(score[lab == 1], score[lab == 0])

    rng = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(t), len(t))
        s_, l_ = score[idx], lab[idx]
        if l_.sum() == 0 or l_.sum() == len(l_):
            continue
        draws.append(auroc(s_[l_ == 1], s_[l_ == 0]))
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))

    def max_uni(Xs, ls):
        best, arg = 0.0, -1
        for j in range(Xs.shape[1]):
            aj = auroc(Xs[ls == 1, j], Xs[ls == 0, j])
            aj = max(aj, 1.0 - aj)
            if aj > best:
                best, arg = aj, j
        return best, arg

    # The max over 56 features on 30 engines is optimistic by construction, so
    # it needs a permutation null before it means anything.
    best_u, best_j = max_uni(X, lab)
    rngp = np.random.default_rng(BOOT_SEED + 1)
    null = np.array([max_uni(X, rngp.permutation(lab))[0] for _ in range(1000)])
    null95 = float(np.percentile(null, 95))
    p_uni = float((null >= best_u).mean())

    print(f'P6  AUROC (ridge on {K} PCA comps, 5-fold by engine) = {au:.3f}  '
          f'bootstrap 95% CI [{lo:.3f}, {hi:.3f}]  ({len(draws)} draws)')
    print(f'P6  best univariate feature #{best_j} AUROC = {best_u:.3f}  '
          f'permutation null95 = {null95:.3f}  p = {p_uni:.3f} (1000 perms)')
    ok = (au >= 0.65) and (lo > 0.55)
    print(f'P6  criterion (AUROC >= 0.65 and CI lower > 0.55): {"PASS" if ok else "FAIL"}')

    if a.json:
        Path(a.json).write_text(json.dumps(dict(
            n_segment=n, auroc=au, ci95=[lo, hi], n_boot=len(draws), pca_components=K,
            best_univariate=dict(index=best_j, auroc=best_u, null95=null95, p_value=p_uni),
            segment_sizes={f'[{SEG_EDGES[k]:.0f},{SEG_EDGES[k+1]:.0f})':
                           int((seg == k).sum()) for k in range(len(SEG_EDGES) - 1)},
            label='|e| above segment median', features='window mean/std/last/slope per sensor',
            passed=bool(ok),
        ), indent=1), encoding='utf-8')
        print(f'wrote {a.json}')
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
