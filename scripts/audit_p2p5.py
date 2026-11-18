"""P2 / P3 / P4 zero-cost pre-checks on the split-43 diagnostic platform.

Definitions are frozen in TAIL_MODULE_DESIGN.md §4.1; this file implements them
literally, on the `model` (last-epoch, selection-free) weights of split 43.

P2  separability: is the 75-100 bucket's yhat median outside the 25-50 bucket's
    [Q1,Q3], AND does yhat separate the two buckets with AUROC >= 0.85?
P3  cross-fitted 5-segment offset calibration, 5 folds BY ENGINE, offset = median
    of (y - yhat) in the training folds. Pass = merged dP90 <= -0.5 AND the
    true-RUL 25-50 segment MAE worsens by <= +0.5.
P4  per-seed P90 of E0 on the same platform -> mean/sd -> threshold
    max(0.5, 1.0 * sd) for §7.1.

P90 definition (frozen, and NOT the same aggregation as the bias):
    seed-averaged per-engine error -> |e_u| -> within-subset 90th percentile ->
    equal weight over the four subsets. Weighting by engine count would let
    FD002/FD004 dominate; the bias in §2.6 uses engine-count weighting, so the
    two must not be mixed.
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
BOOT_SEED = 20261118
N_FOLD_REPS = 200


def auroc(pos, neg):
    """P(score_pos > score_neg) with average ranks for ties."""
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


def p90_by_subset(err_by_subset):
    """Frozen P90: within-subset 90th percentile of |e|, equal weight over subsets."""
    vals = [float(np.percentile(np.abs(err_by_subset[s]), 90)) for s in SUBSETS]
    return float(np.mean(vals)), vals


def rmse_by_subset(err_by_subset):
    vals = [float(np.sqrt(np.mean(err_by_subset[s] ** 2))) for s in SUBSETS]
    return float(np.mean(vals)), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--table', required=True)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root, data_dir = Path(a.root), Path(a.data_dir)
    frozen = json.loads(Path(a.table).read_text())['subsets']

    # ---- collect per-engine y / yhat / unit, and per-seed errors (P4) ----------
    y_by, yhat_by, unit_by, err_seed, subs = {}, {}, {}, {}, []
    for subset in SUBSETS:
        table = {int(r['unit']): int(r['truncation_cycle']) for r in frozen[subset]['rows']}
        per_seed = []
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
        units = per_seed[0][0]
        for u, y, p in per_seed[1:]:
            if not np.array_equal(u, units):
                raise SystemExit(f'{subset}: seeds disagree on the diagnostic units')
        y_by[subset] = per_seed[0][1]
        yhat_by[subset] = np.mean([p for _, _, p in per_seed], axis=0)
        unit_by[subset] = units
        subs.append(np.array([subset] * len(units), dtype=object))
        err_seed[subset] = [p - y for _, y, p in per_seed]

    y_all = np.concatenate([y_by[s] for s in SUBSETS])
    yhat_all = np.concatenate([yhat_by[s] for s in SUBSETS])
    unit_all = np.concatenate([unit_by[s] for s in SUBSETS])
    err_all = yhat_all - y_all

    # ---- P2 -------------------------------------------------------------------
    lo = (y_all >= 25) & (y_all < 50)
    hi = (y_all >= 75) & (y_all < 100)
    med_hi = float(np.median(yhat_all[hi]))
    q1, q3 = (float(v) for v in np.percentile(yhat_all[lo], [25, 75]))
    sep_med = not (q1 <= med_hi <= q3)
    au = auroc(yhat_all[hi], yhat_all[lo])
    sep_auc = au >= 0.85
    print(f'P2  n(25-50)={int(lo.sum())} n(75-100)={int(hi.sum())}  '
          f'median(yhat|75-100)={med_hi:.1f}  [Q1,Q3](25-50)=[{q1:.1f},{q3:.1f}]  '
          f'AUROC={au:.3f}')
    print(f'P2  median outside IQR: {sep_med}   AUROC>=0.85: {sep_auc}   '
          f'-> {"SEPARABLE (A-class alive)" if (sep_med and sep_auc) else "NOT SEPARABLE (A-class dead)"}')

    # ---- P3 -------------------------------------------------------------------
    seg = np.clip(np.digitize(yhat_all, SEG_EDGES) - 1, 0, len(SEG_EDGES) - 2)
    # Fold key must be (subset, unit): unit ids collide across subsets, so
    # np.unique(unit) would put FD001-5 and FD002-5 in the same fold. No leakage
    # today (one sample per engine) but wrong the moment that changes.
    S = np.concatenate(subs)
    key = np.array([f'{s}|{int(x)}' for s, x in zip(S, unit_all)], dtype=object)
    uniq, fold_of = np.unique(key, return_inverse=True)
    fold = fold_of % 5

    def split_by_subset(arr):
        out, i = {}, 0
        for s in SUBSETS:
            n = len(y_by[s])
            out[s] = arr[i:i + n]
            i += n
        return out

    def run_p3(fold_assign):
        yhat_cal = yhat_all.copy()
        offs = []
        for f in range(5):
            tr, te = fold_assign != f, fold_assign == f
            b = np.zeros(len(SEG_EDGES) - 1)
            for k in range(len(SEG_EDGES) - 1):
                mk = tr & (seg == k)
                if mk.any():
                    b[k] = float(np.median((y_all - yhat_all)[mk]))
            offs.append(b.tolist())
            yhat_cal[te] = yhat_all[te] + b[seg[te]]
        ec = yhat_cal - y_all
        p90_r, _ = p90_by_subset(split_by_subset(ec))
        rmse_r, _ = rmse_by_subset(split_by_subset(ec))
        mae_r = float(np.mean(np.abs(ec[mid])))
        return p90_r, rmse_r, mae_r, offs

    p90_a, rmse_a, mae_a, offsets = run_p3(fold)
    p90_b, p90_b_sub = p90_by_subset(split_by_subset(err_all))
    rmse_b, _ = rmse_by_subset(split_by_subset(err_all))
    mid = (y_all >= 25) & (y_all < 50)
    mae_b = float(np.mean(np.abs(err_all[mid])))
    d_p90, d_rmse, d_mae = p90_a - p90_b, rmse_a - rmse_b, mae_a - mae_b
    ok = (d_p90 <= -0.5) and (d_mae <= 0.5)

    # Fold sensitivity: the guardrail is a threshold on a number that depends on
    # how engines are dealt into folds. Repeat with other assignments so a
    # "0.03 over the line" cannot be reported as a settled verdict.
    rngf = np.random.default_rng(BOOT_SEED + 7)
    reps = []
    for _ in range(N_FOLD_REPS):
        perm = rngf.permutation(len(uniq))
        fa = np.empty(len(uniq), int)
        fa[perm] = np.arange(len(uniq)) % 5
        p90_r, rmse_r, mae_r, _ = run_p3(fa[fold_of])
        reps.append((p90_r - p90_b, rmse_r - rmse_b, mae_r - mae_b))
    reps = np.array(reps)
    pass_rate = float(np.mean((reps[:, 0] <= -0.5) & (reps[:, 2] <= 0.5)))
    print(f'P3  P90 {p90_b:.3f} -> {p90_a:.3f} (d={d_p90:+.3f})  '
          f'RMSE {rmse_b:.3f} -> {rmse_a:.3f} (d={d_rmse:+.3f})  '
          f'MAE(25-50) {mae_b:.3f} -> {mae_a:.3f} (d={d_mae:+.3f})')
    print(f'P3  pass (dP90 <= -0.5 and dMAE(25-50) <= +0.5): {ok}')
    print(f'P3  fold sensitivity over {N_FOLD_REPS} assignments: '
          f'dP90 [{reps[:,0].min():+.3f}, {reps[:,0].max():+.3f}]  '
          f'dMAE(25-50) [{reps[:,2].min():+.3f}, {reps[:,2].max():+.3f}]  '
          f'pass rate = {pass_rate:.0%}')

    # ---- P4 -------------------------------------------------------------------
    p90_seeds = []
    for i in range(len(SEEDS)):
        e = {s: err_seed[s][i] for s in SUBSETS}
        v, _ = p90_by_subset(e)
        p90_seeds.append(v)
    p4_mean, p4_sd = float(np.mean(p90_seeds)), float(np.std(p90_seeds, ddof=1))
    thr = max(0.5, 1.0 * p4_sd)
    print(f'P4  E0 P90 per seed {[round(x, 3) for x in p90_seeds]}  '
          f'mean={p4_mean:.3f} sd={p4_sd:.3f}  threshold=max(0.5, sd)={thr:.3f}')

    out = dict(
        P2=dict(n_lo=int(lo.sum()), n_hi=int(hi.sum()), median_hi=med_hi, q1_lo=q1, q3_lo=q3,
                median_outside_iqr=bool(sep_med), auroc=au, auroc_ok=bool(sep_auc),
                separable=bool(sep_med and sep_auc)),
        P3=dict(p90_before=p90_b, p90_after=p90_a, d_p90=d_p90,
                rmse_before=rmse_b, rmse_after=rmse_a, d_rmse=d_rmse,
                mae_mid_before=mae_b, mae_mid_after=mae_a, d_mae_mid=d_mae,
                offsets_per_fold=offsets, folds='engine % 5', passed=bool(ok)),
        P4=dict(p90_per_seed=p90_seeds, mean=p4_mean, sd=p4_sd, threshold=thr),
        p90_definition='within-subset p90 of |seed-averaged error|, equal weight over subsets',
        note='P5 not run: it is required only if the B-class route is chosen.',
    )
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print(f'wrote {a.json}')


if __name__ == '__main__':
    raise SystemExit(main())
