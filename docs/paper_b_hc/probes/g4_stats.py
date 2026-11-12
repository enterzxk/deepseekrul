"""G4 statistics, corrected after the independent review.

Three fixes over g4_probe2/probe3, all raised in review:

  1. RANKS. ``argsort(argsort)`` breaks ties by index. ``y`` ties at the 125
     cap (a long FD004 engine hits it on three cuts), so the old rho(age, y) of
     -1.00 was a tie-breaking artifact. Ranks here are average ranks.

  2. INDEPENDENCE. The old OLS treated 16k-49k windows as independent
     observations and reported no uncertainty. Windows are clustered by engine,
     so the slope gets a cluster-robust SE and an engine-level bootstrap CI.
     ``p_A1`` is also in-sample on train, which inflates its significance; val
     is the out-of-sample read and is reported alongside.

  3. ESTIMAND. The old data baseline was a pooled y~c slope inside p_A1
     deciles, which is dragged down by the (negative) within-engine variation.
     The quantity the probe claims to measure is between-engine: at the same
     health level, does the older engine have more remaining life? That is
     estimated by aggregating each engine to an engine mean inside a health
     band, then regressing across engines.

Evidence runs come from the ledger (``in_gate_a_table`` + ``status=ok``), not
from re-deriving the earliest ``closed_utc``.

Usage: python docs/paper_b_hc/probes/g4_stats.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import data_hc
from paper_b.hc.model_hc import build_hc
from paper_b.hc.train_hc import predict_hc

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
OUT = ROOT / 'experiments/hc'
LEDGER = ROOT / 'docs/paper_b_hc/registry/runs.csv'
DIAG = OUT / 'tmp' / 'g4_diag.json'
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
ARMS = ['A1', 'A2', 'A3a', 'A4', 'A5', 'A3p']
NBOOT = 800
NPERM = 0


def ledger_runs():
    """Admitted evidence per cell, straight from the ledger."""
    keep = {}
    with LEDGER.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['in_gate_a_table'].strip().lower() != 'true':
                continue
            if row['status'].strip() != 'ok':
                continue
            key = (row['variant'], row['subset'])
            keep[key] = row['run_id']
    return keep


def rankdata(a):
    a = np.asarray(a, np.float64)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den else float('nan')


def unit_ids_train(ds):
    sizes = ds['tr'].groupby('unit', sort=True).size().to_numpy()
    return np.repeat(np.arange(len(sizes)), sizes)


def unit_ids_val(ds, cuts):
    out = []
    for k, (_, g) in enumerate(ds['va'].groupby('unit', sort=True)):
        n = len(g)
        seen = set()
        for f in cuts:
            c = int(min(n, max(1, round(f * n))))
            if c in seen:
                continue
            seen.add(c)
            out.append(k)
    return np.asarray(out)


def cluster_ols(X, y, g):
    X1 = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.pinv(X1.T @ X1)
    beta = XtX_inv @ (X1.T @ y)
    u = y - X1 @ beta
    meat = np.zeros((X1.shape[1], X1.shape[1]))
    for gid in np.unique(g):
        s = g == gid
        v = X1[s].T @ u[s]
        meat += np.outer(v, v)
    G, n, k = len(np.unique(g)), len(y), X1.shape[1]
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = XtX_inv @ meat @ XtX_inv * corr
    return beta, np.sqrt(np.diag(V))


def boot_ci(X, y, g, nb=NBOOT, seed=101):
    rng = np.random.default_rng(seed)
    gs = np.unique(g)
    idx = {gid: np.where(g == gid)[0] for gid in gs}
    out = []
    for _ in range(nb):
        pick = rng.choice(gs, size=len(gs), replace=True)
        rows = np.concatenate([idx[gid] for gid in pick])
        X1 = np.column_stack([np.ones(len(rows)), X[rows]])
        beta = np.linalg.lstsq(X1, y[rows], rcond=None)[0]
        out.append(beta[-1])
    out = np.asarray(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def between_engine(c, y, p, g, nband=10, min_eng=5, min_win=2):
    """At fixed health, regress engine-mean RUL on engine-mean age."""
    edges = np.quantile(p, np.linspace(0.0, 1.0, nband + 1))
    num = den = 0.0
    for i in range(nband):
        band = (p >= edges[i]) & (p <= edges[i + 1])
        pts = []
        for gid in np.unique(g[band]):
            s = band & (g == gid)
            if s.sum() >= min_win:
                pts.append((c[s].mean(), y[s].mean()))
        if len(pts) < min_eng:
            continue
        e = np.asarray(pts)
        b = np.linalg.lstsq(np.column_stack([np.ones(len(e)), e[:, 0]]),
                            e[:, 1], rcond=None)[0][1]
        num += b * len(e)
        den += len(e)
    return num / den if den else float('nan')


def load(run_id):
    run = OUT / run_id
    cfg = json.loads((run / 'config.json').read_text())
    net = build_hc(cfg)
    net.load_state_dict(torch.load(run / 'ckpt' / 'best.pt',
                                   map_location='cpu', weights_only=True))
    return cfg, net.to(torch.device('cuda:0')).eval()


def main():
    dev = torch.device('cuda:0')
    runs = ledger_runs()
    print('ledger-admitted cells: %d' % len(runs))
    diag = {(r['variant'], r['subset']): r for r in json.loads(DIAG.read_text())}

    print('\n## A. within-engine Spearman(age, pred), average ranks\n')
    print('%-4s %-6s %8s %8s %8s %7s %7s %7s'
          % ('arm', 'subset', 'rho', 'min_eng', 'p05_eng', 'n_eng', 'neg_eng',
             'weak'))
    rho_agg = {}
    for s in SUBSETS:
        cfg, net = load(runs[('A1', s)])
        ds = data_hc.build(cfg['subset'], cfg['data_dir'], cfg['span'],
                           cfg['val_cuts'], cycle_source='raw')
        va = ds['val']
        raw = np.asarray(va['cycle_raw'], np.float32)
        gid = unit_ids_val(ds, cfg['val_cuts'])
        for v in ARMS:
            cfgv, netv = load(runs[(v, s)])
            # A3p asserts cycle <= 250, so each arm gets its own cycle_hc.
            va_arm = dict(va)
            va_arm['cycle_hc'] = data_hc.cycle_vector(raw, cfgv['cycle_source'])
            p = predict_hc(netv, va_arm, dev)
            per = [spearman(raw[gid == g], p[gid == g])
                   for g in np.unique(gid) if (gid == g).sum() >= 3]
            print('%-4s %-6s %8.3f %8.3f %8.3f %8d %7d %7d'
                  % (v, s, spearman(raw, p), min(per),
                     float(np.percentile(per, 5)), len(per),
                     sum(1 for x in per if x < 0),
                     sum(1 for x in per if x > -0.9)))
            rho_agg.setdefault(v, []).append(float(np.mean(per)))

    print('\n## B. OLS y ~ p_A1 + c, cluster-robust by engine\n')
    print('%-6s %6s %9s %9s %6s %18s' % ('subset', 'set', 'b_per10', 'se*10',
                                         't', 'engine bootstrap 95% CI'))
    bt = {}
    for s in SUBSETS:
        cfg, net = load(runs[('A1', s)])
        ds = data_hc.build(cfg['subset'], cfg['data_dir'], cfg['span'],
                           cfg['val_cuts'], cycle_source='raw')
        for tag, d, gid in (('train', ds['train'], unit_ids_train(ds)),
                            ('val', ds['val'], unit_ids_val(ds, cfg['val_cuts']))):
            p = predict_hc(net, d, dev)
            c = np.asarray(d['cycle_raw'], np.float32)
            y = np.asarray(d['y'], np.float32)
            beta, se = cluster_ols(np.column_stack([p, c]), y, gid)
            lo, hi = boot_ci(np.column_stack([p, c]), y, gid)
            bt[(s, tag)] = (beta[2], lo, hi)
            print('%-6s %6s %9.4f %9.4f %6.2f   [%+.3f, %+.3f]'
                  % (s, tag, beta[2] * 10, se[2] * 10,
                     beta[2] / se[2] if se[2] else float('nan'), lo * 10, hi * 10))

    print('\n## C. between-engine slope inside health bands (per 10 cycles)\n')
    print('%-6s %10s %10s' % ('subset', 'train', 'val'))
    be = {}
    for s in SUBSETS:
        cfg, net = load(runs[('A1', s)])
        ds = data_hc.build(cfg['subset'], cfg['data_dir'], cfg['span'],
                           cfg['val_cuts'], cycle_source='raw')
        row = []
        for tag, d, gid in (('train', ds['train'], unit_ids_train(ds)),
                            ('val', ds['val'], unit_ids_val(ds, cfg['val_cuts']))):
            p = predict_hc(net, d, dev)
            c = np.asarray(d['cycle_raw'], np.float32)
            y = np.asarray(d['y'], np.float32)
            row.append(between_engine(c, y, p, gid) * 10.0)
        be[s] = row
        print('%-6s %10.3f %10.3f' % (s, row[0], row[1]))

    print('\n## D. model response vs the between-engine estimate\n')
    print('%-4s %10s %10s %8s' % ('arm', 'model/+10', 'data/+10', 'ratio'))
    data_hi = float(np.max([be[s][0] for s in SUBSETS]))
    data_lo = float(np.min([be[s][0] for s in SUBSETS]))
    for v in ['A2', 'A3a', 'A4', 'A5', 'A3p']:
        m = float(np.mean([diag[(v, s)]['abs10']['mean'] for s in SUBSETS]))
        print('%-4s %10.3f %10s %8s'
              % (v, m, '%.2f-%.2f' % (data_lo, data_hi),
                 '%.1f-%.1f' % (m / data_hi, m / data_lo)))

    print('\n## E. per-arm mean within-engine rho (average ranks)\n')
    for v in ARMS:
        print('%-4s %8.3f' % (v, float(np.mean(rho_agg[v]))))


if __name__ == '__main__':
    main()
