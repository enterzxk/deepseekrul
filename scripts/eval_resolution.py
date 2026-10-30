"""Does a multi-cutoff evaluation buy resolution at fixed compute?

Same trained runs, same held-out engines, same training budget. The only change is how
many predictions each engine contributes to a contrast. Engines are restricted to those
that have every cutoff in the design, so the only thing that varies is observations per
engine.

Reported per design: the engine-clustered 95% CI width of DeltaRMSE between two arms,
and its ratio to the single-cutoff width.

Re-analysis of stored predictions: no GPU, no retraining.
"""
import argparse, glob, json, os
from collections import Counter
import numpy as np
import pandas as pd

SEEDS = [2027, 2028, 2029]
REPS = 100_000
RNG = [7000, 7001, 7002]
ROOT = '/home/yndx/zxk/bearing-rul-pi/experiments'
DESIGNS = {'single': [100], 'pair': [100, 140], 'all': [60, 100, 140, 180]}


def load(root, name, seed, dataset):
    g = sorted(glob.glob(f'{ROOT}/{root}/{name}_s{seed}/{dataset}/*/protocol_c_predictions.csv'))
    if not g:
        raise FileNotFoundError(f'{root}/{name}_s{seed}/{dataset}')
    return pd.read_csv(g[0])


def build(root, x, y, dataset, cuts):
    """Return per-engine squared-error sums and counts for both arms (aligned)."""
    sx, sy, ks = [], [], []
    for s in SEEDS:
        a = load(root, x, s, dataset); b = load(root, y, s, dataset)
        a = a[a.abs_cut.isin(cuts)]; b = b[b.abs_cut.isin(cuts)]
        m = a.merge(b, on=['unit_id', 'abs_cut'], suffixes=('_a', '_b'))
        assert np.allclose(m.y_true_a, m.y_true_b)
        for u, g in m.groupby('unit_id'):
            if len(g) != len(cuts):
                continue
            sx.append(float(np.sum((g.y_pred_a - g.y_true_a) ** 2)))
            sy.append(float(np.sum((g.y_pred_b - g.y_true_b) ** 2)))
            ks.append(len(g))
    return np.array(sx), np.array(sy), np.array(ks, float)


def boot(sx, sy, k, rng):
    n = len(sx)
    idx = rng.integers(0, n, (REPS, n))
    rx = np.sqrt(sx[idx].sum(1) / k[idx].sum(1))
    ry = np.sqrt(sy[idx].sum(1) / k[idx].sum(1))
    return rx - ry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/dsh/e7/RESOLUTION.json')
    a = ap.parse_args()
    jobs = [('FD002', 'e3', 'A', 'BE'), ('FD002', 'e3', 'A', 'PE'), ('FD001', 'e3_fd001', 'A', 'BE')]
    out = {'designs': DESIGNS, 'reps': REPS, 'contrasts': {}}
    for dataset, root, x, y in jobs:
        key = f'{dataset} {x} vs {y}'
        out['contrasts'][key] = {}
        for dname, cuts in DESIGNS.items():
            sx, sy, k = build(root, x, y, dataset, cuts)
            reps = [boot(sx, sy, k, np.random.default_rng(sd)) for sd in RNG]
            lo = float(np.mean([np.percentile(r, 2.5) for r in reps]))
            hi = float(np.mean([np.percentile(r, 97.5) for r in reps]))
            pt = float(np.sqrt(sx.sum() / k.sum()) - np.sqrt(sy.sum() / k.sum()))
            out['contrasts'][key][dname] = dict(
                cuts=cuts, engines=int(len(sx)), obs_per_engine=int(Counter(k).most_common(1)[0][0]),
                n_obs=int(k.sum()), delta=pt, ci=[lo, hi], width=float(hi - lo),
                mc_hi=float(np.std([np.percentile(r, 97.5) for r in reps], ddof=1)))
            print(f'{key:18s} {dname:7s} cuts={str(cuts):22s} eng={len(sx):3d} '
                  f'delta={pt:+6.3f} CI=[{lo:+6.3f},{hi:+6.3f}] width={hi-lo:5.3f}')
        w1 = out['contrasts'][key]['single']['width']
        for dname in DESIGNS:
            out['contrasts'][key][dname]['width_ratio_to_single'] = (
                out['contrasts'][key][dname]['width'] / w1 if w1 else None)
        print(f'  -> width ratio all/single = '
              f'{out["contrasts"][key]["all"]["width_ratio_to_single"]:.2f}\n')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    print('->', a.out)


if __name__ == '__main__':
    main()
