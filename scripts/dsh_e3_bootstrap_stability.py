"""T0: is the E3 non-inferiority verdict stable under bootstrap Monte-Carlo error?

Same estimator as dsh_e3_contrasts.py:
  Delta = RMSE(PE) - RMSE(A), paired by unit_id, averaged over seeds,
  engines resampled within seed, 97.5th percentile = one-sided upper bound.

Reports:
  * 100k reps x 5 RNG seeds (converged bound)
  * 4000 reps x 20 RNG seeds (the sampling distribution that produced +0.291)

Read-only over the existing prediction files.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

SEEDS = [2027, 2028, 2029]


def load(root, arm, seed):
    g = sorted(glob.glob(f'{root}/{arm}_s{seed}/FD002/*/matched_cutoff_predictions.csv'))
    if not g:
        raise FileNotFoundError(f'{arm} {seed}')
    d = pd.read_csv(g[0]).set_index('unit_id').sort_index()
    return d


def squared_errors(root):
    out = {}
    for arm in ['A', 'PE']:
        for s in SEEDS:
            d = load(root, arm, s)
            out[(arm, s)] = (d['y_pred'].to_numpy(float) - d['y_true'].to_numpy(float)) ** 2
    # align engines across arms inside a seed
    idx = {}
    for s in SEEDS:
        a = load(root, 'A', s).index
        p = load(root, 'PE', s).index
        common = a.intersection(p)
        idx[s] = common
    for s in SEEDS:
        common = idx[s]
        for arm in ['A', 'PE']:
            d = load(root, arm, s).loc[common]
            out[(arm, s)] = (d['y_pred'].to_numpy(float) - d['y_true'].to_numpy(float)) ** 2
    return out, idx


def bounds(se, idx, reps, rng):
    deltas = np.empty(reps)
    acc = np.zeros(reps)
    for s in SEEDS:
        n = len(idx[s])
        ii = rng.integers(0, n, (reps, n))
        ra = np.sqrt(se[('A', s)][ii].mean(axis=1))
        rp = np.sqrt(se[('PE', s)][ii].mean(axis=1))
        acc += (rp - ra)
    deltas = acc / len(SEEDS)
    return dict(mean=float(deltas.mean()), sd=float(deltas.std(ddof=1)),
                p025=float(np.percentile(deltas, 2.5)),
                p975=float(np.percentile(deltas, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='experiments/e3')
    ap.add_argument('--big', type=int, default=100000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[9000, 9001, 9002, 9003, 9004])
    ap.add_argument('--small', type=int, default=4000)
    ap.add_argument('--small_seeds', type=int, default=20)
    ap.add_argument('--out', default='results/dsh/e3/bootstrap_stability.json')
    a = ap.parse_args()

    se, idx = squared_errors(a.root)
    big, small = [], []
    for sd in a.seeds:
        r = bounds(se, idx, a.big, np.random.default_rng(sd))
        r['rng'] = sd
        big.append(r)
        print(f"100k rng={sd}: mean {r['mean']:+.3f} p97.5 {r['p975']:+.3f} sd {r['sd']:.3f}")
    for k in range(a.small_seeds):
        sd = 50000 + k
        r = bounds(se, idx, a.small, np.random.default_rng(sd))
        r['rng'] = sd
        small.append(r)
    up = np.array([r['p975'] for r in small])
    big_up = np.array([r['p975'] for r in big])
    summary = dict(
        big=[{k: v for k, v in r.items()} for r in big],
        small_p975=[float(x) for x in up],
        small_median=float(np.median(up)),
        small_frac_ge_030=float(np.mean(up >= 0.30)),
        big_mean_p975=float(big_up.mean()),
        big_frac_ge_030=float(np.mean(big_up >= 0.30)),
        verdict=('not_reproducible' if np.mean(big_up >= 0.30) >= 0.8 else
                 ('passes_converged' if np.mean(big_up < 0.30) >= 0.8 else 'ambiguous')),
        delta_point=float(np.mean([r['mean'] for r in big])),
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, big=big, small=small), open(a.out, 'w'), indent=2)
    pd.DataFrame(small).to_csv(a.out.replace('.json', '_small.csv'), index=False)
    pd.DataFrame(big).to_csv(a.out.replace('.json', '_big.csv'), index=False)
    print('\n' + json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
