"""E6 statistical audit: one CI recipe for every contrast that has per-engine predictions.

Conventions (fixed here, used by every number in the paper):
  Delta = RMSE(first) - RMSE(second)      positive => first is worse
  CI    = engine-clustered bootstrap, engines resampled within seed, seeds averaged
  reps  = 100,000; RNG seeds 7000/7001/7002 -> the SD of the upper bound across the
          three runs is reported as the Monte-Carlo error of the interval

Also emits the list of claims that have NO per-engine artefact and therefore cannot
be recomputed (they must be downgraded or rebuilt, not quietly kept).
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

SEEDS = [2027, 2028, 2029]
REPS = 100_000
RNG_SEEDS = [7000, 7001, 7002]
ROOT = '/home/yndx/zxk/bearing-rul-pi/experiments'


def load_npz(sub, name, seed):
    p = f'{ROOT}/{sub}/{name}_s{seed}/pred.npz'
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = np.load(p, allow_pickle=True)
    return pd.DataFrame({'unit_id': np.arange(len(d['y_true'])),
                         'y_true': d['y_true'].astype(float),
                         'y_pred': d['y_pred'].astype(float)})


def load_csv(sub, name, seed, dataset='FD002'):
    g = sorted(glob.glob(f'{ROOT}/{sub}/{name}_s{seed}/{dataset}/*/matched_cutoff_predictions.csv'))
    if not g:
        raise FileNotFoundError(f'{sub}/{name}_s{seed}')
    return pd.read_csv(g[0])[['unit_id', 'y_true', 'y_pred']]


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def contrast(arms, x, y, load):
    per_seed = {}
    err = {}
    for s in SEEDS:
        px, py = load(x, s), load(y, s)
        m = px.merge(py, on='unit_id', suffixes=('_x', '_y')).sort_values('unit_id')
        assert np.allclose(m['y_true_x'], m['y_true_y']), 'targets differ'
        err[s] = (m['y_pred_x'].to_numpy() - m['y_true_x'].to_numpy(),
                  m['y_pred_y'].to_numpy() - m['y_true_y'].to_numpy())
        per_seed[str(s)] = rmse(err[s][0]) - rmse(err[s][1])
    ups, los, means = [], [], []
    for sd in RNG_SEEDS:
        rng = np.random.default_rng(sd)
        acc = np.zeros(REPS)
        for s in SEEDS:
            ex, ey = err[s]
            n = len(ex)
            ii = rng.integers(0, n, (REPS, n))
            acc += np.sqrt((ex[ii] ** 2).mean(1)) - np.sqrt((ey[ii] ** 2).mean(1))
        acc /= len(SEEDS)
        means.append(acc.mean()); los.append(np.percentile(acc, 2.5)); ups.append(np.percentile(acc, 97.5))
    return dict(per_seed=per_seed, mean=float(np.mean(means)),
                ci_lo=float(np.mean(los)), ci_hi=float(np.mean(ups)),
                mc_sd_hi=float(np.std(ups, ddof=1)), mc_sd_lo=float(np.std(los, ddof=1)),
                excludes_zero=bool(np.mean(ups) < 0 or np.mean(los) > 0))


def contrast_r3preds(x, y, subset=None):
    d = pd.read_csv('/home/yndx/zxk/bearing-rul-pi/results/dsh/review_r3/R3_preds.csv')
    if subset:
        d = d[d.dataset == subset]
    a = d[d.arm == x][['dataset', 'engine', 'cutoff', 'y_true', 'y_pred']]
    b = d[d.arm == y][['dataset', 'engine', 'cutoff', 'y_true', 'y_pred']]
    m = a.merge(b, on=['dataset', 'engine', 'cutoff'], suffixes=('_x', '_y'))
    assert np.allclose(m.y_true_x, m.y_true_y)
    m['ex'] = m.y_pred_x - m.y_true_x
    m['ey'] = m.y_pred_y - m.y_true_y
    groups = [g for _, g in m.groupby(['dataset', 'engine'])]
    ex = [g.ex.to_numpy() for g in groups]; ey = [g.ey.to_numpy() for g in groups]
    point = rmse(np.concatenate(ex)) - rmse(np.concatenate(ey))
    ups, los = [], []
    for sd in RNG_SEEDS:
        rng = np.random.default_rng(sd); acc = np.empty(REPS)
        for r in range(REPS):
            idx = rng.integers(0, len(groups), len(groups))
            acc[r] = (rmse(np.concatenate([ex[i] for i in idx])) -
                      rmse(np.concatenate([ey[i] for i in idx])))
        ups.append(np.percentile(acc, 97.5)); los.append(np.percentile(acc, 2.5))
    return dict(mean=float(point), ci_lo=float(np.mean(los)), ci_hi=float(np.mean(ups)),
                mc_sd_hi=float(np.std(ups, ddof=1)),
                n_engines=int(len(groups)), n_obs=int(len(m)),
                excludes_zero=bool(np.mean(ups) < 0 or np.mean(los) > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/dsh/e6/E6_stats.json')
    a = ap.parse_args()
    out = {'conventions': dict(delta='RMSE(first)-RMSE(second); positive means first is worse',
                               ci='engine-clustered bootstrap, engines within seed, seeds averaged',
                               reps=REPS, rng_seeds=RNG_SEEDS,
                               mc_error='SD of the bound across the RNG seeds')}

    sub = {'est': 'r3_v3', 'est_rate': 'r3_v3', 'pooled': 'r3_v2'}
    r3 = lambda n, s: load_npz(sub.get(n, 'r3_scale'), n, s)
    e3 = lambda n, s: load_csv('e3' if n not in ('BEfix',) else 'e3fix', n, s)
    fd1 = lambda n, s: load_csv('e3_fd001', n, s, 'FD001')
    def combo(dataset):
        base = 'e3' if dataset == 'FD002' else 'e3_fd001'
        def f(n, s):
            if n == 'PS':
                return load_csv('e3c', f'PS_{dataset}', s, dataset)
            return load_csv(base, n, s, dataset)
        return f

    jobs = [
        ('R3.1 est - cycle', 'est', 'cycle', r3),
        ('R3.1 est_rate - cycle', 'est_rate', 'cycle', r3),
        ('R3.1 est - pooled', 'est', 'pooled', r3),
        ('R3.1 oracle - cycle', 'oracle', 'cycle', r3),
        ('E3 PE - A', 'PE', 'A', e3),
        ('E3 BE - A', 'BE', 'A', e3),
        ('E3 BEfix - A', 'BEfix', 'A', e3),
        ('E3b FD001 PE - A', 'PE', 'A', fd1),
        ('E3b FD001 BE - A', 'BE', 'A', fd1),
        ('E3c FD002 PS - PE', 'PS', 'PE', combo('FD002')),
        ('E3c FD002 PS - A', 'PS', 'A', combo('FD002')),
        ('E3c FD002 PS - BE', 'PS', 'BE', combo('FD002')),
        ('E3c FD001 PS - PE', 'PS', 'PE', combo('FD001')),
        ('E3c FD001 PS - A', 'PS', 'A', combo('FD001')),
    ]
    out['contrasts'] = {}
    for label, x, y, load in jobs:
        try:
            r = contrast(None, x, y, load)
            out['contrasts'][label] = r
            print(f'{label:24s} {r["mean"]:+.3f}  [{r["ci_lo"]:+.3f},{r["ci_hi"]:+.3f}]  mc±{r["mc_sd_hi"]:.4f}')
        except FileNotFoundError as e:
            out['contrasts'][label] = {'error': str(e)}
            print(f'{label:24s} MISSING {e}')

    for label, x, y, sub in [('R3E cycle - nocycle (pooled)', 'arm_cycle', 'arm_nocycle', None),
                             ('R3E cycle - nocycle (FD002)', 'arm_cycle', 'arm_nocycle', 'FD002'),
                             ('R3E pe - nocycle (pooled)', 'arm_pe', 'arm_nocycle', None)]:
        r = contrast_r3preds(x, y, sub)
        out['contrasts'][label] = r
        print(f'{label:32s} {r["mean"]:+.3f}  [{r["ci_lo"]:+.3f},{r["ci_hi"]:+.3f}]  n={r["n_engines"]}')

    out['not_recomputable'] = [
        {'claim': 'C2 MDE (1.60 / 1.14 / 0.97)', 'why': 'no per-engine residual artefact for the 260-engine pool; values live only in FINAL_DELIVERABLE prose'},
        {'claim': 'C3 21 paired comparisons, 0 after Bonferroni', 'why': 'no per-engine predictions for the 7 methods; only an 18-row 20-engine FD001 table exists (clustered_bootstrap_vs_ref.csv), which is a different design'},
        {'claim': 'C10 cap sweep 24.8 / 26.9 / 12.06', 'why': 'cutoff_sensitivity.csv holds the sweep but not per-engine rows, so no clustered CI'},
        {'claim': 'C1 variance decomposition 2.6/31.5/65.9', 'why': 'point estimates in c_interaction.json; no interval recorded'},
    ]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    print('\n->', a.out)


if __name__ == '__main__':
    main()
