"""R3.1 closing analysis: the corrected estimator against a capacity-matched control.

Reads the finished 30-epoch runs (r3_scale for cycle/none/oracle, r3_v2 for
est/est_rate/pooled), checks that the engine-level ground truth matches inside a
seed, and reports paired engine-clustered bootstrap intervals.

No training happens here; this script is read-only over existing artifacts.
"""
import argparse, glob, json, os
import numpy as np

SEEDS = [2027, 2028, 2029]
ARMS = ['cycle', 'none', 'oracle', 'pooled', 'est', 'est_rate', 'est_leaky', 'est_rate_leaky']
ROOT = '/home/yndx/zxk/bearing-rul-pi/experiments'
# arm -> experiment directory. est/est_rate now point at r3_v3 (prefix-only val
# estimator); the *_leaky arms keep the superseded r3_v2 runs for comparison.
WHERE = {'cycle': 'r3_scale', 'none': 'r3_scale', 'oracle': 'r3_scale', 'pooled': 'r3_v2',
         'est': 'r3_v3', 'est_rate': 'r3_v3', 'est_leaky': 'r3_v2', 'est_rate_leaky': 'r3_v2'}


def load(arm, seed):
    sub = WHERE[arm]
    stem = arm.replace('_leaky', '')
    p = f'{ROOT}/{sub}/{stem}_s{seed}/pred.npz'
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = np.load(p, allow_pickle=True)
    return dict(y=d['y_true'].astype(float), p=d['y_pred'].astype(float),
                life=d['life'].astype(float), path=p)


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=4000)
    ap.add_argument('--out', default='results/dsh/review_r3/R3_1_final.json')
    a = ap.parse_args()
    rng = np.random.default_rng(20261029)

    data = {arm: {s: load(arm, s) for s in SEEDS} for arm in ARMS}
    # integrity: identical ground truth inside a seed across arms
    for s in SEEDS:
        ref = data['cycle'][s]['y']
        for arm in ARMS:
            assert np.allclose(ref, data[arm][s]['y']), f'y_true mismatch {arm} s{s}'
            assert np.allclose(data['cycle'][s]['life'], data[arm][s]['life']), f'life mismatch {arm} s{s}'

    out = {'seeds': SEEDS, 'n_val': {str(s): int(len(data['cycle'][s]['y'])) for s in SEEDS},
           'runs': {}, 'arms': {}, 'paired': {}, 'bands': {}, 'estimator': {}}

    for arm in ARMS:
        per_seed = {}
        for s in SEEDS:
            d = data[arm][s]
            per_seed[str(s)] = dict(rmse=rmse(d['p'] - d['y']), path=d['path'].replace(ROOT + '/', ''))
        vals = [v['rmse'] for v in per_seed.values()]
        out['arms'][arm] = dict(per_seed=per_seed, mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)))

    def paired(arm_a, arm_b):
        """Positive = arm_a better (lower RMSE). Bootstrap resamples engines within a seed."""
        per = {}
        for s in SEEDS:
            ea = data[arm_a][s]['p'] - data[arm_a][s]['y']
            eb = data[arm_b][s]['p'] - data[arm_b][s]['y']
            per[str(s)] = rmse(eb) - rmse(ea)
        n = len(data[arm_a][SEEDS[0]]['y'])
        reps = np.empty(a.reps)
        for r in range(a.reps):
            acc = []
            for s in SEEDS:
                idx = rng.integers(0, n, n)
                ea = data[arm_a][s]['p'][idx] - data[arm_a][s]['y'][idx]
                eb = data[arm_b][s]['p'][idx] - data[arm_b][s]['y'][idx]
                acc.append(rmse(eb) - rmse(ea))
            reps[r] = np.mean(acc)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        return dict(per_seed=per, mean=float(np.mean(list(per.values()))),
                    ci95=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0))

    for a_, b_ in [('est', 'cycle'), ('est_rate', 'cycle'), ('est', 'pooled'),
                   ('est_rate', 'pooled'), ('est_rate', 'est'), ('est', 'est_leaky'),
                   ('pooled', 'cycle'), ('oracle', 'cycle'), ('none', 'cycle')]:
        out['paired'][f'{a_}_vs_{b_}'] = paired(a_, b_)

    # lifespan bands, pooled over seeds (tertiles recomputed per seed)
    for arm in ARMS:
        band_err ={'short': [], 'mid': [], 'long': []}
        band_life = {'short': [], 'mid': [], 'long': []}
        for s in SEEDS:
            d = data[arm][s]
            q1, q2 = np.quantile(d['life'], [1 / 3, 2 / 3])
            for name, m in [('short', d['life'] <= q1), ('mid', (d['life'] > q1) & (d['life'] <= q2)),
                            ('long', d['life'] > q2)]:
                band_err[name].append((d['p'][m] - d['y'][m]).tolist())
                band_life[name].append(d['life'][m].tolist())
        out['bands'][arm] = {}
        for name in band_err:
            e = np.concatenate(band_err[name])
            out['bands'][arm][name] = dict(n=int(len(e)), rmse=rmse(e), bias=float(e.mean()))

    # estimator quality (which windows are estimable at all)
    for arm in ['est', 'est_rate', 'est_leaky', 'pooled']:
        ratios, cors = [], []
        for s in SEEDS:
            th = np.load(f"{ROOT}/{WHERE[arm]}/{arm.replace('_leaky', '')}_s{s}/pred.npz",
                         allow_pickle=True)['that'].astype(float)
            life = data[arm][s]['life']
            ratios.append(th / life)
            cors.append(float(np.corrcoef(th, life)[0, 1]))
        r = np.concatenate(ratios)
        out['estimator'][arm] = dict(median_ratio=float(np.median(r)),
                                     iqr=[float(np.percentile(r, 25)), float(np.percentile(r, 75))],
                                     corr_per_seed=cors, corr_mean=float(np.mean(cors)))

    json.dump(out, open(a.out, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in ['arms', 'paired']}, indent=2)[:6000])
    print('\nbands:')
    for arm in ARMS:
        print(arm, {k: (round(v['rmse'], 2), round(v['bias'], 1)) for k, v in out['bands'][arm].items()})
    print('\nestimator:', json.dumps(out['estimator'], indent=2))


if __name__ == '__main__':
    main()
