"""E3 contrasts: A vs BE vs PE on the frozen pipeline, engine-clustered bootstrap.

Reads <out>/<arm>_s<seed>/FD002/<sub>/metrics.json and matched_cutoff_predictions.csv,
reports per-seed metrics, paired deltas and a one-sided 97.5% upper bound for the
pre-registered non-inferiority rule (paper/E3_PREREGISTRATION.md §5).
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

ARMS = ['A', 'BE', 'PE']
SEEDS = [2027, 2028, 2029]


def load(root, arm, seed, dataset='FD002'):
    g = sorted(glob.glob(f'{root}/{arm}_s{seed}/{dataset}/*/metrics.json'))
    if not g:
        raise FileNotFoundError(f'{arm} s{seed}')
    m = json.load(open(g[0]))
    p = os.path.join(os.path.dirname(g[0]), 'matched_cutoff_predictions.csv')
    pred = pd.read_csv(p)
    return m, pred


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='experiments/e3')
    ap.add_argument('--out', default='results/dsh/e3/E3_contrasts.json')
    ap.add_argument('--reps', type=int, default=4000)
    ap.add_argument('--dataset', default='FD002')
    a = ap.parse_args()
    rng = np.random.default_rng(20261029)

    data, rows = {}, []
    for arm in ARMS:
        for s in SEEDS:
            m, p = load(a.root, arm, s, a.dataset)
            data[(arm, s)] = (m, p)
            e = p['y_pred'].to_numpy(float) - p['y_true'].to_numpy(float)
            nz = np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)
            rows.append(dict(arm=arm, seed=s, rmse=rmse(e), mae=float(np.abs(e).mean()),
                             nasa=float(nz.mean()), late=float((e > 0).mean()),
                             severe=float((e > 10).mean()), params=m['params'],
                             best_epoch=m.get('best_epoch'), epochs_ran=m.get('epochs_ran'),
                             val=m['best_val_rmse'], pseudo=m.get('pseudo_test_rmse_mean')))
    t = pd.DataFrame(rows)
    summary = t.groupby('arm').agg(['mean', 'std']).round(4)
    print(t.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print()
    print(summary.to_string())

    # paired, engine-clustered
    contrasts = {}
    pairs = [('PE', 'A'), ('BE', 'A'), ('PE', 'BE')]
    for x, y in pairs:
        per_seed, reps = {}, []
        for s in SEEDS:
            px = data[(x, s)][1].set_index('unit_id').sort_index()
            py = data[(y, s)][1].set_index('unit_id').sort_index()
            common = px.index.intersection(py.index)
            ex = (px.loc[common, 'y_pred'] - px.loc[common, 'y_true']).to_numpy(float)
            ey = (py.loc[common, 'y_pred'] - py.loc[common, 'y_true']).to_numpy(float)
            per_seed[str(s)] = rmse(ex) - rmse(ey)
        n = len(common)
        for _ in range(a.reps):
            acc = []
            for s in SEEDS:
                px = data[(x, s)][1].set_index('unit_id').sort_index()
                py = data[(y, s)][1].set_index('unit_id').sort_index()
                common = px.index.intersection(py.index)
                idx = rng.integers(0, len(common), len(common))
                ex = (px.loc[common, 'y_pred'] - px.loc[common, 'y_true']).to_numpy(float)[idx]
                ey = (py.loc[common, 'y_pred'] - py.loc[common, 'y_true']).to_numpy(float)[idx]
                acc.append(rmse(ex) - rmse(ey))
            reps.append(np.mean(acc))
        reps = np.array(reps)
        contrasts[f'{x}_minus_{y}'] = dict(
            per_seed=per_seed, mean=float(np.mean(list(per_seed.values()))),
            ci95=[float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
            upper975=float(np.percentile(reps, 97.5)),
            excludes_zero=bool(np.percentile(reps, 2.5) > 0 or np.percentile(reps, 97.5) < 0),
            better=bool(np.mean(list(per_seed.values())) < 0))
    print()
    for k, v in contrasts.items():
        print(f'{k:12s} mean {v["mean"]:+.3f}  CI [{v["ci95"][0]:+.3f},{v["ci95"][1]:+.3f}]  excl0={v["excludes_zero"]}')

    # pre-registered rule for PE
    a_arm = t[t.arm == 'A']; p_arm = t[t.arm == 'PE']
    d = contrasts['PE_minus_A']
    rule = dict(
        delta_upper=d['upper975'],
        rmse_non_inferior=bool(d['upper975'] < 0.30),
        nasa_ok=bool(p_arm['nasa'].mean() <= 1.05 * a_arm['nasa'].mean()),
        late_ok=bool(p_arm['late'].mean() <= a_arm['late'].mean() + 0.02),
    )
    rule['non_inferior'] = bool(rule['rmse_non_inferior'] and rule['nasa_ok'] and rule['late_ok'])
    rule['pe_better'] = bool(d['ci95'][1] < 0)
    print('\nRULE:', json.dumps(rule, indent=2))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(per_run=t.to_dict('records'), contrasts=contrasts, rule=rule), open(a.out, 'w'), indent=2)
    t.to_csv(a.out.replace('.json', '.csv'), index=False)
    print('\n->', a.out)


if __name__ == '__main__':
    main()
