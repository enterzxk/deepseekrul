"""CPU-only mechanism probes for the Paper B negative ablations.

1) HI target diagnostics: is the PCA-1 health indicator a stable monotone
   function of life, or a noisy re-parameterisation of RUL?
2) Sensor-correlation heterogeneity across operating conditions: does a single
   condition-agnostic cross-sensor attention layer have a well-posed target?
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_b.training.data import (prepare_train, SENSOR_COLS, OP_COLS, fit_hi,
                                   _prepare_split)


def hi_diagnostics(subset, data_dir, split_seed=7, fold=0):
    s = _prepare_split(subset, data_dir, split_seed, fold, 0.2, 6)
    hi = fit_hi(s['train_units'], s['tr'], s['tr_s'], split_seed)
    d = s['tr'].copy()
    d['hi'] = [hi.get((int(u), int(c)), np.nan) for u, c in zip(d.unit, d.cycle)]
    d['rul'] = d.groupby('unit')['cycle'].transform('max') - d['cycle']
    rows = []
    for uid, g in d.groupby('unit'):
        if len(g) < 10:
            continue
        rho = pd.Series(g['hi'].values).corr(pd.Series(g['cycle'].values), method='spearman')
        viol = float(np.mean(np.diff(g['hi'].values) < -1e-6))
        # linear fit hi ~ rul, report R^2
        x = g['rul'].values.astype(float); y = g['hi'].values.astype(float)
        if np.std(x) > 1e-9 and np.std(y) > 1e-9:
            c = np.corrcoef(x, y)[0, 1]
            r2 = c * c
        else:
            r2 = np.nan
        rows.append(dict(unit=int(uid), n=len(g), spearman_hi_cycle=rho,
                         violation_rate=viol, r2_hi_rul=r2))
    r = pd.DataFrame(rows)
    return dict(subset=subset, n_engines=int(len(r)),
                spearman_mean=float(r.spearman_hi_cycle.mean()),
                spearman_min=float(r.spearman_hi_cycle.min()),
                frac_spearman_gt_09=float((r.spearman_hi_cycle > 0.9).mean()),
                violation_mean=float(r.violation_rate.mean()),
                violation_p90=float(r.violation_rate.quantile(0.9)),
                r2_hi_rul_mean=float(r.r2_hi_rul.mean()),
                r2_hi_rul_min=float(r.r2_hi_rul.min()))


def sensor_heterogeneity(subset, data_dir, split_seed=7, fold=0, min_rows=200):
    s = _prepare_split(subset, data_dir, split_seed, fold, 0.2, 6)
    Z = s['tr_s']; lab = s['lt']
    mats = []
    for c in range(6):
        m = lab == c
        if m.sum() < min_rows:
            continue
        mats.append(np.corrcoef(Z[m].T))
    if len(mats) < 2:
        return dict(subset=subset, n_clusters=len(mats), hetero=np.nan, within=np.nan)
    diffs = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            iu = np.triu_indices(mats[i].shape[0], k=1)
            diffs.append(np.mean(np.abs(mats[i][iu] - mats[j][iu])))
    iu = np.triu_indices(mats[0].shape[0], k=1)
    within = float(np.mean([np.mean(np.abs(m[iu])) for m in mats]))
    return dict(subset=subset, n_clusters=len(mats), hetero=float(np.mean(diffs)),
                within=within, hetero_over_within=float(np.mean(diffs) / (within + 1e-9)))


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/CMAPSS')
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('paper_b/results/mechanism_probe.json')
    res = {'hi': [], 'sensor': []}
    for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
        res['hi'].append(hi_diagnostics(subset, data_dir))
        res['sensor'].append(sensor_heterogeneity(subset, data_dir))
        print(subset, res['hi'][-1], res['sensor'][-1], flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
