"""Apply the pre-registered model-selection rule and write FREEZE.json.

Rule (fixed before any test number is computed):

1. Eight-config DSPT ablation: a config is ELIGIBLE for the primary model iff
   it beats DSPT-base on at least 3 of the 4 subsets (mean validation RMSE over
   the three seeds) and its equal-weight mean relative validation RMSE across
   subsets is <= 0.
2. Mechanism variants (MS/SA/HI) are SUPPORTED iff they beat their own
   reference on at least 3 of 4 subsets by a mean of at least 1% AND are not
   worse than DSPT-base overall.
3. The final model is the eligible candidate with the lowest equal-weight mean
   relative validation RMSE; ties are broken by fewer parameters (simpler model).
4. If no config is eligible, the final model is DSPT-base.
5. Only the final model, the two reference models (DSPT-base, DSPT-full), the
   two Vanilla baselines and the supported mechanism variants are evaluated on
   the official test set. The ablation table itself stays validation-based.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CFG8 = ['base', 'ms', 'sa', 'hi', 'mssa', 'mshi', 'sahi', 'full']
MECH = {
    'MSM_single_proj_L70': 'A8_base_L70', 'MSM_multi_noproj_L70': 'A8_base_L70',
    'MSM_multi3_L70': 'A8_base_L70',
    'SAM_identity_L70': 'A8_sa_L70', 'SAM_cond_L70': 'A8_sa_L70',
    'HIM_hf_wrong_L70': 'A8_hi_L70', 'HIM_hf_nomono_L70': 'A8_hi_L70',
    'HIM_hB_correct_L70': 'A8_hi_L70',
}


def subset_means(df, tag, metric='val_rmse'):
    d = df[df.tag == tag]
    return d.groupby('subset')[metric].mean()


def rel_stats(df, tag, ref_tag):
    t = subset_means(df, tag); b = subset_means(df, ref_tag)
    common = t.index.intersection(b.index)
    if len(common) == 0:
        return None
    rel = (t.loc[common] / b.loc[common] - 1).values
    return dict(n_subsets=len(common), mean_rel=float(rel.mean()),
                wins=int((rel < 0).sum()), per_subset={s: float(x) for s, x in zip(common, rel)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='paper_b/results')
    ap.add_argument('--out', default='paper_b/results/FREEZE.json')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    df = pd.read_csv(Path(a.results) / 'all_runs.csv')
    rows, eligible = [], []
    base = 'A8_base_L70'
    for cfg in CFG8:
        tag = f'A8_{cfg}_L70'
        st = rel_stats(df, tag, base)
        if st is None:
            continue
        params = float(df[df.tag == tag].params.median())
        ok = (st['wins'] >= 3 and st['mean_rel'] <= 0)
        rows.append(dict(kind='8cfg', tag=tag, ref=base, **st, params=params, eligible=ok))
        if ok:
            eligible.append(dict(tag=tag, score=st['mean_rel'], params=params, kind='8cfg'))
    supported = []
    for tag, ref in MECH.items():
        st = rel_stats(df, tag, ref)
        if st is None:
            continue
        overall = rel_stats(df, tag, base)
        params = float(df[df.tag == tag].params.median())
        ok = (st['wins'] >= 3 and st['mean_rel'] <= -0.01 and overall is not None
              and overall['mean_rel'] <= 0)
        rows.append(dict(kind='mech', tag=tag, ref=ref, **st,
                         overall_vs_base=(overall['mean_rel'] if overall else np.nan),
                         params=params, eligible=ok))
        if ok:
            supported.append(tag)
            eligible.append(dict(tag=tag, score=overall['mean_rel'], params=params, kind='mech'))
    tab = pd.DataFrame(rows).sort_values(['eligible', 'mean_rel'], ascending=[False, True])
    print(tab.to_string(index=False))
    if eligible:
        best = sorted(eligible, key=lambda x: (x['score'], x['params']))[0]
    else:
        best = dict(tag=base, score=0.0, params=float(df[df.tag == base].params.median()), kind='fallback')
    print('\nFINAL:', best)
    print('SUPPORTED:', supported)

    eval_tags = [best['tag']]
    for t in (base, 'A8_full_L70', 'VAN_raw_L70', 'VAN_dsp_cycle_L70') + tuple(supported):
        if t not in eval_tags and t in set(df.tag):
            eval_tags.append(t)
    runs = []
    for t in eval_tags:
        for _, r in df[df.tag == t].iterrows():
            runs.append(dict(tag=t, subset=r['subset'], seed=int(r['seed'])))
    man = dict(frozen=True, created='2026-11-11', rule=__doc__,
               final=best, supported=supported, eval_tags=eval_tags, runs=runs,
               note='Official test is opened once for these tags; the ablation table is validation-based.')
    print('eval tags:', eval_tags, 'runs:', len(runs))
    if a.write:
        Path(a.out).write_text(json.dumps(man, indent=1))
        tab.to_csv(Path(a.results) / 'freeze_selection.csv', index=False)
        print('wrote', a.out)
    else:
        print('dry run; pass --write to create the manifest')


if __name__ == '__main__':
    main()
