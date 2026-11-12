"""Build the DSPT-Base vs Vanilla 2x2 comparison table (validation, engine level).

Inputs (read-only):
  paper_b/results/vanilla_engine_level.csv  -- recomputed from val_pred.csv
  paper_b/results/a8_engine_level.csv       -- A8 96-cell engine-level table
Outputs:
  paper_b/results/vanilla_vs_base.csv
  stdout markdown table
"""
import numpy as np
import pandas as pd

ARMS = {
    'DSPT-Base (A8_base_L70)': ('a8', 'A8_base_L70'),
    'Vanilla raw': ('van', 'VAN_raw_L70'),
    'Vanilla +cycle': ('van', 'VAN_raw_cycle_L70'),
    'Vanilla +3ch': ('van', 'VAN_dsp_L70'),
    'Vanilla +3ch+cycle': ('van', 'VAN_dsp_cycle_L70'),
}
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']


def load():
    a8 = pd.read_csv('paper_b/results/a8_engine_level.csv')
    a8['seed'] = a8['seed'].str.replace('s', '', regex=False).astype(int)
    van = pd.read_csv('paper_b/results/vanilla_engine_level.csv')
    out = []
    for name, (src, tag) in ARMS.items():
        d = a8 if src == 'a8' else van
        d = d[d.tag == tag]
        for _, r in d.iterrows():
            out.append(dict(arm=name, subset=r.subset, seed=int(r.seed),
                            val_eng=float(r.val_eng)))
    return pd.DataFrame(out)


def main():
    d = load()
    d.to_csv('paper_b/results/vanilla_vs_base.csv', index=False)
    print('cells per arm:')
    print(d.groupby('arm').size().to_string())
    print()
    hdr = '| arm | ' + ' | '.join(SUBSETS) + ' | mean |'
    print(hdr)
    print('|---|' + '---|' * (len(SUBSETS) + 1))
    for arm in ARMS:
        cells = []
        for s in SUBSETS:
            v = d[(d.arm == arm) & (d.subset == s)].val_eng
            cells.append(f'{v.mean():.3f} ± {v.std(ddof=1):.3f}' if len(v) > 1 else 'NA')
        m = d[d.arm == arm].val_eng.mean()
        print(f'| {arm} | ' + ' | '.join(cells) + f' | {m:.3f} |')
    print()
    base = d[d.arm == 'DSPT-Base (A8_base_L70)'].set_index(['subset', 'seed']).val_eng
    print('| arm | ' + ' | '.join(SUBSETS) + ' |')
    print('|---|' + '---|' * len(SUBSETS))
    for arm in list(ARMS)[1:]:
        cells = []
        for s in SUBSETS:
            v = d[(d.arm == arm) & (d.subset == s)].set_index('seed').val_eng
            delta = (v - base.loc[s]).mean()
            cells.append(f'{delta:+.3f}')
        print(f'| {arm} − Base | ' + ' | '.join(cells) + ' |')
    print('\n(negative = Vanilla better than DSPT-Base)')


if __name__ == '__main__':
    main()
