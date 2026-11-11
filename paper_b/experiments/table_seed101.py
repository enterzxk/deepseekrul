"""Single-seed (101) ablation table: validation and test, all four subsets."""
import json
from pathlib import Path
import pandas as pd

CFG = ['base','ms','sa','hi','mssa','mshi','sahi','full']
SUBS = ['FD001','FD002','FD003','FD004']
raw = Path('paper_b/results/raw')
val, test = {}, {}
for c in CFG:
    for s in SUBS:
        d = raw / f'A8_{c}_L70' / s / 's101'
        val[(c, s)] = json.loads((d/'metrics.json').read_text())['val_engine_rmse'] if (d/'metrics.json').exists() else float('nan')
        test[(c, s)] = json.loads((d/'test_metrics.json').read_text())['rmse'] if (d/'test_metrics.json').exists() else float('nan')

def table(d, title):
    rows = []
    for c in CFG:
        r = {'config': c}
        for s in SUBS:
            r[s] = d[(c, s)]
            r[s+'_rel'] = 100*(d[(c, s)]/d[('base', s)]-1) if d[('base', s)] == d[('base', s)] else float('nan')
        rows.append(r)
    df = pd.DataFrame(rows)
    print('\n##', title)
    print(df[['config']+SUBS].round(3).to_string(index=False))
    print('\nrelative to base (%):')
    print(df[['config']+[s+'_rel' for s in SUBS]].round(2).to_string(index=False))
    return df

v = table(val, 'Validation RMSE (seed 101)')
t = table(test, 'Official test RMSE (seed 101)')
v.to_csv('paper_b/results/seed101_ablation_val.csv', index=False)
t.to_csv('paper_b/results/seed101_ablation_test.csv', index=False)
