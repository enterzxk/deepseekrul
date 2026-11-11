"""FD001-only freeze manifest (scope reduction requested 2026-11-11)."""
import json
from pathlib import Path

import pandas as pd

CFG8 = ['base', 'ms', 'sa', 'hi', 'mssa', 'mshi', 'sahi', 'full']
raw = Path('paper_b/results/raw')
rows = []
for tag_dir in raw.glob('A8_*_L70'):
    tag = tag_dir.name
    for f in tag_dir.glob('FD001/s*/metrics.json'):
        m = json.loads(f.read_text())
        rows.append(dict(tag=tag, seed=int(f.parent.name[1:]), val=m['val_engine_rmse']))
df = pd.DataFrame(rows)
means = df.groupby('tag').val.mean().sort_values()
base = means['A8_base_L70']
print(means.round(4).to_string())
final = means.index[0]
if means[final] >= base:
    final = 'A8_base_L70'
print('FINAL:', final, 'mean', round(float(means[final]), 4), 'base', round(float(base), 4))
tags = [final, 'A8_base_L70', 'VAN_raw_L70', 'VAN_dsp_cycle_L70']
runs = []
for t in tags:
    d = df[df.tag == t]
    if d.empty:
        # Vanilla runs are not in the A8 scan
        for f in (raw / t / 'FD001').glob('s*/metrics.json'):
            runs.append(dict(tag=t, subset='FD001', seed=int(f.parent.name[1:])))
        continue
    for _, r in d.iterrows():
        runs.append(dict(tag=t, subset='FD001', seed=int(r.seed)))
man = dict(frozen=True, scope='FD001', created='2026-11-11', final=final,
           rule='lowest mean validation RMSE on FD001; test once for final + base + Vanilla baselines',
           eval_tags=tags, runs=runs)
Path('paper_b/results/FREEZE_FD001.json').write_text(json.dumps(man, indent=1))
print('wrote manifest with', len(runs), 'runs')
