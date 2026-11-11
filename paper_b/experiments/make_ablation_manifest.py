"""Manifest for the user-requested descriptive test evaluation of the full ablation."""
import json
from pathlib import Path
CFG = ['base','ms','sa','hi','mssa','mshi','sahi','full']
runs = [dict(tag=f'A8_{c}_L70', subset=s, seed=101)
        for c in CFG for s in ('FD001','FD002','FD003','FD004')]
man = dict(frozen=True, scope='ablation-seed101', created='2026-11-11',
           note='User-requested descriptive test evaluation of all 8 configs on all 4 subsets, seed 101. Not used for model selection; final model remains MS+SA.',
           runs=runs)
Path('paper_b/results/FREEZE_ABLATION_SEED101.json').write_text(json.dumps(man, indent=1))
print('wrote', len(runs), 'runs')
