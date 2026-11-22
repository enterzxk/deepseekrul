"""Paper C Phase 0 hard assertions: zero test-file references + fold isolation."""
import glob
import json
import re
from pathlib import Path

from paper_c.data import add_rul, load_train, split_units

out = {}
# A0.1a: no test/RUL file reads in paper_c
bad = []
for f in glob.glob('paper_c/*.py'):
    if Path(f).name in ('audit.py', 'repro_check.py'):
        continue
    txt = Path(f).read_text(encoding='utf-8')
    for pat in ['test_FD', 'RUL_FD', 'test_FD00', 'RUL_FD00']:
        if pat in txt:
            bad.append((f, pat))
out['test_file_refs'] = bad
out['A0_1a_pass'] = len(bad) == 0

# A0.1b: folds are engine-disjoint
df = add_rul(load_train('FD001'))
overlaps = []
for fold in range(5):
    tr, va = split_units(df.unit.unique(), fold)
    if tr & va:
        overlaps.append((fold, len(tr & va)))
out['fold_overlaps'] = overlaps
out['A0_1b_pass'] = len(overlaps) == 0

# A0.1c: train file exists, test file not required
out['train_units'] = int(df.unit.nunique())
Path('results/paper_c').mkdir(parents=True, exist_ok=True)
Path('results/paper_c/p0_audit.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
