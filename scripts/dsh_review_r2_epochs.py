import csv
rows = list(csv.DictReader(open('results/dsh/review_r2/A1_runs.csv')))
missing = [r for r in rows if r['epochs_ran'] == 'unknown']
print('runs with NO epochs_ran field:', len(missing))
from collections import Counter
print('  by root:', dict(Counter(r['root'] for r in missing)))
print('  sample:', missing[0]['run_id'] if missing else '-')
known = [(r, int(r['epochs_ran'])) for r in rows if r['epochs_ran'] != 'unknown']
over30 = [(r,e) for r,e in known if e > 30]
print()
print('runs with epochs_ran > 30:', len(over30))
print('  by root:', dict(Counter(r['root'] for r,_ in over30)))
print('  epochs values:', sorted(e for _,e in over30))
tot = sum(e for _,e in known)
print(f'  total epochs over known runs: {tot}  (= {tot/30:.1f} x 30-epoch units)')
print()
per_root = {}
for r,e in known: per_root.setdefault(r['root'], []).append(e)
print('known-run epoch totals by root:')
for k,v in sorted(per_root.items()): print(f'  {k:20s} n={len(v):3d} epochs_sum={sum(v):4d} -> {sum(v)/30:5.1f} units')
