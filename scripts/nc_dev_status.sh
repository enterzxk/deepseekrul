#!/usr/bin/env bash
# Progress of the full 60-cell x 2-evaluation-point dev run.
set -u
cd "$HOME/deepseek-rul" || exit 1
L=experiments/ncmapss/runs_dev.csv
echo "procs=$(pgrep -fc 'paper_b.ncmapss.run_loeo' || true)"
[ -f "$L" ] || { echo "ledger not created"; exit 0; }
echo "rows=$(($(wc -l < "$L") - 1))  expected=120"
echo "last: $(tail -1 "$L" | cut -d, -f1)"
python3 - "$L" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
by_rung = collections.Counter(r['rung'] for r in rows)
print('by rung:', dict(sorted(by_rung.items())))
cells = collections.defaultdict(dict)
for r in rows:
    cells[(r['rung'], r['fold'], r['seed'])][r['selection_rule']] = r['rmse_raw_unit']
both = sum(1 for c in cells.values() if len(c) == 2)
print(f'cells={len(cells)}  with both rules={both}')
for (rung, fold, seed), c in sorted(cells.items())[-4:]:
    print(f'  {rung} {fold} s{seed}: '
          + '  '.join(f'{k}={float(v):.2f}' for k, v in sorted(c.items())))
PY
tail -2 logs/nc_dev.log 2>&1
