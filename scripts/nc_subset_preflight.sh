#!/usr/bin/env bash
# Pre-launch evidence for the four-subset grid: contract test, a synthetic
# end-to-end run, and the split read from each real HDF5.
set -u
cd "$HOME/deepseek-rul" || exit 1
PY=/home/user/miniconda3/envs/ncmapss/bin/python

echo "=== 1. contract test ==="
"$PY" -m paper_b.ncmapss.test_contract 2>&1 | tail -5

echo
echo "=== 2. official split read from each real file ==="
"$PY" - <<'PY'
import glob
from paper_b.ncmapss.data import official_split
for s in ('DS02', 'DS05', 'DS06', 'DS07'):
    p = glob.glob(f'data/NCMAPSS/extracted/N-CMAPSS_{s}*.h5')[0]
    dev, test = official_split(p)
    print(f'{s}: dev={list(dev)}  test={list(test)}  '
          f'n_dev={len(dev)} n_test={len(test)}  file={p.split("/")[-1]}')
PY

echo
echo "=== 3. synthetic end-to-end (official_split + load_subset + --subset + audit) ==="
rm -rf /tmp/nc_sub_smoke
"$PY" -m paper_b.ncmapss.run_loeo --synthetic --folds 1 \
      --subset SMOKE --out-root /tmp/nc_sub_smoke 2>&1 | tail -12

echo
echo "=== 4. artifacts produced ==="
find /tmp/nc_sub_smoke -maxdepth 3 \( -name 'audit.json' -o -name 'runs_dev.csv' \) | sort
echo "--- audit.json (head) ---"
head -32 /tmp/nc_sub_smoke/SMOKE/audit.json 2>&1
echo "--- ledger ---"
"$PY" - <<'PY'
import csv, pathlib
p = pathlib.Path('/tmp/nc_sub_smoke/SMOKE/runs_dev.csv')
rows = list(csv.DictReader(open(p)))
print('rows', len(rows))
print('run_ids', [r['run_id'] for r in rows][:4])
print('rules', sorted({r['selection_rule'] for r in rows}))
print('phase', sorted({r['phase'] for r in rows}))
print('held_out', sorted({r['held_out_engine'] for r in rows}))
PY
