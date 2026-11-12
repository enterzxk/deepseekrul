#!/usr/bin/env python3
"""Print the ledger rows for one (rung, fold) so a disputed cell can be checked."""
import csv
import sys

path = sys.argv[1]
rung, fold = sys.argv[2], sys.argv[3]
rows = [r for r in csv.DictReader(open(path, encoding='utf-8'))
        if r['rung'] == rung and r['fold'] == fold]
for r in rows:
    print(f"{r['run_id']:34s} {r['selection_rule']:18s} "
          f"ep={r['epochs_run']:>4s} rmse={r['rmse_raw_unit']:>8s} "
          f"nasa={r['nasa_raw_unit'][:8]:>9s} status={r['status']}")
if not rows:
    print(f'no rows for rung={rung} fold={fold}')
