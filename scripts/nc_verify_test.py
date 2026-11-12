#!/usr/bin/env python3
"""Inspect a test-phase output directory: artifacts, epoch* provenance, ledger rows.

Usage: python scripts/nc_verify_test.py <subset-dir> [<subset-dir> ...]
e.g.   python scripts/nc_verify_test.py /tmp/nc_test_check/DS02
"""
import csv
import json
import sys
from pathlib import Path

KEYS = ('epoch_evaluated', 'selection_rule', 'epoch_star', 'epoch_star_rule',
        'epoch_star_fold_values', 'n_flights', 'rmse_raw_unit', 'mae_raw_unit',
        'bias_raw_unit', 'nasa_raw_unit', 'passes_trained')


def main():
    for d in (Path(a) for a in sys.argv[1:]):
        print(f'=== {d} ===')
        dirs = sorted(p for p in d.iterdir() if p.is_dir())
        print(f'dirs={len(dirs)}: {[p.name for p in dirs][:6]}')
        for p in dirs:
            files = sorted(f.name for f in p.iterdir())
            print(f'  {p.name}: {files}')
        for p in dirs:
            m = p / 'metrics.json'
            if m.exists():
                j = json.loads(m.read_text(encoding='utf-8'))
                print(f'  {p.name}: ' + '  '.join(
                    f'{k}={j.get(k)}' for k in KEYS))
        led = d / 'runs_test.csv'
        if led.exists():
            rows = list(csv.DictReader(open(led, encoding='utf-8')))
            print(f'  ledger rows={len(rows)}')
            for r in rows:
                print(f"    {r['run_id']:24s} phase={r['phase']} fold={r['fold']} "
                      f"n_train={r['n_train_engines']} ep={r['epochs_run']} "
                      f"ckpt={Path(r['ckpt_path']).name if r['ckpt_path'] else '-'} "
                      f"sha={r['ckpt_sha256'][:12]}")


if __name__ == '__main__':
    main()
