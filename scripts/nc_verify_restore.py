#!/usr/bin/env python3
"""Verify the regenerated round-1 cells against the sealed tables.

Acceptance is metrics AND ckpt_sha256, both against the sealed row. A metric-only
check would pass the N0 cells whose checkpoint was silently replaced, because N0 is
insensitive to all four fixes.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path.home() / 'deepseek-rul'
REG = ROOT / 'docs' / 'paper_b_ncmapss' / 'registry'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = Path('/tmp/r1out')
    for sub in ('DS02', 'DS06'):
        sealed = {r['run_id']: r for r in csv.DictReader(
            open(REG / f'runs_{sub}.csv', encoding='utf-8'))}
        new = {r['run_id']: r for r in csv.DictReader(
            open(out / f'{sub}.csv', encoding='utf-8'))}
        print(f'=== {sub}: regenerated {len(new)} rows ===')
        for rid, r in sorted(new.items()):
            s = sealed.get(rid)
            if s is None:
                print(f'  {rid}: not in the sealed table')
                continue
            ck = Path(r['ckpt_path'])
            got = sha(ck) if ck.exists() else 'MISSING'
            want = s.get('ckpt_sha256', '')
            ok_ck = (got == want)
            ok_m = abs(float(r['rmse_raw_unit']) - float(s['rmse_raw_unit'])) < 1e-9
            flag = 'OK' if (ok_ck and ok_m) else 'MISMATCH'
            print(f'  {rid:34s} metric={ok_m} ckpt={ok_ck}  {flag}')
            if not ok_ck:
                print(f'      sealed {want[:16]}  regenerated {got[:16]}')


if __name__ == '__main__':
    main()
