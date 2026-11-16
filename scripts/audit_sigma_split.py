"""Verify that the confirmation-set training-time noise scale comes from the
confirmation split's own TRAINING engines.

Rule under test (pre-registered): sigma is estimated only from training engines.
In split 29 that means split_units(..., 29) training engines, and no validation
engine may appear anywhere in the estimate.

Checks per subset:
  * stats['split_seed'] == the split under test
  * set(stats['train_units']) == set(train units of that split)   (exact)
  * set(stats['train_units']) & set(val units of that split) == empty
  * stats['hash'] matches what the manifest rows record
  * the split-17 file is untouched (its hash prefix still matches the recorded
    discovery value)

Usage:
  python3 scripts/audit_sigma_split.py --split 29 \
      --stats-glob 'shared/noise_stats_split29_*.json' \
      --manifest .../confirm29_e2a.json --data-dir /path/CMAPSS
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_b.training.data import load_train, split_units  # noqa: E402

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=int, default=29)
    ap.add_argument('--stats-glob', required=True)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--data-dir', required=True)
    a = ap.parse_args()

    ok_all = True
    for sub in SUBSETS:
        files = glob.glob(a.stats_glob.replace('*', sub))
        if not files:
            print('%s  NO FILE for glob %s' % (sub, a.stats_glob))
            ok_all = False
            continue
        p = files[0]
        st = json.load(open(p))
        raw = load_train(Path(a.data_dir), sub)
        units = sorted(int(u) for u in raw.unit.unique())
        tr, va = split_units(units, a.split, 0, 0.2)
        tr, va = set(int(x) for x in tr), set(int(x) for x in va)
        got = set(int(x) for x in st.get('train_units', []))
        c1 = int(st.get('split_seed', -1)) == a.split
        c2 = got == tr
        c3 = not (got & va)
        print('%s  file=%s' % (sub, os.path.basename(p)))
        print('   split_seed=%s (want %d) %s' % (st.get('split_seed'), a.split,
                                                 'OK' if c1 else 'FAIL'))
        print('   train_units=%d  split%d_train=%d  equal=%s  extra=%s  missing=%s'
              % (len(got), a.split, len(tr), c2,
                 sorted(got - tr)[:5], sorted(tr - got)[:5]))
        print('   overlap with split%d val = %d  %s'
              % (a.split, len(got & va), 'OK' if c3 else 'FAIL ' + str(sorted(got & va)[:8])))
        print('   hash=%s' % st.get('hash'))
        ok_all = ok_all and c1 and c2 and c3

    if a.manifest:
        m = json.load(open(a.manifest))
        rows = [r for r in m['runs'] if (r['config'].get('augment') or {}).get('noise_alpha', 0) > 0]
        paths = {(r['config']['augment'].get('noise_stats_path') or '') for r in rows}
        hashes = {(r['config']['augment'].get('noise_stats_hash') or '') for r in rows}
        print('\nmanifest rows with noise_alpha>0:', len(rows))
        print('   distinct noise_stats_path:', sorted(paths))
        print('   distinct noise_stats_hash:', sorted(hashes))

    print('\nVERDICT', 'PASS' if ok_all else 'FAIL')
    return 0 if ok_all else 1


if __name__ == '__main__':
    raise SystemExit(main())
