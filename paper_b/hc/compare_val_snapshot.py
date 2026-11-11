"""Element-wise proof that the R1 repair in data.py did not move any tensor.

Compares two ``snapshot_val`` dumps (before / after the edit) field by field:
shape, dtype, exact equality, and the maximum absolute difference. A repair
that is truly behaviour preserving must give ``equal == True`` for every field
of every subset, with ``max_abs_diff == 0.0``.

Usage::

    python -m paper_b.hc.compare_val_snapshot \
        --before experiments/hc/tmp/val_before.npz \
        --after  experiments/hc/tmp/val_after.npz
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha_file(path):
    return hashlib.sha256(np.fromfile(path, dtype=np.uint8)).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True)
    ap.add_argument('--after', required=True)
    ap.add_argument('--out', default=None, help='optional JSON report path')
    a = ap.parse_args()

    b = np.load(a.before)
    c = np.load(a.after)
    keys_b, keys_c = sorted(b.files), sorted(c.files)
    report = dict(before=a.before, after=a.after,
                  before_npz_sha256=sha_file(a.before),
                  after_npz_sha256=sha_file(a.after),
                  keys_equal=keys_b == keys_c, fields={}, all_equal=True,
                  n_fields=len(keys_b))
    if keys_b != keys_c:
        report['missing_after'] = [k for k in keys_b if k not in keys_c]
        report['extra_after'] = [k for k in keys_c if k not in keys_b]
        report['all_equal'] = False

    for k in keys_b:
        if k not in keys_c:
            continue
        x, y = b[k], c[k]
        same_shape = x.shape == y.shape
        same_dtype = x.dtype == y.dtype
        if same_shape:
            eq = bool(np.array_equal(x, y))
            diff = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) \
                if x.size else 0.0
        else:
            eq, diff = False, float('inf')
        report['fields'][k] = dict(shape_before=list(x.shape), shape_after=list(y.shape),
                                   dtype=str(x.dtype), equal=eq,
                                   max_abs_diff=diff)
        report['all_equal'] &= eq and same_shape and same_dtype

    print(f'{"field":28s} {"shape":>14s} {"equal":>6s} {"max|diff|":>10s}')
    for k, v in report['fields'].items():
        print(f'{k:28s} {str(tuple(v["shape_before"])):>14s} '
              f'{str(v["equal"]):>6s} {v["max_abs_diff"]:>10.3e}')
    print(f'\nALL EQUAL: {report["all_equal"]}   fields={report["n_fields"]}')
    print(f'npz sha256 before {report["before_npz_sha256"]}')
    print(f'npz sha256 after  {report["after_npz_sha256"]}')

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
        print('wrote', a.out)
    raise SystemExit(0 if report['all_equal'] else 1)


if __name__ == '__main__':
    main()
