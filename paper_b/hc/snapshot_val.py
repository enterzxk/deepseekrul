"""Regression snapshot of the frozen validation tensors.

Purpose: prove that the R1 repair in ``paper_b/training/data.py``
(``prepare_train(cuts=None)`` -> explicit ``ValueError``) is behaviour
preserving. Run once before the edit and once after; the two ``.npz`` files
must be byte-identical after decompression, and every array is also hashed
individually so a mismatch names the offending field.

Also prints the R6 split fingerprint
``md5(",".join(str(u) for u in sorted(val_ids)))[:16]`` so the split can be
checked against the administrator's reference values.

Usage::

    python -m paper_b.hc.snapshot_val --out experiments/hc/tmp/val_before.npz
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from .config_hc import SUBSETS, VAL_CUTS

FIELDS = ['x', 'y', 'mask', 'baseline', 'cycle', 'cond', 'unit', 'cut']


def _load_data_module(path):
    """C7: import a *specific* data.py file so the pre-edit state stays runnable.

    The repair already lives in the working tree, so "before" can only be
    reconstructed from history::

        git show HEAD:paper_b/training/data.py > /tmp/data_head.py
        python -m paper_b.hc.snapshot_val --out before.npz --data-py /tmp/data_head.py
    """
    spec = importlib.util.spec_from_file_location('data_head', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['data_head'] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-dir', default='data/CMAPSS')
    ap.add_argument('--span', type=int, default=70)
    ap.add_argument('--data-py', default=None,
                    help='import data.py from this path (C7: use git HEAD for "before")')
    a = ap.parse_args()

    if a.data_py:
        mod = _load_data_module(a.data_py)
        prepare_train = mod.prepare_train
        print(f'[snapshot] using data module from {a.data_py} '
              f'(md5 {hashlib.md5(Path(a.data_py).read_bytes()).hexdigest()})')
    else:
        from ..training.data import prepare_train
        print('[snapshot] using the in-tree paper_b.training.data')

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    store, report = {}, {'cuts': list(VAL_CUTS), 'span': a.span,
                         'data_py': a.data_py, 'subsets': {}}
    for subset in SUBSETS:
        ds = prepare_train(subset, a.data_dir, span=a.span, cuts=tuple(VAL_CUTS))
        va = ds['val']
        entry = {}
        for f in FIELDS:
            arr = np.asarray(va[f])
            store[f'{subset}/{f}'] = arr
            entry[f] = dict(shape=list(arr.shape), dtype=str(arr.dtype), sha256=sha(arr))
        ids = sorted(int(u) for u in ds['val_units'])
        entry['val_ids'] = ids
        entry['val_engine_ids_hash'] = hashlib.md5(
            ','.join(str(u) for u in ids).encode()).hexdigest()[:16]
        entry['train_ids'] = sorted(int(u) for u in ds['train_units'])
        entry['overlap'] = sorted(set(entry['val_ids']) & set(entry['train_ids']))
        report['subsets'][subset] = entry
        print(f'{subset}: val={len(ids)} train={len(entry["train_ids"])} '
              f'hash={entry["val_engine_ids_hash"]} n_windows={len(va["y"])}')

    np.savez_compressed(out, **store)
    rp = out.with_suffix('.json')
    rp.write_text(json.dumps(report, indent=1))
    print('wrote', out, 'and', rp)
    print('npz sha256', sha(np.fromfile(out, dtype=np.uint8)))


if __name__ == '__main__':
    main()
