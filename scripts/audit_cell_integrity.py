#!/usr/bin/env python3
"""Check that every cell is one training + one evaluation, in that order.

Two facts about this runner version make "COMPLETE + all artifacts present"
insufficient:

1. `status='COMPLETE'` is written by `train.py:361`, i.e. when **training**
   ends. `evaluate_run` runs after that. So there is a window in which a cell
   is COMPLETE while its robustness artifacts are still from a previous model.
2. A cell retried through the eval-only path never reaches COMPLETE at all
   (the worker's success branch does not write status.json), so it is excluded
   here by the status check rather than silently accepted.

So the artifact timestamps must be ordered:

    best.pt  <=  robustness_val.json  <=  val_noise_predictions.csv
             <=  val_missing_predictions.csv  <=  provenance.json

Usage
-----
    python scripts/audit_cell_integrity.py --root experiments/.../runs \
        --manifest .../confirm29.json [--manifest ...] [--all-on-disk]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ARTIFACTS = ('config.json', 'metrics.json', 'val_pred.csv', 'best.pt',
             'robustness_val.json', 'val_noise_predictions.csv',
             'val_missing_predictions.csv', 'provenance.json')
ORDER = ('best.pt', 'robustness_val.json', 'val_noise_predictions.csv',
         'val_missing_predictions.csv', 'provenance.json')


def check_cell(cell: Path) -> dict:
    out = {'cell': str(cell), 'problems': []}
    st = cell / 'status.json'
    status = None
    if st.exists():
        try:
            status = json.loads(st.read_text()).get('status')
        except Exception:
            status = 'UNREADABLE'
    out['status'] = status
    if status != 'COMPLETE':
        out['problems'].append(f'status={status}')
        return out
    missing = [a for a in ARTIFACTS if not (cell / a).exists()]
    if missing:
        out['problems'].append(f'missing artifacts {missing}')
        return out
    mt = {a: (cell / a).stat().st_mtime for a in ORDER}
    for a, b in zip(ORDER, ORDER[1:]):
        if mt[a] > mt[b]:
            out['problems'].append(
                f'mtime out of order: {a} newer than {b} '
                f'(delta {mt[a] - mt[b]:.0f}s)')
    return out


def cells_from_manifest(root: Path, path: Path) -> list[Path]:
    man = json.loads(path.read_text())
    out = []
    for r in man['runs']:
        d = root / r['run_id'] / r['subset'] / f's{r["seed"]}'
        out.append(d)
    return out


def cells_on_disk(root: Path) -> list[Path]:
    """All cells that have a config.json, discovered by file, not by name prefix."""
    out = []
    for cfg in sorted(root.glob('*/*/s*/config.json')):
        out.append(cfg.parent)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--manifest', action='append', default=[])
    ap.add_argument('--all-on-disk', action='store_true')
    ap.add_argument('--quiet-ok', action='store_true')
    args = ap.parse_args()

    root = Path(args.root)
    cells = []
    for m in args.manifest:
        cells.extend(cells_from_manifest(root, Path(m)))
    if args.all_on_disk:
        cells.extend(cells_on_disk(root))
    # de-dup, keep order
    seen, uniq = set(), []
    for c in cells:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    ok = bad = absent = 0
    problems = []
    for c in uniq:
        if not c.exists():
            absent += 1
            continue
        r = check_cell(c)
        if r['problems']:
            bad += 1
            problems.append(r)
        else:
            ok += 1
    print(f'cells checked {len(uniq)}  ok {ok}  problems {bad}  not-yet-created {absent}')
    for r in problems:
        print(f'  {r["cell"]}: ' + ' | '.join(r['problems']))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
