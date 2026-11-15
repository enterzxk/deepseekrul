#!/usr/bin/env python3
"""Backfill per-run provenance.json from a manifest.

Why this exists
---------------
`train.py` writes `metadata.json` (git_head, torch, cuda, gpu_uuid, window
counts) but not the code fingerprint; only the manifest carries it. Reading
the fingerprint from the manifest is correct but easy to forget, and a run
directory should be self-describing.

Why not edit train.py
---------------------
`train.py` is inside the h8 code fingerprint. Editing it mid-round would give
E0 one fingerprint and E1/E2 another, which breaks the paired baseline. This
script writes only experiment artifacts, so h8 does not move.

Usage
-----
    python scripts/backfill_provenance.py --manifest <path/to/manifest.json>
    python scripts/backfill_provenance.py --manifest ... --dry-run

It only touches run directories whose `status.json` says COMPLETE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    man_path = Path(a.manifest).resolve()
    man = json.loads(man_path.read_text())
    study_dir = man_path.parent.parent
    runs_root = study_dir / 'runs'

    fp = man.get('code_fingerprint')
    fp7 = man.get('code_fingerprint_legacy7')
    head = man.get('git_head')
    if not fp:
        raise SystemExit(f'manifest has no code_fingerprint: {man_path}')

    written = skipped = 0
    for r in man.get('runs', []):
        rd = runs_root / r['run_id'] / r['subset'] / f"s{r['seed']}"
        st = rd / 'status.json'
        if not st.exists():
            skipped += 1
            continue
        try:
            status = json.loads(st.read_text()).get('status')
        except Exception:
            status = None
        if status != 'COMPLETE':
            skipped += 1
            continue

        run_head = None
        meta = rd / 'metadata.json'
        if meta.exists():
            try:
                run_head = json.loads(meta.read_text()).get('git_head')
            except Exception:
                run_head = None

        out = {
            'run_id': r['run_id'],
            'subset': r['subset'],
            'seed': int(r['seed']),
            'code_fingerprint': fp,
            'code_fingerprint_legacy7': fp7,
            'manifest_git_head': head,
            'run_git_head': run_head,
            'config_sha256': sha256_file(rd / 'config.json') if (rd / 'config.json').exists() else None,
            'backfilled_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'note': 'code identity = code_fingerprint (h8); git_head is the HEAD at '
                    'build time and may lag the branch head after doc-only commits',
        }
        if a.dry_run:
            print(f'[dry-run] would write {rd / "provenance.json"}')
        else:
            (rd / 'provenance.json').write_text(json.dumps(out, indent=1))
        written += 1

    print(f'[backfill] fingerprint={fp} legacy7={fp7} head={head}')
    print(f'[backfill] written={written} skipped={skipped} '
          f'({"dry-run" if a.dry_run else "written"})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
