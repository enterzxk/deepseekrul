#!/usr/bin/env python3
"""Freeze confirmation manifests without touching the h8 runner or `build`.

Why this exists
---------------
`build` has two side effects that are unsafe while a confirmation batch is
running: it rewrites `shared/noise_stats_*.json` and
`shared/missing_patterns_val_*.json` (a worker reading mid-write would see a
truncated file), and it rewrites `paper_b/sensor_aug/configs/`.

This script instead imports the runner's own `task_config` and `run_id_for`,
so every config and every `run_id` is byte-identical to what
`build` + `freeze_confirm_manifest.py` would produce, and it only ever reads
from `shared/`.

Frozen-arm artifacts are deliberately reused
--------------------------------------------
`noise_stats_hash` (E2a) points at the split_seed=17 artifact, exactly as in
the discovery stage: the noise scale is injected during training, so it is part
of the frozen arm definition and re-estimating it on the confirmation partition
would change the arm. `mask_hash` is different in kind — the missing-pattern
bank is evaluation-only and keyed by engine id, so it MUST be rebuilt on the
confirmation partition (see `refresh_eval_patterns.py`). Only the data
partition changes, from 17 to 29.

Usage
-----
    python scripts/freeze_confirm_extra.py --data-dir <dir> \
        --reference <old confirm29.json> --groups E0,E1a --name confirm29.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_sensor_aug import (  # noqa: E402
    MANIFESTS, SHARED, STUDY_ID, code_fingerprint, run_id_for, task_config,
)

SEEDS = [101, 202, 303, 42, 43]
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']


def canon(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def make_row(group: str, subset: str, seed: int, data_dir: Path, split_seed: int) -> dict:
    cfg = task_config(group, subset, seed, data_dir)
    aug = cfg['augment']
    if aug is not None and aug['noise_alpha'] > 0:
        st = json.loads((SHARED / f'noise_stats_{subset}.json').read_text())
        aug['noise_stats_hash'] = st['hash']
    if aug is not None and aug['mask_prob'] > 0:
        mp = json.loads((SHARED / f'missing_patterns_val_{subset}.json').read_text())
        aug['mask_hash'] = mp['hash']
    # The confirmation split is applied before the run id is derived, exactly
    # like freeze_confirm_manifest.py does it.
    cfg['split_seed'] = split_seed
    rid = run_id_for(group, subset, seed, cfg)
    cfg['tag'] = rid
    cfg['run_id'] = rid
    return dict(run_id=rid, group=group, subset=subset, seed=seed, tag=rid, config=cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--groups', required=True, help='arms to emit, e.g. E1b or E0,E1a')
    ap.add_argument('--name', required=True)
    ap.add_argument('--split-seed', type=int, default=29)
    ap.add_argument('--reference', default='', help='manifest whose rows must be reproduced exactly')
    ap.add_argument('--reference-groups', default='E0',
                    help='which arms of --reference must match (E0 carries no mask/noise hash)')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    groups = [g for g in args.groups.split(',') if g]
    ref_groups = [g for g in args.reference_groups.split(',') if g] if args.reference else []

    # Build the reference arms too, so we can prove the construction path is the
    # same one that produced the batch already on disk.
    rows = {}
    for g in groups:
        for sd in SEEDS:
            for s in SUBSETS:
                r = make_row(g, s, sd, data_dir, args.split_seed)
                rows[r['run_id']] = r

    if args.reference:
        ref = json.loads(Path(args.reference).read_text())
        want = [r for r in ref['runs'] if r['group'] in ref_groups]
        # reference arms may not be in --groups; rebuild them just for the check
        rebuilt = [rows.get(x['run_id'])
                   or make_row(x['group'], x['subset'], x['seed'], data_dir, args.split_seed)
                   for x in want]
        check = {r['run_id']: r for r in rebuilt}
        missing, mismatched = [], []
        for r in want:
            got = check.get(r['run_id'])
            if got is None:
                missing.append(r['run_id'])
            elif canon(got['config']) != canon(r['config']):
                mismatched.append(r['run_id'])
        if missing or mismatched:
            print(f'REFUSING: reference not reproduced (missing={len(missing)} '
                  f'mismatched={len(mismatched)})')
            for x in (missing + mismatched)[:5]:
                print('  -', x)
            return 1
        print(f'[check] {len(want)} reference rows ({",".join(ref_groups)}) reproduced exactly')

    out_rows = [rows[k] for k in sorted(rows)]
    ids = [r['run_id'] for r in out_rows]
    if len(set(ids)) != len(ids):
        print('REFUSING: duplicate run_id inside the new manifest')
        return 1

    noise_free = sorted({r['group'] for r in out_rows
                         if (r['config'].get('augment') or {}).get('noise_alpha', 0) == 0})
    code_hash, legacy7, head = code_fingerprint()
    man = {
        'study_id': STUDY_ID,
        'code_fingerprint': code_hash,
        'code_fingerprint_legacy7': legacy7,
        'git_head': head,
        'split_seed_override': args.split_seed,
        'shared_noise_stats_split_seed': 17,
        'shared_noise_stats_unused_by': noise_free,
        'shared_missing_patterns_split_seed': args.split_seed,
        'derived_from': Path(args.reference).name if args.reference else '',
        'baseline_rows_are_shared': True,
        'note': ('confirmation set: same arms as the discovery stage, only the data partition '
                 'changes (17 -> 29). noise_stats stays on split_seed=17 on purpose: the noise '
                 'scale is injected during training and is part of the frozen arm definition. '
                 'missing_patterns is rebuilt on split_seed=29 because it is evaluation-only and '
                 'keyed by engine id. The E0 baseline is NOT rerun; the paired baseline is the '
                 'E0 rows of the confirmation manifest.'),
        'runs': out_rows,
    }
    out = MANIFESTS / args.name
    out.write_text(json.dumps(man, indent=1))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f'[write] {out}')
    print(f'[write] runs={len(out_rows)} groups={sorted({r["group"] for r in out_rows})} '
          f'split_seed={args.split_seed} fp={code_hash} sha256={sha[:16]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
