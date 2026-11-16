#!/usr/bin/env python3
"""Freeze extra confirmation arms without touching the h8 runner or `build`.

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

It also refuses to emit anything unless the reference manifest's rows are
reproduced exactly, which is how we know the confirmation baseline really is
shared with the batch that is already running.

Frozen-arm artifacts are deliberately reused
--------------------------------------------
`noise_stats_hash` (E2a) and `mask_hash` (E1a/E1b) point at the split_seed=17
artifacts, exactly as in the discovery stage. That is the intended
confirmation design: the scale and the mask bank are part of the frozen arm
definition. Re-estimating them on the confirmation partition would change the
arm, not confirm it. Only the data partition changes, from 17 to 29.

Usage
-----
    python scripts/freeze_confirm_extra.py \
        --data-dir /home/user/deepseek-rul/data/CMAPSS \
        --reference experiments/.../manifests/confirm29.json \
        --groups E1b --name confirm29_e1b.json
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
BASELINE_GROUPS = ['E0', 'E1a']  # already frozen and running


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
    # Confirmation split is applied before the run id is derived, exactly like
    # freeze_confirm_manifest.py does it.
    cfg['split_seed'] = split_seed
    rid = run_id_for(group, subset, seed, cfg)
    cfg['tag'] = rid
    cfg['run_id'] = rid
    return dict(run_id=rid, group=group, subset=subset, seed=seed, tag=rid, config=cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--reference', required=True, help='manifest already running; rows must be reproduced exactly')
    ap.add_argument('--groups', required=True, help='comma separated extra arms, e.g. E1b or E1b,E2a')
    ap.add_argument('--name', required=True)
    ap.add_argument('--split-seed', type=int, default=29)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    extra = [g for g in args.groups.split(',') if g]

    # Rebuild the reference arms too, so we can prove the construction path is
    # the same one that produced the batch already in flight.
    baseline = [make_row(g, s, sd, data_dir, args.split_seed)
                for g in BASELINE_GROUPS for sd in SEEDS for s in SUBSETS]
    by_rid = {r['run_id']: r for r in baseline}

    ref = json.loads(Path(args.reference).read_text())
    missing, mismatched = [], []
    for r in ref['runs']:
        got = by_rid.get(r['run_id'])
        if got is None:
            missing.append(r['run_id'])
        elif canon(got['config']) != canon(r['config']):
            mismatched.append(r['run_id'])
    if missing or mismatched:
        print(f'REFUSING: reference not reproduced (missing={len(missing)} mismatched={len(mismatched)})')
        for x in (missing + mismatched)[:5]:
            print('  -', x)
        return 1
    print(f'[check] reference {len(ref["runs"])} rows reproduced exactly')

    rows = [make_row(g, s, sd, data_dir, args.split_seed)
            for g in extra for sd in SEEDS for s in SUBSETS]

    rid_all = [r['run_id'] for r in rows]
    if len(set(rid_all)) != len(rid_all):
        print('REFUSING: duplicate run_id inside the new manifest')
        return 1
    clash = set(rid_all) & set(by_rid)
    if clash:
        print(f'REFUSING: {len(clash)} run_id(s) collide with the reference arms: {sorted(clash)[:3]}')
        return 1

    noise_free = sorted({r['group'] for r in rows
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
        'derived_from': str(Path(args.reference).name),
        'baseline_rows_are_shared': True,
        'note': ('confirmation extension: same arms as the discovery stage, only the data '
                 'partition changes (17 -> 29). The frozen arm-definition artifacts '
                 '(noise_stats_hash / mask_hash) still point at the split_seed=17 shared '
                 'files on purpose: re-estimating them on the confirmation partition would '
                 'change the arm instead of confirming it. The E0 baseline is NOT rerun; '
                 'the paired baseline is the E0 rows of ' + Path(args.reference).name + '.'),
        'runs': rows,
    }
    out = MANIFESTS / args.name
    out.write_text(json.dumps(man, indent=1))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    groups = sorted({r['group'] for r in rows})
    print(f'[write] {out}')
    print(f'[write] runs={len(rows)} groups={groups} seeds={SEEDS} subsets={SUBSETS} '
          f'split_seed={args.split_seed} fp={code_hash} sha256={sha[:16]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
