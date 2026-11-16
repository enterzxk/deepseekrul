#!/usr/bin/env python3
"""Build the E5 manifest: four arms of the low-rank sensor interaction.

Derived from `task_config('E0', ...)` plus a single `lr_mode` key, so every arm
differs from the E0 config by exactly one variable -- the same discipline used for
the confirmation manifests. The runner's group table is NOT touched (E3 is still
out of scope and E5 is not an augmentation arm), so `scripts/run_sensor_aug.py`
stays byte-identical and only `dspt.py` + `train.py` move the h8.

Arms (E5_PREREG.md §2): none | lr4 | lr14 | mlp, 4 subsets x 3 seeds = 48 cells.

Usage
-----
    python scripts/freeze_e5_manifest.py --data-dir <dir> --name e5_lowrank.json
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
    MANIFESTS, STUDY_ID, code_fingerprint, run_id_for, task_config,
)

ARMS = ['none', 'lr4', 'lr14', 'mlp']
SEEDS = [101, 202, 303]
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']


def make_row(arm: str, subset: str, seed: int, data_dir: Path) -> dict:
    cfg = task_config('E0', subset, seed, data_dir)
    cfg['lr_mode'] = arm
    rid = run_id_for(f'E5{arm}', subset, seed, cfg)
    cfg['tag'] = rid
    cfg['run_id'] = rid
    return dict(run_id=rid, group=f'E5{arm}', subset=subset, seed=seed, tag=rid, config=cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--name', default='e5_lowrank.json')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = [make_row(a, s, sd, data_dir) for a in ARMS for sd in SEEDS for s in SUBSETS]
    ids = [r['run_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit('duplicate run_id in E5 manifest')

    code_hash, legacy7, head = code_fingerprint()
    man = {
        'study_id': STUDY_ID,
        'experiment': 'E5 low-rank sensor interaction (mechanism probe)',
        'code_fingerprint': code_hash,
        'code_fingerprint_legacy7': legacy7,
        'git_head': head,
        'arms': ARMS,
        'prereg': 'docs/paper_b_sensor_generalization/E5_PREREG.md (v1.1, frozen)',
        'note': ('All four arms share this h8. The none arm is retrained here as the '
                 'within-E5 baseline; the stage-1 E0 is a magnitude reference only and '
                 'must not be used as the paired baseline. Every arm is the E0 config plus '
                 'one lr_mode key, so arms differ by a single variable.'),
        'runs': rows,
    }
    out = MANIFESTS / args.name
    out.write_text(json.dumps(man, indent=1))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f'[write] {out}')
    print(f'[write] runs={len(rows)} arms={ARMS} seeds={SEEDS} subsets={SUBSETS} '
          f'split_seed={rows[0]["config"]["split_seed"]} fp32={rows[0]["config"]["fp32"]} '
          f'fp={code_hash} legacy7={legacy7} sha256={sha[:16]}')
    print('sample run_id:', rows[0]['run_id'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
