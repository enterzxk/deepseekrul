#!/usr/bin/env python3
"""Switch a built manifest to the confirmation split and recompute run ids.

Why this exists
---------------
`split_seed` lives in the runner's BASE config, and the runner is part of the
h8 code fingerprint. Editing it to add a `--split-seed` flag would change the
fingerprint of the whole round, which is exactly what the protocol forbids.
So the manifest is built normally and then rewritten here: only `split_seed`
changes, every `run_id` is recomputed from the same `config_hash` the runner
uses, and nothing else in the config is touched.

Usage
-----
    python scripts/freeze_confirm_manifest.py <manifest.json> [split_seed]
    python scripts/freeze_confirm_manifest.py <manifest.json> --check

`--check` verifies an already-frozen manifest without rewriting it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def config_hash(cfg) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def run_id_for(group, subset, seed, cfg) -> str:
    return f'{group}-{subset}-s{seed}-{config_hash(cfg)}'


def check(man, split):
    problems = []
    for r in man['runs']:
        cfg = dict(r['config'])
        cfg['tag'] = ''
        cfg.pop('run_id', None)
        expect = run_id_for(r['group'], r['subset'], r['seed'], cfg)
        if r['run_id'] != expect:
            problems.append(f'run_id mismatch {r["run_id"]} != {expect}')
        if cfg.get('split_seed') != split:
            problems.append(f'{r["run_id"]}: split_seed={cfg.get("split_seed")}')
        if cfg.get('fp32') is not True:
            problems.append(f'{r["run_id"]}: fp32={cfg.get("fp32")}')
    return problems


def main() -> int:
    path = Path(sys.argv[1])
    man = json.loads(path.read_text())
    mode_check = '--check' in sys.argv
    split = 29
    for a in sys.argv[2:]:
        if a.isdigit():
            split = int(a)

    if mode_check:
        problems = check(man, split)
        print(f'runs {len(man["runs"])} split {split} problems {len(problems)}')
        for p in problems[:5]:
            print(' -', p)
        return 0 if not problems else 1

    changed = 0
    for r in man['runs']:
        cfg = r['config']
        if cfg.get('split_seed') == split:
            continue
        cfg['split_seed'] = split
        cfg['tag'] = ''
        cfg.pop('run_id', None)
        rid = run_id_for(r['group'], r['subset'], r['seed'], cfg)
        cfg['tag'] = rid
        cfg['run_id'] = rid
        r['run_id'] = rid
        r['tag'] = rid
        changed += 1

    man['split_seed_override'] = split
    man['shared_noise_stats_split_seed'] = 17
    man['shared_noise_stats_unused_by'] = ['E0', 'E1a']
    man['note'] = ('confirmation set: split_seed switched before any run; '
                   'no hyperparameter, seed, or arm definition changed. '
                   'The shared noise statistics were estimated under split_seed=17; '
                   'E0 and E1a have noise_alpha=0 and do not read them.')
    path.write_text(json.dumps(man, indent=1))

    problems = check(man, split)
    print(f'runs {len(man["runs"])} changed {changed} split {split} problems {len(problems)}')
    for p in problems[:5]:
        print(' -', p)
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
