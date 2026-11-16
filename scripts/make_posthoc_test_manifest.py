#!/usr/bin/env python3
"""Build the post-hoc official-TEST manifest + a hand-written freeze.

Why a hand-written freeze: `freeze()` in run_sensor_aug.py accepts exactly one
candidate (`group in ('E0', cand)`), so it cannot express "two ENGINEERING-ONLY
arms plus their controls". `test_role()` only requires `frozen: true` and a
`runs` list of `(group, subset, seed)`, so the allow-list is written here.

Why derived from disk: the manifest is built by scanning each cell's own
`config.json` and requiring `status == 'COMPLETE'`. A plan file would happily
list cells that never finished; a disk scan cannot.

Order matters: this script writes the manifest and the freeze and their sha256,
and those must be committed BEFORE the first test evaluation. `test_rules`
records `opened_once=True` -- the test set is opened once, so the protocol is
fixed before anything runs.

Usage
-----
    python scripts/make_posthoc_test_manifest.py --root experiments/paper_b_sensor_generalization
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ARMS = ('E0', 'E1a', 'E1b', 'E2a')
SPLITS = (17, 29)
SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
VERDICT_FROZEN_AT = '11c57bc0'


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(root: Path) -> list[dict]:
    runs_dir = root / 'runs'
    out, skipped = [], []
    for cfg_path in sorted(runs_dir.glob('*/*/s*/config.json')):
        cfg = json.loads(cfg_path.read_text())
        run_dir = cfg_path.parents[2]
        run_id = run_dir.name
        arm = run_id.split('-')[0]
        if arm not in ARMS or cfg.get('split_seed') not in SPLITS:
            continue
        status = None
        st = cfg_path.parent / 'status.json'
        if st.exists():
            status = json.loads(st.read_text()).get('status')
        if status != 'COMPLETE':
            skipped.append((run_id, cfg['subset'], cfg['seed'], status))
            continue
        out.append(dict(run_id=run_id, group=arm, subset=cfg['subset'],
                        seed=int(cfg['seed']), tag=run_id, config=cfg))
    out.sort(key=lambda t: (t['config']['split_seed'], t['group'], t['subset'], t['seed']))
    return out, skipped


def write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=1) + '\n', encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    a = ap.parse_args()
    root = Path(a.root)
    man_dir = root / 'manifests'
    man_dir.mkdir(parents=True, exist_ok=True)

    runs, skipped = collect(root)
    if skipped:
        print(f'[warn] {len(skipped)} cell(s) on disk but not COMPLETE -> excluded')
        for s in skipped[:8]:
            print('   ', s)

    counts = {}
    for t in runs:
        counts[(t['config']['split_seed'], t['group'])] = \
            counts.get((t['config']['split_seed'], t['group']), 0) + 1
    for k in sorted(counts):
        print(f'  split {k[0]}  {k[1]:4s} {counts[k]:3d}')
    print(f'total {len(runs)} cells')

    # ---- full manifest ---------------------------------------------------
    man = dict(
        study_id='paper_b_sensor_generalization',
        purpose='post-hoc official TEST evaluation (does NOT change any frozen verdict)',
        generated_by='scripts/make_posthoc_test_manifest.py',
        derived_from='on-disk config.json of each cell (status must be COMPLETE)',
        n_runs=len(runs),
        runs=runs,
    )
    man_path = man_dir / 'posthoc_test_128.json'
    write_json(man_path, man)

    # ---- hand-written freeze --------------------------------------------
    freeze = dict(
        study_id='paper_b_sensor_generalization',
        frozen=True,
        post_hoc=True,
        cannot_change_verdict=True,
        verdict_frozen_at=VERDICT_FROZEN_AT,
        opened_once=True,
        why_handwritten=('run_sensor_aug.freeze() accepts a single candidate '
                         '(group in ("E0", cand)); this round has two ENGINEERING-ONLY '
                         'arms (E1a, E1b) plus controls, which that gate cannot express. '
                         'test_role() only needs frozen:true plus a runs allow-list.'),
        arms=list(ARMS),
        splits=list(SPLITS),
        scope_note=('Official TEST results are a POST-HOC robustness report only. The '
                    'GO / ENGINEERING-ONLY / STOP verdict was fixed on the validation '
                    'partition under the preregistered criteria; TEST may not revise it. '
                    'If TEST disagrees with the validation ordering, that is recorded as '
                    'a limitation and the claim scope is narrowed -- never the verdict.'),
        metric_note=('C-MAPSS standard: one prediction per test engine at the last '
                     'window, pooled RMSE over all test engines, NASA score as usual. '
                     'The validation quantity val_engine_mean_roots_rmse (mean over 5 '
                     'cut points per engine) has NO TEST counterpart; a validation delta '
                     'must not be compared numerically against a TEST delta.'),
        test_rules=dict(metrics=['pooled_rmse', 'engine_equal_rmse', 'mae',
                                 'nasa_mean', 'nasa_sum'],
                        missing_ks=[0, 1, 2, 3, 4],
                        noise_betas=[0.0, 0.25, 0.50, 1.00],
                        single_sensor=True, opened_once=True),
        runs=[dict(group=t['group'], subset=t['subset'], seed=t['seed'],
                   run_id=t['run_id'], split_seed=t['config']['split_seed'])
              for t in runs],
    )
    fz_path = man_dir / 'posthoc_test_128.freeze.json'
    write_json(fz_path, freeze)

    # ---- smoke pair: exactly one cell ------------------------------------
    smoke = [t for t in runs if t['run_id'].startswith('E0-FD001-s101-')
             and t['config']['split_seed'] == 17]
    if len(smoke) != 1:
        raise SystemExit(f'smoke cell: expected exactly 1, got {len(smoke)}')
    smoke_man = dict(study_id='paper_b_sensor_generalization',
                     purpose='single-cell smoke test of the TEST path; plumbing only',
                     generated_by='scripts/make_posthoc_test_manifest.py',
                     n_runs=1, runs=smoke)
    sm_path = man_dir / 'posthoc_test_smoke.json'
    write_json(sm_path, smoke_man)
    smoke_freeze = dict(freeze)
    smoke_freeze['smoke_only'] = True
    smoke_freeze['runs'] = [r for r in freeze['runs'] if r['run_id'] == smoke[0]['run_id']]
    sfz_path = man_dir / 'posthoc_test_smoke.freeze.json'
    write_json(sfz_path, smoke_freeze)

    # ---- hashes ----------------------------------------------------------
    lines = []
    for p in (man_path, fz_path, sm_path, sfz_path):
        lines.append(f'{sha256_file(p)}  {p.name}')
    sha_path = man_dir / 'posthoc_test_sha256.txt'
    sha_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print()
    for ln in lines:
        print(ln)
    print(f'-> {sha_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
