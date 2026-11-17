#!/usr/bin/env python3
"""Build the exploratory manifests (E3 + confirmation-29 + TEST extension-72).

Why these are hand-written instead of `run_sensor_aug.py build`
---------------------------------------------------------------
`task_config()` rejects E3 (`mask_prob > 0 and noise_alpha > 0`) by design, and it is
called **only** from `build()`. The worker consumes `task['config']` verbatim, so an
E3 run can be launched from a manifest without touching `run_sensor_aug.py` -- which
the protocol forbids, because the runner is part of the h8 fingerprint. E4 and the E5
arms are not in `GROUPS` either; their split-17 manifests were built the same way.

Two different pairings, deliberately
------------------------------------
*Training* comparison (E3@17, E2b/E4/E5@29) is **within-tree**: the current tree
fingerprints as h8=574ac4a06ae7, so the only same-tree baseline is `E5none`
(`lr_mode='none'`, verified bit-identical to E0 on 12/12 cells). `E0`'s split-29 cells
are on 515eb76fc396 and cannot be re-trained on this tree (run_id collides -> skipped),
so pairing against them would be cross-tree.

*TEST inference* comparison has no such problem: those cells were all trained before
the tree changed, and inference does not retrain. So the TEST manifest keeps the
analysis convention -- E5 arms pair `E5none`, E2b/E4 pair `E0`.

`paired_baseline` is a TASK-level key, never inside `config`: putting it in the config
would change `config_hash` and therefore the run_id.

Usage
-----
    python scripts/make_exploratory_manifests.py --root experiments/paper_b_sensor_generalization
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

E3_SEEDS = (101, 202, 303)
CONF_SEEDS = (42, 43, 101, 202, 303)
SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
E5_ARMS = ('E5none', 'E5lr4', 'E5lr14', 'E5mlp')


def config_hash(cfg) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def run_id_for(group, subset, seed, cfg) -> str:
    return f'{group}-{subset}-s{seed}-{config_hash(cfg)}'


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def find_cell(runs: Path, prefix: str, subset: str, seed: int, split: int):
    """Exactly one matching cell, or raise -- never `sorted()[0]`."""
    hits = []
    for d in runs.glob(f'{prefix}-{subset}-s{seed}-*'):
        cfg = json.loads((d / subset / f's{seed}' / 'config.json').read_text())
        if cfg.get('split_seed') == split:
            hits.append((d, cfg))
    if len(hits) != 1:
        raise SystemExit(f'{prefix} {subset} s{seed} split{split}: {len(hits)} matches')
    return hits[0]


def make_task(group, subset, seed, base_cfg, *, split_seed, augment=None, baseline=None):
    cfg = dict(base_cfg)
    cfg['augment'] = dict(base_cfg['augment']) if base_cfg.get('augment') else None
    if augment is not None:
        cfg['augment'] = dict(cfg['augment'] or {})
        cfg['augment'].update(augment)
    cfg.update(group=group, subset=subset, seed=seed, split_seed=split_seed)
    cfg.pop('run_id', None)
    cfg['tag'] = ''
    rid = run_id_for(group, subset, seed, cfg)
    cfg['tag'] = rid
    cfg['run_id'] = rid
    task = dict(run_id=rid, group=group, subset=subset, seed=seed, tag=rid, config=cfg)
    if baseline is not None:
        task['paired_baseline'] = baseline      # task-level on purpose: not hashed
    return task


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    a = ap.parse_args()
    root = Path(a.root)
    runs, shared, man_dir = root / 'runs', root / 'shared', root / 'manifests'
    man_dir.mkdir(parents=True, exist_ok=True)

    # ---------- A. E3 (mask 0.10 + noise 0.25), split 17 ----------
    e3 = []
    for subset in SUBSETS:
        st = json.loads((shared / f'noise_stats_{subset}.json').read_text())
        for seed in E3_SEEDS:
            _, cfg = find_cell(runs, 'E1a', subset, seed, 17)
            e3.append(make_task(
                'E3', subset, seed, cfg, split_seed=17,
                augment=dict(noise_alpha=0.25,
                             noise_stats_path=str(shared / f'noise_stats_{subset}.json'),
                             noise_stats_hash=st['hash']),
                baseline=dict(arm='E5none', split_seed=17, subset=subset, seed=seed,
                              why='same-tree baseline; E5none == E0 on 12/12 cells')))
    print(f'E3 split17: {len(e3)} runs')

    # ---------- B. confirmation split 29: E2b + E4 + E5 arms ----------
    conf = []
    for subset in SUBSETS:
        st29 = json.loads((shared / f'noise_stats_{subset}_split29.json').read_text())
        for seed in CONF_SEEDS:
            _, cfg = find_cell(runs, 'E2b', subset, 101, 17)
            conf.append(make_task(
                'E2b', subset, seed, cfg, split_seed=29,
                augment=dict(noise_stats_path=str(shared / f'noise_stats_{subset}_split29.json'),
                             noise_stats_hash=st29['hash']),
                baseline=dict(arm='E5none', split_seed=29, subset=subset, seed=seed,
                              why='same-tree; E0 split-29 is 515eb76fc396 and cannot be '
                                  're-trained on this tree (run_id collides)')))
    for arm in ('E4',) + E5_ARMS:
        for subset in SUBSETS:
            for seed in CONF_SEEDS:
                _, cfg = find_cell(runs, arm, subset, 101, 17)
                conf.append(make_task(
                    arm, subset, seed, cfg, split_seed=29,
                    baseline=dict(arm='E5none', split_seed=29, subset=subset, seed=seed,
                                  why='same-tree; E5 arms never pair against E0')))
    print(f'confirmation split29: {len(conf)} runs '
          f'({ {a: sum(1 for r in conf if r["group"] == a) for a in ("E2b", "E4") + E5_ARMS} })')

    # ---------- C. TEST extension: 72 already-trained cells ----------
    ext = []
    for arm in ('E2b', 'E4') + E5_ARMS:
        for subset in SUBSETS:
            for seed in E3_SEEDS:
                d, cfg = find_cell(runs, arm, subset, seed, 17)
                leaf = d / subset / f's{seed}'
                st = json.loads((leaf / 'status.json').read_text()).get('status')
                if st != 'COMPLETE':
                    raise SystemExit(f'{d.name}: status {st}, refusing to list it for TEST')
                base = 'E5none' if arm.startswith('E5') else 'E0'
                ext.append(dict(run_id=d.name, group=arm, subset=subset, seed=seed,
                                tag=d.name, config=cfg,
                                paired_baseline=dict(
                                    arm=base, split_seed=17, subset=subset, seed=seed,
                                    why='TEST inference pairing; all these cells were '
                                        'trained before the tree changed')))
    print(f'TEST extension: {len(ext)} runs')
    assert len(e3) == 12 and len(conf) == 120 and len(ext) == 72, (len(e3), len(conf), len(ext))

    def dump(name, obj):
        p = man_dir / name
        p.write_text(json.dumps(obj, indent=1) + '\n', encoding='utf-8')
        return p

    p_e3 = dump('exploratory_e3_split17.json', dict(
        study_id='paper_b_sensor_generalization',
        purpose='exploratory: E3 = mask 0.10 + noise 0.25 (was skipped by design)',
        built_by='scripts/make_exploratory_manifests.py',
        baseline_policy=dict(arm='E5none', split_seed=17,
                             why='same-tree baseline; E5none == E0 on 12/12 cells'),
        note=('E3 is rejected by task_config()/build(); this manifest is hand-written so '
              'run_sensor_aug.py (part of h8) is not modified. Exploratory only: E3 does '
              'not enter any frozen criterion and cannot change the three-tier verdict.'),
        n_runs=len(e3), runs=e3))

    p_conf = dump('exploratory_confirm29.json', dict(
        study_id='paper_b_sensor_generalization',
        purpose='confirmation split for arms that never had one (E2b, E4, E5 four arms)',
        built_by='scripts/make_exploratory_manifests.py',
        split_seed_override=29,
        shared_noise_stats_split_seed=29,
        baseline_policy=dict(arm='E5none', split_seed=29,
                             why='within-tree pairing; E0 split-29 is on 515eb76fc396 and '
                                 'cannot be re-trained on this tree because run_id collides '
                                 'and the worker skips completed cells'),
        note=('Cloned from the split-17 arm definitions; only split_seed and the seed set '
              'change. noise_stats for E2b is re-estimated on the split-29 training '
              'engines (noise_stats_<subset>_split29.json) because the prereg says the '
              'scale is estimated from training engines only. h8 will be 574ac4a06ae7 '
              '(current tree) rather than 515eb76fc396; lr_mode defaults to none and the '
              'E5 none arm reproduced E0 bit-for-bit on 12/12 cells.'),
        n_runs=len(conf), runs=conf))

    p_ext = dump('exploratory_test_ext72.json', dict(
        study_id='paper_b_sensor_generalization',
        purpose='post-hoc TEST evaluation of the 72 already-trained exploratory cells',
        built_by='scripts/make_exploratory_manifests.py',
        baseline_policy=dict(E5_ARMS='E5none', E2b='E0', E4='E0',
                             why='TEST inference pairing; no retraining, so the tree change '
                                 'does not make E0 a cross-tree baseline here'),
        n_runs=len(ext), runs=ext))

    freeze = dict(
        study_id='paper_b_sensor_generalization',
        frozen=True, post_hoc=True, cannot_change_verdict=True, exploratory=True,
        verdict_frozen_at='11c57bc0',
        test_read_extension=True,
        what_was_already_read=('the primary post-hoc TEST read covered E0/E1a/E1b/E2a x '
                               'split 17 and 29 (128 cells) and is recorded under '
                               'posthoc_test_128.freeze.json; this extension adds arms that '
                               'were NOT in that read'),
        extension_policy=('These 72 cells are evaluated AFTER the primary read. They are '
                          'exploratory: they enter no frozen criterion and cannot change the '
                          'three-tier verdict or revive a candidate. Reporting them without '
                          'this label would be exactly the "look, then keep what looks good" '
                          'failure the opened-once rule exists to prevent.'),
        runs=[dict(group=r['group'], subset=r['subset'], seed=r['seed'],
                   run_id=r['run_id'], split_seed=17) for r in ext],
    )
    p_fz = dump('exploratory_test_ext72.freeze.json', freeze)

    lines = [f'{sha256_file(p)}  {p.name}' for p in (p_e3, p_conf, p_ext, p_fz)]
    sha = man_dir / 'exploratory_sha256.txt'
    sha.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print()
    for ln in lines:
        print(ln)
    print(f'-> {sha}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
