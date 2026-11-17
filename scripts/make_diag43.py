#!/usr/bin/env python3
"""Build split-43 diagnostic runs + the pre-registered truncation table.

Two artifacts, both hashed BEFORE any training:

  1. `diag43_e0.json`        -- 12 runs (E0 x 4 subsets x 3 seeds) at split_seed=43
                                with `patience == epochs`, so early stopping can never
                                fire (`ep - best_ep <= 39 < 40`). That is what makes the
                                validation engines selection-free: `last.pt` holds both
                                `model` (epoch 40) and `best_state`, and only `model`
                                was never chosen by the validation metric.
  2. `diag43_truncations.json` -- per validation engine: bucket, target RUL, truncation
                                cycle, plus the seed. Buckets follow the official TEST
                                RUL histogram of the same subset (cap 125), so the
                                diagnostic population matches TEST instead of the fixed
                                40/55/70/85/95% cuts.

The engine partition is reproduced exactly as `paper_b/training/data.py:split_units`
does it (same seed arithmetic, same permutation, same rounding) -- recomputing it any
other way would silently diagnose a different set of engines.

Usage
-----
    python scripts/make_diag43.py --root experiments/paper_b_sensor_generalization \
        --data-dir /home/user/deepseek-rul/data/CMAPSS
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = (101, 202, 303)
SPLIT = 43
VAL_FRAC = 0.2
FOLD = 0
EPOCHS = 40
PATIENCE = 40
CAP = 125.0
BUCKETS = ((0, 25), (25, 50), (50, 75), (75, 100), (100, 126))
# official TEST RUL histogram (capped at 125), measured from RUL_FD00x.txt
TEST_FRAC = {
    'FD001': (0.190, 0.110, 0.130, 0.230, 0.340),
    'FD002': (0.224, 0.112, 0.120, 0.185, 0.359),
    'FD003': (0.150, 0.140, 0.200, 0.180, 0.330),
    'FD004': (0.198, 0.121, 0.105, 0.165, 0.411),
}


def config_hash(cfg) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def run_id_for(group, subset, seed, cfg) -> str:
    return f'{group}-{subset}-s{seed}-{config_hash(cfg)}'


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def split_units(units, split_seed=SPLIT, fold=FOLD, val_frac=VAL_FRAC):
    """Byte-for-byte the partition rule of paper_b/training/data.py:split_units."""
    units = np.sort(np.asarray(units))
    rng = np.random.default_rng(split_seed + 20261020)
    sh = rng.permutation(units)
    n_val = max(1, int(round(units.size * val_frac)))
    val = sorted(int(x) for x in sh[fold * n_val:(fold + 1) * n_val])
    train = sorted(int(x) for x in units if int(x) not in set(val))
    return train, val


def lifetimes(path: Path) -> dict[int, int]:
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(float(parts[0]))
            cyc = int(float(parts[1]))
            out[u] = max(out.get(u, 0), cyc)
    return out


def allocate(n, fracs):
    """Largest-remainder allocation of n items over len(fracs) buckets."""
    raw = [n * f for f in fracs]
    base = [int(np.floor(x)) for x in raw]
    rem = n - sum(base)
    order = sorted(range(len(raw)), key=lambda i: -(raw[i] - base[i]))
    for i in order[:rem]:
        base[i] += 1
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--data-dir', required=True)
    a = ap.parse_args()
    root, data = Path(a.root), Path(a.data_dir)
    man_dir = root / 'manifests'
    man_dir.mkdir(parents=True, exist_ok=True)

    # ---- reference config: an existing E0 cell, re-pointed at split 43 ----
    # The run-id prefix alone matches BOTH the split-17 and split-29 batches; filter on
    # config.json and demand exactly one, never `sorted()[0]`.
    ref = []
    for d in sorted((root / 'runs').glob('E0-FD001-s101-*')):
        cfg = json.loads((d / 'FD001' / 's101' / 'config.json').read_text())
        if cfg.get('split_seed') == 29:
            ref.append((d, cfg))
    if len(ref) != 1:
        raise SystemExit(f'need exactly one split-29 E0-FD001-s101 reference, got {len(ref)}')
    base_cfg = ref[0][1]

    runs, trunc = [], {}
    for subset in SUBSETS:
        life = lifetimes(data / f'train_{subset}.txt')
        units = sorted(life)
        train, val = split_units(units)
        n = len(val)
        counts = allocate(n, TEST_FRAC[subset])
        if sum(counts) != n:
            raise SystemExit(f'{subset}: allocation {sum(counts)} != {n}')

        rng = np.random.default_rng(SPLIT * 1000 + SUBSETS.index(subset))
        rows, k = [], 0
        fallback = 0
        for b, (lo, hi) in enumerate(BUCKETS):
            for _ in range(counts[b]):
                u = val[k]
                k += 1
                top = min(int(CAP), life[u])
                rlo, rhi = max(1, lo), min(hi - 1, top)
                if rlo > rhi:                      # engine too short for this bucket
                    rlo = max(1, top)
                    rhi = top
                    fallback += 1
                r = int(rng.integers(rlo, rhi + 1))
                rows.append(dict(unit=u, life=life[u], bucket=f'{lo}-{hi}',
                                 target_rul=r, truncation_cycle=life[u] - r,
                                 fallback=bool(rlo == max(1, top) and rhi == top
                                               and not (max(1, lo) <= min(hi - 1, top)))))
        rows.sort(key=lambda x: x['unit'])
        trunc[subset] = dict(n_val=n, n_diag=n, bucket_counts=counts,
                             test_fraction=TEST_FRAC[subset],
                             val_units=val, n_fallback=fallback, rows=rows)
        print(f'{subset}: engines={len(units)} train={len(train)} val={n} '
              f'buckets={counts} fallback={fallback}')

        for seed in SEEDS:
            cfg = dict(base_cfg)
            cfg['augment'] = None
            cfg.update(group='E0', subset=subset, seed=seed, split_seed=SPLIT,
                       epochs=EPOCHS, patience=PATIENCE)
            cfg.pop('run_id', None)
            cfg['tag'] = ''
            rid = run_id_for('E0', subset, seed, cfg)
            cfg['tag'] = rid
            cfg['run_id'] = rid
            runs.append(dict(run_id=rid, group='E0', subset=subset, seed=seed,
                             tag=rid, config=cfg))

    man = dict(study_id='paper_b_sensor_generalization',
               purpose=('diagnostic platform: selection-free checkpoints + TEST-shaped '
                        'truncation, split_seed=43'),
               built_by='scripts/make_diag43.py',
               split_seed=SPLIT, epochs=EPOCHS, patience=PATIENCE,
               selection_free_because=('patience == epochs, so early stopping never fires '
                                       '(ep - best_ep <= 39 < 40). last.pt therefore holds '
                                       '`model` (never chosen by the validation metric) '
                                       'alongside `best_state`.'),
               not_reusing='no split-29 checkpoint is reused; those were selection-'
                           'contaminated by construction',
               n_runs=len(runs), runs=runs)
    p_man = man_dir / 'diag43_e0.json'
    p_man.write_text(json.dumps(man, indent=1) + '\n', encoding='utf-8')

    trunc_doc = dict(study_id='paper_b_sensor_generalization',
                     purpose='pre-registered diagnostic truncation table',
                     built_by='scripts/make_diag43.py',
                     rule=('bucket allocation = official TEST RUL histogram of the same '
                           'subset (cap 125), largest-remainder rounding; target RUL drawn '
                           'uniformly inside the bucket with a fixed per-subset seed; '
                           'truncation_cycle = engine life - target_rul'),
                     cap=int(CAP), seeds=dict(per_subset='split_seed*1000 + subset_index'),
                     subsets=trunc)
    p_tr = man_dir / 'diag43_truncations.json'
    p_tr.write_text(json.dumps(trunc_doc, indent=1) + '\n', encoding='utf-8')

    lines = [f'{sha256_file(p)}  {p.name}' for p in (p_man, p_tr)]
    sha = man_dir / 'diag43_sha256.txt'
    sha.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print()
    for ln in lines:
        print(ln)
    print(f'-> {sha}')
    tot = sum(v['n_diag'] for v in trunc.values())
    print(f'total diagnostic engines: {tot}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
