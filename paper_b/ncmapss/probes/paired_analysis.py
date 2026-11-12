# Provenance (header added 2026-11-13 by zxk; the code below is unchanged apart from the
# ledger path, which moved to DS02_pilot/ when the pilot batch was archived).
#   produced: paired inner_val - fixed_passes:200 difference on the same (rung, engine, seed) cell
#   output:   stdout only; feeds the paired sign test table (DS02: 6/6, 4/6, 5/6 splits)
#   run:      python -m paper_b.ncmapss.probes.paired_analysis   (from the repo root)
"""Paired per-seed comparison of the two evaluation points.

Headline statistic: the paired difference `inner_val - fixed_passes:200` for the
same (rung, held-out engine, seed) cell -- the same trained weights, so the
difference carries no initialisation variance. Two variances side by side would
confound seed count with fold count; a paired difference does not.
"""
from __future__ import annotations

import collections
import csv
import statistics as st

LEDGER = 'experiments/ncmapss/DS02_pilot/runs_dev.csv'
RUNGS = ('N0', 'N1', 'N2', 'N3')
# max(Y) per engine: the same denominator evaluation.py uses for relative segments
MAXY = {2: 74, 5: 88, 10: 81, 16: 62, 18: 70, 20: 65}


def load():
    rows = [r for r in csv.DictReader(open(LEDGER)) if r.get('status') == 'ok']
    cell = collections.defaultdict(dict)          # (rung, engine, seed) -> {rule: rmse}
    for r in rows:
        v = r.get('rmse_raw_unit')
        if not v:
            continue
        eng = int(r['held_out_engine'])
        cell[(r['rung'], eng, r.get('seed', ''))][r['selection_rule']] = float(v)
    return cell


def main():
    cell = load()
    engines = sorted({k[1] for k in cell})
    print(f'cells={len(cell)} engines={engines}')

    print('\n=== paired difference inner_val - fixed_passes:200 (same weights) ===')
    for rung in RUNGS:
        diffs = [d['inner_val'] - d['fixed_passes:200'] for (rg, e, s), d in cell.items()
                 if rg == rung and 'inner_val' in d and 'fixed_passes:200' in d]
        if not diffs:
            continue
        pos = sum(1 for x in diffs if x > 0)
        print(f'{rung:4s} n={len(diffs):3d} median={st.median(diffs):+7.2f} '
              f'mean={st.mean(diffs):+7.2f} pos={pos}/{len(diffs)} '
              f'[{min(diffs):+.2f},{max(diffs):+.2f}]')

    print('\n=== per-engine mean RMSE (n seeds) ===')
    print('eng maxY  ' + '  '.join(f'{r}:fix/iv(n)' for r in RUNGS))
    for e in engines:
        line = f'{e:3d} {MAXY.get(e,0):4d}  '
        for rung in RUNGS:
            fx = [d['fixed_passes:200'] for (rg, ee, s), d in cell.items()
                  if rg == rung and ee == e and 'fixed_passes:200' in d]
            iv = [d['inner_val'] for (rg, ee, s), d in cell.items()
                  if rg == rung and ee == e and 'inner_val' in d]
            line += (f'{st.mean(fx):6.2f}/{st.mean(iv):6.2f}({len(fx)})  '
                     if fx and iv else '     n/a        ')
        print(line)

    print('\n=== N2 vs N3 (MS/SA switch effect), per engine ===')
    for e in engines:
        n2 = [d['fixed_passes:200'] for (rg, ee, s), d in cell.items()
              if rg == 'N2' and ee == e and 'fixed_passes:200' in d]
        n3 = [d['fixed_passes:200'] for (rg, ee, s), d in cell.items()
              if rg == 'N3' and ee == e and 'fixed_passes:200' in d]
        if n2 and n3:
            print(f'  engine {e:2d} (maxY {MAXY.get(e,0):2d}): '
                  f'N2={st.mean(n2):6.2f}  N3={st.mean(n3):6.2f}  '
                  f'N3-N2={st.mean(n3)-st.mean(n2):+7.2f}')

    print('\n=== within-engine seed sd (fixed column) ===')
    for rung in RUNGS:
        sds = []
        for e in engines:
            v = [d['fixed_passes:200'] for (rg, ee, s), d in cell.items()
                 if rg == rung and ee == e and 'fixed_passes:200' in d]
            if len(v) > 1:
                sds.append(st.pstdev(v))
        if sds:
            print(f'{rung:4s} mean_seed_sd={st.mean(sds):5.2f} max={max(sds):5.2f}')


if __name__ == '__main__':
    main()
