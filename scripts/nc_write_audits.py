#!/usr/bin/env python3
"""Write the per-subset audit.json for all four N-CMAPSS subsets, then print a table."""
import glob
import json
import sys

sys.path.insert(0, '.')

from paper_b.ncmapss.data import official_split          # noqa: E402
from paper_b.ncmapss.run_loeo import _write_subset_audit  # noqa: E402

for s in ('DS02', 'DS05', 'DS06', 'DS07'):
    p = glob.glob(f'data/NCMAPSS/extracted/N-CMAPSS_{s}*.h5')[0]
    dev, test = official_split(p)
    _write_subset_audit(p, s, f'experiments/ncmapss/{s}', dev, test)

print('\n--- audit summary ---')
for f in sorted(glob.glob('experiments/ncmapss/DS*/audit.json')):
    a = json.load(open(f))
    d, t = a['splits']['dev'], a['splits']['test']
    print(f"{a['subset']}: dev={a['dev_units']} flights={d['n_flights']} "
          f"max_rul={d['max_rul']:.0f} | test={a['test_units']} "
          f"flights={t['n_flights']} max_rul={t['max_rul']:.0f} | "
          f"capped_active={a['capped_active']}")
