#!/usr/bin/env python3
"""Parameter counts and token layouts per rung, so N2 and N2s are provably equal."""
import sys

sys.path.insert(0, '.')

from paper_b.ncmapss.train import build, load_frozen  # noqa: E402

cfg = load_frozen()
print(f'{"rung":6s} {"params":>8s}  {"token_layout":16s} {"ms":6s} {"sa":6s} dual_span')
for r in ('N0', 'N1', 'N2', 'N2s', 'N3'):
    net = build(r, cfg)
    n = sum(p.numel() for p in net.parameters())
    print(f'{r:6s} {n:8d}  {getattr(net, "token_layout", "-"):16s} '
          f'{getattr(net, "ms_mode", "-"):6s} {getattr(net, "sa_mode", "-"):6s} '
          f'{getattr(net, "dual_span", "-")}')
