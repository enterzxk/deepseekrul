"""Self-test for gate1.py: prove the gate can both PASS and FAIL, and that each
sub-criterion (EOL guardrail, win rate) can independently veto a method.

Synthetic data only; nothing here touches real results. A checker that has only
ever been seen to pass is not a checker.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

METHODS = ('aidgn', 'two_phase', 'dual_mixer', 'ffve', 'fcstgnn')
SUBSETS = ('FD001', 'FD004')
SEEDS = (101, 202, 303)
CUTS = (0.40, 0.55, 0.70, 0.85, 0.95)
N_ENG = 20


def build(kind):
    """Return (baseline_rows, module_rows) for one scenario."""
    rng = np.random.default_rng(20261122)
    base, mod = [], []
    for method in METHODS:
        for subset in SUBSETS:
            for seed in SEEDS:
                for eng in range(1, N_ENG + 1):
                    life = 180 + 4 * eng
                    for cut in CUTS:
                        y = max(1.0, life * (1.0 - cut))
                        eb = rng.normal(0.0, 6.0)
                        if kind == 'all_good':
                            em = eb * 0.5
                        elif kind == 'three_good':
                            em = eb * (0.5 if method in METHODS[:3] else 1.5)
                        elif kind == 'eol_break':
                            # Better everywhere except the EOL cut, where the
                            # absolute error must be genuinely larger: halving
                            # then adding 3 still beats the baseline
                            # (E|N(3,3)|=3.8 < E|N(0,6)|=4.8), so the offset has
                            # to be big enough to cross over.
                            em = eb * 0.5 if cut < 0.95 else eb * 0.5 + 8.0
                        elif kind == 'win_rate_break':
                            # mean RMSE better, but only half the engines improve
                            em = eb * (0.2 if eng % 2 == 0 else 1.4)
                        else:
                            raise SystemExit(kind)
                        for store, pred in ((base, y + eb), (mod, y + em)):
                            store.append(dict(method=method, subset=subset, seed=seed,
                                              engine=eng, cut=cut, y=y, pred=pred))
    return base, mod


def run(kind):
    base, mod = build(kind)
    with tempfile.TemporaryDirectory() as td:
        bp, mp = Path(td) / 'base.json', Path(td) / 'mod.json'
        bp.write_text(json.dumps(base))
        mp.write_text(json.dumps(mod))
        r = subprocess.run([sys.executable, 'paper_c/gate1.py',
                            '--baseline', str(bp), '--module', str(mp),
                            '--json', str(Path(td) / 'out.json')],
                           capture_output=True, text=True)
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.startswith(('GATE', 'backbones'))]
        return r.returncode, tail


EXPECT = {
    'all_good': (0, 'PASS'),
    'three_good': (2, 'FAIL'),
    'eol_break': (2, 'FAIL'),
    'win_rate_break': (2, 'FAIL'),
}

ok = True
for kind, (want_rc, want_verdict) in EXPECT.items():
    rc, tail = run(kind)
    got = tail[-1].split(':')[-1].strip() if tail else '?'
    good = (rc == want_rc) and (got == want_verdict)
    ok &= good
    print(f'{kind:16} expect {want_verdict:4} rc={want_rc}  got {got:4} rc={rc}  '
          f'{"OK" if good else "MISMATCH"}   {tail}')

print('\nSELFTEST', 'PASSED' if ok else 'FAILED')
raise SystemExit(0 if ok else 1)
