"""Self-test for any implementation of the E5 spectrum / principal-angle math.

Runs three synthetic cases whose answers are known in closed form, so a silent
binding error (the kind that still returns a plausible number) cannot pass:

  isotropic H          -> rho == r/14 exactly in expectation (0.286 for r=4)
  rank-4 H with true U -> rho(U) ~= 1.0, rho(random) ~= 4/14
  random 4-dim subspaces -> null 95th percentile ~= 0.41-0.43 for N=14
  principal angle      -> 0 deg for the same basis, 90 deg for an exact
                          orthogonal complement

Usage (point it at whichever module implements chat/rho_of/principal_angle):
  python3 scripts/audit_spectrum_selftest.py --module scripts/probe_spectrum.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def load_module(path):
    spec = importlib.util.spec_from_file_location('under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', required=True)
    a = ap.parse_args()
    m = load_module(a.module)
    need = ('chat', 'rho_of', 'principal_angle', 'random_rho')
    missing = [f for f in need if not hasattr(m, f)]
    if missing:
        print('MISSING functions:', missing)
        return 2

    gen = torch.Generator().manual_seed(0)
    fails = []

    def check(name, got, want, tol):
        ok = abs(got - want) <= tol
        print(f'{"PASS" if ok else "FAIL"}  {name}: {got:.4f} (want {want:.4f} +- {tol})')
        if not ok:
            fails.append(name)

    H = torch.randn(64, 70, 14, 64, generator=gen)
    C = m.chat(H)
    if tuple(C.shape) != (14, 14):
        print('FAIL  chat() shape', tuple(C.shape), 'expected (14,14)')
        return 1
    q, _ = torch.linalg.qr(torch.randn(14, 4, generator=gen, dtype=torch.float64))
    check('rho on isotropic H', m.rho_of(C, q), 4 / 14, 0.03)
    # The isotropic null is a THEORETICAL constant (4/14, sd ~1e-4), so it is
    # the only cross-implementation anchor: two implementations agree iff both
    # return 0.2857 here. A null95 near 0.41 can only come from an anisotropic
    # C and must never be quoted as evidence that two implementations match.
    check('null95 on isotropic H (theory-anchored)', m.random_rho(C, 4, 200, gen)[0], 4 / 14, 0.01)

    # ---- centring: covariance vs raw second moment -------------------------
    # This is a GUARD, not a discriminator between the two current
    # implementations: both centre correctly (verified on real E0 H -- rho agrees
    # to 5.6e-16), so both pass. It exists because a future implementation that
    # forgets to subtract the per-sensor mean would agree bit-for-bit on every
    # other case here (all zero-mean) and still be wrong on real H.
    #
    # H = mu + tiny noise separates the two definitions: a centred covariance is
    # ~isotropic noise, so any 4-dim subspace holds ~4/14; an uncentred second
    # moment is dominated by mu, so a 4-dim subspace containing mu holds ~1.0.
    # E5_PREREG.md 3 H1 says "covariance", so 4/14 is the expected value.
    mu = torch.randn(14, generator=gen, dtype=torch.float64)
    mu = mu / mu.norm()
    H3 = (mu.float().view(1, 1, 14, 1).repeat(64, 70, 1, 64)
          + 0.01 * torch.randn(64, 70, 14, 64, generator=gen))
    U4 = torch.linalg.qr(torch.cat([mu.view(14, 1),
                                    torch.eye(14, dtype=torch.float64)[:, :3]], dim=1))[0]
    check('chat is a covariance (not a raw second moment)', m.rho_of(m.chat(H3), U4), 4 / 14, 0.03)

    U0, _ = torch.linalg.qr(torch.randn(14, 4, generator=gen, dtype=torch.float64))
    lat = torch.randn(64, 70, 4, generator=gen)
    H2 = (torch.einsum('nr,blr->bln', U0.float(), lat).unsqueeze(-1).repeat(1, 1, 1, 64)
          + 0.01 * torch.randn(64, 70, 14, 64, generator=gen))
    C2 = m.chat(H2)
    check('rho on rank-4 H, true U', m.rho_of(C2, U0), 1.0, 0.01)
    # Average over many random bases: a single draw has a spread of ~0.03 and
    # made this check flaky (0.3120 vs 0.3367 on identical code).
    vals = []
    for _ in range(200):
        qq, _ = torch.linalg.qr(torch.randn(14, 4, generator=gen, dtype=torch.float64))
        vals.append(m.rho_of(C2, qq))
    check('rho on rank-4 H, mean over 200 random U', float(np.mean(vals)), 4 / 14, 0.02)
    # Regression anchor only: this value is a property of THIS synthetic C,
    # not a theoretical constant, so it cannot be compared across data sets.
    check('null95 on rank-4 H (regression anchor)', m.random_rho(C2, 4, 200, gen)[0], 0.42, 0.05)
    check('angle(U, U)', m.principal_angle(U0, U0), 0.0, 0.5)
    # a true orthogonal complement: extend U0 into a full basis and take the
    # next four columns. Q[:, 4:8] of an unrelated random basis is NOT
    # orthogonal to U0, and would legitimately give ~86 deg.
    Qf = torch.linalg.qr(torch.cat([U0, torch.randn(14, 10, generator=gen,
                                                     dtype=torch.float64)], dim=1))[0]
    check('angle(U, exact orthogonal complement)', m.principal_angle(U0, Qf[:, 4:8]), 90.0, 0.5)

    print('\nVERDICT', 'PASS' if not fails else 'FAIL: ' + ', '.join(fails))
    return 0 if not fails else 1


if __name__ == '__main__':
    raise SystemExit(main())
