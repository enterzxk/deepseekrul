#!/usr/bin/env python3
"""Read-only spectrum probe for H1: how much sensor variance does a rank-4 subspace hold?

The E5 pre-registration needs a defensible threshold for `rho`, the fraction of
per-sensor variance captured by a rank-r subspace. Rather than guess, measure it
on an existing E0 checkpoint before freezing.

`tokens` at the point `sensor_score` is applied has shape (B, L, N, D) -- that is
exactly H. A forward pre-hook on `sensor_score` captures it without touching the
model or the h8 fingerprint.

Definition of the covariance (frozen here so the threshold means something):

    for each feature dim d: C_d = cov over all (b, l) of the 14-dim vector H[b,l,:,d]
    C_hat = mean_d C_d                     (14x14)
    eigenvalues lambda_1 >= ... >= lambda_14
    rho_r = sum_{i<=r} lambda_i / sum_i lambda_i

Null distribution: 1000 random orthonormal rank-r subspaces, same statistic.

Usage
-----
    python scripts/probe_spectrum.py --runs <runs> --arm E0 --split-seed 17 \
        --seeds 101 --subsets FD001,FD002,FD003,FD004 --out spectrum.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paper_b.sensor_aug.evaluation import _predict  # noqa: E402
from paper_b.training.data import prepare_train  # noqa: E402
from paper_b.training.train import _build  # noqa: E402
from scripts._cells import resolve  # noqa: E402


class TokenTap:
    """Capture the (B,L,N,D) tensor handed to sensor_score."""

    def __init__(self):
        self.chunks = []
        self.handle = None

    def attach(self, net):
        for name, mod in net.named_modules():
            if name.endswith('sensor_score'):
                self.handle = mod.register_forward_pre_hook(self._hook)
                return True
        return False

    def _hook(self, _module, inputs):
        t = inputs[0]
        if t.dim() == 4:
            self.chunks.append(t.detach().float().cpu())

    def close(self):
        if self.handle is not None:
            self.handle.remove()


def chat(h):
    """Per-sensor covariance averaged over feature dims. Canonical name shared
    with scripts/audit_spectrum_selftest.py so one closed-form test can check
    both implementations.

    h: (B, L, N, D) torch tensor or ndarray. Returns (N, N) float64.
    """
    if hasattr(h, 'detach'):
        h = h.detach().float().cpu().numpy()
    h = np.asarray(h, dtype=np.float64)
    _, _, N, D = h.shape
    x = h.reshape(-1, N, D)
    C = np.zeros((N, N), dtype=np.float64)
    for d in range(D):
        C += np.cov(x[:, :, d], rowvar=False)
    return C / D


def covariance_from_tokens(chunks):
    return chat(torch.cat(chunks, dim=0)), tuple(torch.cat(chunks, dim=0).shape)


def rho_of(C, U):
    """Variance fraction captured by the given subspace U (N, r)."""
    if hasattr(U, 'detach'):
        U = U.detach().double().cpu().numpy()
    U = np.asarray(U, dtype=np.float64)
    total = float(np.linalg.eigvalsh(C).sum())
    return float(np.trace(U.T @ C @ U) / total)


def rho_optimal(C, r):
    vals = np.linalg.eigvalsh(C)[::-1]
    return float(vals[:r].sum() / vals.sum()), vals


def random_rho(C, r, n, gen=None):
    """95th percentile and the whole null of rho for random rank-r subspaces.

    Note the draw shape: torch.linalg.qr batches over the LAST TWO dims, so the
    draws must be (n, 14, r) to get n independent (14, r) orthonormal bases.
    Drawing (14, r, n) instead makes it factorise n matrices of shape (r, n) and
    the resulting "bases" are not orthonormal -- which yields rho > 1.
    """
    if gen is not None:
        draws = torch.randn(n, 14, r, generator=gen, dtype=torch.float64)
        Q = torch.linalg.qr(draws)[0].numpy()                    # (n, 14, r)
    else:
        rng = np.random.default_rng(20261116)
        Q = np.stack([np.linalg.qr(rng.standard_normal((14, r)))[0]
                      for _ in range(n)], axis=0)                # (n, 14, r)
    total = float(np.linalg.eigvalsh(C).sum())
    # Explicit loop: n <= a few thousand 14x14 matmuls is instant, and this form
    # has no index-labeling to get wrong. (A three-operand einsum with a repeated
    # batch index does not broadcast the way it reads.)
    null = np.array([float(np.trace(Q[k].T @ C @ Q[k]) / total)
                     for k in range(Q.shape[0])])
    return float(np.percentile(null, 95)), null


def principal_angle(U, V):
    """Largest principal angle in degrees between two subspaces."""
    if hasattr(U, 'detach'):
        U = U.detach().double().cpu().numpy()
    if hasattr(V, 'detach'):
        V = V.detach().double().cpu().numpy()
    q1 = np.linalg.qr(np.asarray(U, dtype=np.float64))[0]
    q2 = np.linalg.qr(np.asarray(V, dtype=np.float64))[0]
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--arm', default='E0')
    ap.add_argument('--split-seed', type=int, default=17)
    ap.add_argument('--seeds', default='101')
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    ap.add_argument('--ranks', default='4,14')
    ap.add_argument('--n-null', type=int, default=1000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cpu', action='store_true', default=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    rng = np.random.default_rng(20261116)
    seeds = [int(s) for s in args.seeds.split(',')]
    subsets = args.subsets.split(',')
    ranks = [int(r) for r in args.ranks.split(',')]

    results = []
    for seed in seeds:
        for subset in subsets:
            cell = Path(resolve(args.runs, args.arm, subset, seed, args.split_seed))
            cfg = json.loads((cell / 'config.json').read_text())
            net = _build(cfg)
            sd = torch.load(cell / 'best.pt', map_location='cpu', weights_only=True)
            net.load_state_dict(sd)
            net.eval()

            tap = TokenTap()
            if not tap.attach(net):
                raise SystemExit(f'{cell}: no sensor_score module to hook '
                                 '(sa_mode without sensor attention?)')
            ds = prepare_train(cfg['subset'], Path(args.data_dir), span=cfg['span'],
                               split_seed=cfg['split_seed'], fold=cfg['fold'],
                               val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'],
                               n_cond=cfg['n_cond'], cuts=tuple(cfg['cuts']),
                               legacy_rul=cfg['legacy_rul'], global_hi=cfg['global_hi'])
            va = ds['val']
            # Reuse the evaluation helper so the batching convention matches
            # the one the reported metrics were produced with; the hook fires
            # inside it.
            _predict(net, va, 'cpu', cfg)
            tap.close()

            C, shape = covariance_from_tokens(tap.chunks)
            row = {'cell': str(cell), 'subset': subset, 'seed': seed,
                   'token_shape': list(shape), 'sa_mode': cfg.get('sa_mode')}
            for r in ranks:
                rho, vals = rho_optimal(C, r)
                p95, null = random_rho(C, r, args.n_null)
                row[f'rho_r{r}'] = rho
                row[f'null_r{r}_p95'] = p95
                row[f'null_r{r}_mean'] = float(null.mean())
            row['eigenvalues'] = [float(v) for v in vals]
            results.append(row)
            print(f"{subset} s{seed}: " + '  '.join(
                f"rho_r{r}={row[f'rho_r{r}']:.4f} (null95={row[f'null_r{r}_p95']:.4f})"
                for r in ranks), flush=True)

    Path(args.out).write_text(json.dumps({'definition': 'see module docstring',
                                          'results': results}, indent=1))
    print(f'[write] {args.out} with {len(results)} cells')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
