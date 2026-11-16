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


def covariance_from_tokens(chunks):
    H = torch.cat(chunks, dim=0)            # (B_total, L, N, D)
    B, L, N, D = H.shape
    X = H.reshape(-1, N, D).numpy()         # (S, N, D)
    C = np.zeros((N, N), dtype=np.float64)
    for d in range(D):
        C += np.cov(X[:, :, d], rowvar=False)
    return C / D, (B, L, N, D)


def rho_of(C, r, U=None):
    vals = np.linalg.eigvalsh(C)[::-1]
    total = vals.sum()
    if U is None:
        return float(vals[:r].sum() / total), vals
    return float(np.trace(U.T @ C @ U) / total), vals


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
                rho, vals = rho_of(C, r)
                null = []
                for _ in range(args.n_null):
                    Q, _ = np.linalg.qr(rng.standard_normal((14, r)))
                    null.append(float(np.trace(Q.T @ C @ Q) / vals.sum()))
                null = np.array(null)
                row[f'rho_r{r}'] = rho
                row[f'null_r{r}_p95'] = float(np.percentile(null, 95))
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
