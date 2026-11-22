"""Smoke: one fold x one seed, forward/backward/param delta for the CADF wrapper.

Run:  python -m paper_c.smoke_wrapper
"""
import json

import torch

from paper_c.models import GRUModel
from paper_c.m1v12 import CADFv12
from paper_c.wrapper import (CADFWrapper, K0Policy, assert_identity, build_wrapped,
                             param_delta)

SEED = 101
B, D = 4, 14


def make_batch(L):
    torch.manual_seed(7)
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L)
    mask[:, : L // 6] = 0.0                      # a short warm-up region, as in real data
    cluster = torch.randint(0, 6, (B, L))
    return x, mask, cluster


def run(L, policy):
    x, mask, cluster = make_batch(L)
    bb = lambda: GRUModel(input_dim=D)          # noqa: E731
    cf = lambda: CADFv12(dim=D, K0=policy.resolve(L), mode='win')   # noqa: E731

    wrapped = build_wrapped(bb, cf, seed=SEED, sensor_idx=list(range(D)), n_channels=D,
                            cadf_on=True, k0_policy=policy)
    assert_identity(wrapped, bb, SEED, x, mask, cluster)

    wrapped.train()
    out = wrapped(x, mask, cluster)
    loss = out.pow(2).mean()
    loss.backward()

    grads = {}
    for name, p in wrapped.cadf.named_parameters():
        grads[name] = None if p.grad is None else round(float(p.grad.norm()), 8)
    bb_grad_norms = [float(p.grad.norm()) for p in wrapped.backbone.parameters()
                     if p.grad is not None]

    return dict(
        window=L,
        k0_policy=policy.label(),
        k0_resolved=policy.resolve(L),
        k0_fraction_of_window=round(policy.resolve(L) / L, 3),
        params=param_delta(wrapped),
        out_shape=list(out.shape),
        loss=round(float(loss), 6),
        cadf_grad_norms=grads,
        backbone_params_with_grad=len(bb_grad_norms),
        backbone_grad_all_finite=all(torch.isfinite(torch.tensor(g)).item()
                                     for g in bb_grad_norms),
    )


def main():
    rows = []
    for L in (30, 50):
        for policy in (K0Policy('absolute', 15), K0Policy('fraction', 0.2)):
            rows.append(run(L, policy))
    print(json.dumps(rows, indent=1))

    # the point of the two policies: at L=30 the absolute K0 eats half the window
    a = next(r for r in rows if r['window'] == 30 and r['k0_policy'].startswith('absolute'))
    f = next(r for r in rows if r['window'] == 30 and r['k0_policy'].startswith('fraction'))
    print()
    print(f"L=30: absolute K0={a['k0_resolved']} ({a['k0_fraction_of_window']:.0%} of window) "
          f"vs fraction K0={f['k0_resolved']} ({f['k0_fraction_of_window']:.0%}) "
          f"-> these are different modules, keep them as separate arms")


if __name__ == '__main__':
    main()
