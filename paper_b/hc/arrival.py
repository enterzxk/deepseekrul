"""Stage D arrival operator: exact truncated first-passage time (spec D3.1 v3).

This is the ONLY implementation. The tests call it; nothing re-implements it in
numpy. The reviewer's reason is concrete: the earlier version had one numpy copy
inside the test file and the production code would have had a second, and the
two would drift the first time either was touched.

Index convention, stated once and used everywhere:

    K predicted intervals, K+1 nodes D[0..K]
    D[0] = current damage, D[k+1] = D[k] + nu[k]
    T = min(first passage, K) in [0, K]; not hit within K returns K

The operator (for a monotone path, exactly one interval crosses the threshold):

    frac_k = clip((1 - D[k]) / (D[k+1] - D[k]), 0, 1)   if D[k+1] > D[k]
    frac_k = 1[D[k] < 1]                                if D[k+1] == D[k]
    T      = sum_k frac_k

so T = j + (1 - D[j]) / nu[j] for a crossing inside interval j. This is an
identity for piecewise-linear paths, not a smooth surrogate: the hazard-product
form was capped at 1+e cycles, and the rate-temperature form let survival
rebound under deceleration.

Gradient, with the crossing inside interval j (zero-based interval index):

    dT/dnu_i = -1/nu[j]     for i <  j   (equal, non-decaying)
    dT/dnu_j = -f/nu[j]     with f = frac_j
    dT/dnu_i =  0           for i >  j

and dT/dd0 = -1/nu[j] for an interior crossing. clip's derivative is 1 on the
boundary, so an exact node hit still carries gradient; only strict saturation
(clamped away from 0/1) gives exactly zero.

Zero-gradient trap: if sum(nu) <= 1 - d0 the path never reaches 1, T == K for
every parameter setting and the sample carries no gradient at all. The damage
rate head must be initialised so that nu * K > 1 over the horizon it is used.
"""
import torch


def damage_path(d0, nu):
    """(B,) current damage + (B, K) increments -> (B, K+1) monotone path."""
    d0 = d0.reshape(-1, 1)
    return torch.cat([d0, d0 + torch.cumsum(nu, dim=-1)], dim=-1)


def first_passage(d0, nu):
    """Exact truncated first-passage time, in cycles. Returns (B,).

    ``d0`` is the damage at the prediction time (already accumulated, not zero),
    ``nu`` the predicted per-cycle increments. Both are (B,) / (B, K).
    """
    if nu.dim() != 2:
        raise ValueError('nu must be (B, K), got %s' % (tuple(nu.shape),))
    D = damage_path(d0, nu)
    K = D.shape[-1] - 1
    lo, hi = D[:, :-1], D[:, 1:]
    d = hi - lo
    rising = d > 0
    safe = torch.where(rising, d, torch.ones_like(d))
    frac = torch.where(rising,
                       ((1.0 - lo) / safe).clamp(0.0, 1.0),
                       (lo < 1.0).to(D.dtype))
    return frac.sum(dim=-1).clamp(0.0, float(K))


def hard_first_passage(d0, nu):
    """Non-differentiable reference used only for the D-G4 consistency check."""
    D = damage_path(d0, nu)
    K = D.shape[-1] - 1
    hit = D >= 1.0
    idx = torch.where(hit.any(dim=-1), hit.float().argmax(dim=-1),
                      torch.full_like(D[:, 0], K, dtype=torch.long))
    return idx.to(D.dtype)
