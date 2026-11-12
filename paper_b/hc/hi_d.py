"""Stage D model skeleton (B2/B3): structured HI on the frozen backbone.

Only two things are new; everything else is the frozen DSPT:

    z_t     = pooled spatio-temporal feature at cycle t  (frozen encoder)
    delta_t = softplus(f_theta(z_t)) >= 0                 (damage increment head)
    D_t     = D_{t-1} + delta_t                           (engine-level recursion)
    nu_t    = EMA(delta)                                  (current rate)
    nu_hat  = softplus(g_phi(D_t, nu_t, z_t, k))          (future rate, B3)
    T       = first_passage(D_t, nu_hat)                  (arrival.py)

B2 replaces the k-dependent rate with the constant nu_t.

The pooled feature z_t is reproduced here because the frozen ``DSPT.forward``
computes it inline and does not expose it. That reproduction is a COPY, and a
copy can drift -- so ``assert_pooling_parity`` checks ``reg_head(z)`` against the
frozen ``forward(...)['prediction']`` on the same batch. If the two ever differ,
the damage head would be reading a different vector than the regression head and
the whole arm would be unfair; the assertion makes that impossible to miss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.dspt import SHORT_SPAN, masked_mean
from .arrival import first_passage
from .model_hc import DSPT_HC


class DamageRateHead(nn.Module):
    """f_theta (increment) and g_phi (future rate). The only new parameters."""

    def __init__(self, d_model, hidden=32, fixed_rate=False):
        super().__init__()
        self.fixed_rate = bool(fixed_rate)
        self.delta = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1))
        # inputs: z_t (d), D_t, nu_t, and the two fixed bases of k
        self.rate = nn.Sequential(nn.Linear(d_model + 2 + 2, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))

    def increment(self, z):
        """(B, L, d) -> (B, L) non-negative per-cycle damage increments."""
        return F.softplus(self.delta(z)).squeeze(-1)

    def future_rates(self, D_t, nu_t, z_t, K):
        """(B,) D_t, (B,) nu_t, (B, d) z_t -> (B, K) predicted increments."""
        B = z_t.shape[0]
        if self.fixed_rate:
            return nu_t.unsqueeze(-1).expand(B, K)
        k = torch.arange(1, K + 1, device=z_t.device, dtype=z_t.dtype)
        kk = torch.stack([k, torch.log1p(k)], dim=-1).unsqueeze(0).expand(B, -1, -1)
        base = torch.cat([z_t, D_t.unsqueeze(-1), nu_t.unsqueeze(-1)], dim=-1)
        inp = torch.cat([base.unsqueeze(1).expand(-1, K, -1), kk], dim=-1)
        return F.softplus(self.rate(inp)).squeeze(-1)


class DSPT_HC_Structured(DSPT_HC):
    """Frozen backbone + structured HI. No bypass regression head is used."""

    def __init__(self, fixed_rate=False, K=200, rate_alpha=0.2, **kw):
        super().__init__(**kw)
        self.head = DamageRateHead(self.d_model, fixed_rate=fixed_rate)
        self.K = int(K)
        self.rate_alpha = float(rate_alpha)

    def pooled_feature(self, x, mask, baseline, cycle, cond):
        """z_t, reproducing the frozen forward's dual-span pooling."""
        h = self.encode(x, baseline, mask, cycle, cond)
        L = h.shape[1]
        dist = torch.arange(L - 1, -1, -1, device=x.device)
        valid = mask.bool()
        short = valid & (dist.unsqueeze(0) < SHORT_SPAN)
        h_A = masked_mean(h, short.to(h.dtype))
        h_B = masked_mean(h, valid.to(h.dtype))
        m = mask.sum(dim=1, keepdim=True)
        g = torch.clamp((m - SHORT_SPAN) / SHORT_SPAN, 0.0, 1.0)
        return g * h_B + (1.0 - g) * h_A

    @torch.no_grad()
    def assert_pooling_parity(self, x, mask, baseline, cycle, cond, tol=1e-5):
        """reg_head(z) must equal the frozen forward's prediction."""
        z = self.pooled_feature(x, mask, baseline, cycle, cond)
        p_copy = self.reg_head(z).reshape(-1)
        p_frozen = self(x, mask=mask, baseline=baseline, cycle=cycle, cond=cond,
                        mode='dual')['prediction'].reshape(-1)
        worst = float((p_copy - p_frozen).abs().max())
        if worst > tol:
            raise AssertionError(
                'pooling copy drifted: reg_head(z) vs frozen forward max|diff|=%.3e' % worst)
        return worst

    def damage_along(self, z):
        """(B, L, d) -> (B, L) cumulative damage along the trajectory."""
        return torch.cumsum(self.head.increment(z), dim=-1)

    def arrival(self, D_t, nu_t, z_t):
        K = self.K
        nu = self.head.future_rates(D_t, nu_t, z_t, K)
        return first_passage(D_t, nu)

    def current_rate(self, delta):
        """Causal EMA of the increment; v_0 = delta_0."""
        a = self.rate_alpha
        out = torch.empty_like(delta)
        v = delta[:, 0]
        out[:, 0] = v
        for t in range(1, delta.shape[1]):
            v = (1.0 - a) * v + a * delta[:, t]
            out[:, t] = v
        return out
