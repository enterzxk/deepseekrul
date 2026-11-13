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
import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.dspt import SHORT_SPAN, masked_mean
from .arrival import first_passage
from .model_hc import DSPT_HC


def mean_life_bias(l_mean):
    """b0 with softplus(b0) == 1 / l_mean, i.e. b0 = log(e^{1/L} - 1).

    Why the head needs this: PyTorch's default bias is 0, so a freshly built
    increment head emits softplus(0) = 0.693 per cycle. With D_0 = 0 that puts
    every engine past the threshold within two cycles, and the arrival operator
    returns 0 for ~98% of samples that have not failed yet (measured; see
    STAGE_D_ACCEPTANCE.md section 2). The defect is a 138x scale error in the
    increment, not a convergence problem, so it is fixed in the initialization.

    The mean life comes from the TRAINING engines of the frozen split only --
    never from validation or test lifetimes.
    """
    r = 1.0 / float(l_mean)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError('l_mean must be a positive finite lifetime, got %r' % (l_mean,))
    return float(math.log(math.expm1(r)))


class DamageRateHead(nn.Module):
    """f_theta (increment) and g_phi (future rate). The only new parameters."""

    def __init__(self, d_model, hidden=32, fixed_rate=False, l_mean=None,
                 anchored=False, anchor_clip=1.0):
        super().__init__()
        self.fixed_rate = bool(fixed_rate)
        # Age-anchored increment (M3-B): delta_t = lambda0 * exp(clamp(u(z_t))),
        # lambda0 = 1/L_mean of the TRAINING split. The last layer is zero-init,
        # so at step 0 u == 0 and delta == lambda0 exactly -- the damage path is
        # then D_t = t/L_mean, i.e. the age timer. That is the whole point of the
        # arm: whatever the sensor modulation buys has to show up as a departure
        # from that timer, and if it never departs then the structured HI is just
        # re-parameterising cycle and the route is dead.
        self.anchored = bool(anchored)
        self.anchor_clip = float(anchor_clip)
        self.anchor_lambda0 = None
        self.delta = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1))
        if self.anchored:
            if l_mean is None:
                raise ValueError('anchored increment needs l_mean (training split)')
            self.anchor_lambda0 = 1.0 / float(l_mean)
            self.anchor = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                        nn.Linear(hidden, 1))
            nn.init.zeros_(self.anchor[2].weight)
            nn.init.zeros_(self.anchor[2].bias)
        # inputs: z_t (d), D_t, nu_t, and the two fixed bases of k
        self.rate = nn.Sequential(nn.Linear(d_model + 2 + 2, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))
        # Both heads emit a per-cycle damage RATE, so both take the same prior:
        # at init the future rate is 1/L_mean per cycle, which is also what the
        # EMA of the observed increments starts at. B2 (fixed rate) and B3
        # (learned rate) therefore coincide at step 0 and differ only in whether
        # the rate may vary -- which is the variable the ablation is about.
        self.init_l_mean = None if l_mean is None else float(l_mean)
        self.init_b0 = None if l_mean is None else mean_life_bias(l_mean)
        if self.init_b0 is not None:
            nn.init.constant_(self.delta[2].bias, self.init_b0)
            # The rate head's last layer starts at zero, so at init the predicted
            # future rate is exactly softplus(b0) = 1/L_mean and B3 coincides
            # with B2. This is not cosmetic: with a random last layer the rate
            # head's output at init is 7.3e-8 instead of 4.8e-3 (measured), the
            # arrival is pinned at K for every sample and the whole arm gets
            # exactly zero gradient. Gradient still reaches the head afterwards:
            # dL/dW_out = dL/dpre * h1 with h1 != 0, so only the first layer
            # waits one step for its gradient.
            nn.init.zeros_(self.rate[2].weight)
            nn.init.constant_(self.rate[2].bias, self.init_b0)

    def init_report(self):
        """Everything an auditor needs to re-create this head bit for bit."""
        return dict(l_mean=self.init_l_mean, b0=self.init_b0,
                    softplus_b0=None if self.init_b0 is None
                    else float(F.softplus(torch.tensor(self.init_b0))),
                    fingerprint=self.fingerprint())

    def fingerprint(self):
        """sha256[:16] over the head's own parameters, float32 little-endian."""
        h = hashlib.sha256()
        for k, v in sorted(self.state_dict().items()):
            h.update(k.encode())
            h.update(np.ascontiguousarray(v.detach().cpu().numpy(),
                                          dtype='<f4').tobytes())
        return h.hexdigest()[:16]

    def increment(self, z):
        """(B, L, d) -> (B, L) non-negative per-cycle damage increments."""
        if self.anchored:
            u = self.anchor(z).squeeze(-1)
            return self.anchor_lambda0 * torch.exp(
                torch.clamp(u, -self.anchor_clip, self.anchor_clip))
        return F.softplus(self.delta(z)).squeeze(-1)

    def future_rates(self, D_t, nu_t, z_t, K):
        """(B,) D_t, (B,) nu_t, (B, d) z_t -> (B, K) predicted increments."""
        B = z_t.shape[0]
        if self.fixed_rate:
            return nu_t.unsqueeze(-1).expand(B, K)
        k = torch.arange(1, K + 1, device=z_t.device, dtype=z_t.dtype)
        # Basis psi(k) = (k, log(1+k)) as specified, but divided by its value at
        # K so every coordinate lands in [0, 1]. Feeding raw k next to
        # standardised z makes the first linear layer explode: measured at
        # K=600, hidden pre-activation range [-69, +74] and rate output 7.3e-8,
        # i.e. 1/65000 of 1/L_mean. The span of psi is unchanged; only the
        # numerical scale is, and the scale is what the linear layer sees.
        kk = torch.stack([k / float(K),
                          torch.log1p(k) / math.log1p(float(K))], dim=-1)
        kk = kk.unsqueeze(0).expand(B, -1, -1)
        base = torch.cat([z_t, D_t.unsqueeze(-1), nu_t.unsqueeze(-1)], dim=-1)
        inp = torch.cat([base.unsqueeze(1).expand(-1, K, -1), kk], dim=-1)
        return F.softplus(self.rate(inp)).squeeze(-1)


class DSPT_HC_Structured(DSPT_HC):
    """Frozen backbone + structured HI. No bypass regression head is used."""

    def __init__(self, fixed_rate=False, K=200, rate_alpha=0.2, l_mean=None,
                 head_seed=None, anchored=False, anchor_clip=1.0, **kw):
        super().__init__(**kw)
        # The new head must be reproducible on its own: loading the shared
        # backbone overwrites every backbone tensor but leaves head.* at whatever
        # the global RNG happened to be, so two probe runs would disagree. Seed
        # around the head construction only, then restore, so the head init does
        # not depend on the backbone's construction order.
        if head_seed is None:
            self.head = DamageRateHead(self.d_model, fixed_rate=fixed_rate,
                                       l_mean=l_mean, anchored=anchored,
                                       anchor_clip=anchor_clip)
        else:
            cpu_state = torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(int(head_seed))
            self.head = DamageRateHead(self.d_model, fixed_rate=fixed_rate,
                                       l_mean=l_mean, anchored=anchored,
                                       anchor_clip=anchor_clip)
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
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
