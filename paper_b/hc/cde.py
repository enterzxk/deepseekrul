"""M2 CDE: condition-matched degradation evidence.

The hypothesis is narrow and testable: DSPT's cross-sensor attention mixes
operating-condition changes with degradation changes. If the mixing is real,
then an explicit condition-normalised residual, its trend, its persistence and
the confidence in that history should carry information the backbone's pooled
feature z_t does not.

What this is NOT: a renamed normalisation. The frozen pipeline already
normalises by a condition reference (``baseline``); that is the *input*. CDE
builds four statistics of the residual over the observed window and lets a head
predict a correction from them.

Interface (per cycle, one window of ``span`` cycles):

    e_t          = x_t - baseline_t        condition-matched residual, (B, S)
    trend_t      = least-squares slope of e over the observed window, (B, S)
    persistence  = fraction of consecutive steps where e rises, (B, S)
    confidence   = observed cycles / span, (B, 1)

    q_t = proj([e_t, trend_t, persistence_t, confidence_t])   (B, d_q)

The correction head is zero-initialised, so at step 0 the M2 prediction is
bit-identical to B0. That is deliberate: M2 is a one-variable addition to the
strongest arm, and any movement away from B0's number is attributable to the
CDE evidence rather than to a different random head.
"""
import torch
import torch.nn as nn


class CDEEvidence(nn.Module):
    def __init__(self, n_sensors=14, d_out=32):
        super().__init__()
        self.n_sensors = int(n_sensors)
        self.d_out = int(d_out)
        d_in = 3 * self.n_sensors + 1
        self.proj = nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(),
                                  nn.Linear(d_out, d_out))

    def forward(self, x, baseline, mask):
        """x, baseline: (B, span, S); mask: (B, span) -> q: (B, d_out)."""
        B, L, S = x.shape
        m = mask.unsqueeze(-1)                      # (B, L, 1)
        e = (x - baseline) * m                      # condition-matched residual
        n = m.sum(1).clamp_min(1.0)                 # (B, 1)
        e_now = e.sum(1) / n                        # (B, S)

        idx = torch.arange(L, device=x.device, dtype=x.dtype)
        idx = idx.unsqueeze(0).unsqueeze(-1)        # (1, L, 1)
        tbar = (m * idx).sum(1) / n                 # (B, 1, 1)
        cov = (m * (idx - tbar) * e).sum(1)         # (B, S)
        var = (mask * (idx.squeeze(-1) - tbar.squeeze(-1)) ** 2).sum(1, keepdim=True)
        trend = cov / var.clamp_min(1e-6)           # (B, S)

        d = e[:, 1:] - e[:, :-1]                    # (B, L-1, S)
        step = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
        n_step = step.sum(1).clamp_min(1.0)         # (B, 1)
        persistence = ((d > 0).to(d.dtype) * step).sum(1) / n_step   # (B, S)

        confidence = n / float(L)                   # (B, 1)
        q = torch.cat([e_now, trend, persistence, confidence], dim=-1)
        return self.proj(q)


class CDECorrection(nn.Module):
    """Zero-initialised correction from the evidence vector to the RUL."""

    def __init__(self, d_q=32, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_q, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, q):
        return self.net(q).reshape(-1)
