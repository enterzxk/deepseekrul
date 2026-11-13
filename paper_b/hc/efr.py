"""M6 (EFR): evidence-gated future degradation rate.

    nu_future(t,k) = nu_age(t,k) * exp( clamp( g_t * a_tk, -a, a ) )

``nu_age`` is B3's learned rate (unchanged); ``q_t`` is the CDE evidence vector
that M2 already uses; ``g_t = sigmoid(gate(q_t))`` is the evidence gate and
``a_tk`` is the per-horizon amplitude.

Why this is a one-variable change on B3: the amplitude head's last layer is
zero-initialised, so at step 0 ``exp(0) = 1`` and the future rates are exactly
B3's. The arrival operator is the reviewed piecewise-linear first passage and is
NOT reimplemented here -- it is called with the corrected rate sequence.

The gate is bounded in (0,1) by the sigmoid and the exponent is clamped to
[-a, a], so the correction can never flip the sign of a rate or blow up: the
worst case multiplies the rate by exp(a) (a=1 -> 2.72x).
"""
import math

import torch
import torch.nn as nn


class EFRCorrection(nn.Module):
    def __init__(self, d_model, q_dim, hidden=32, clip=1.0):
        super().__init__()
        self.clip = float(clip)
        self.q_dim = int(q_dim)
        self.gate = nn.Sequential(nn.Linear(self.q_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))
        # (z_t, D_t, nu_t, q_t, psi(k))
        self.amp = nn.Sequential(nn.Linear(d_model + 2 + self.q_dim + 2, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.amp[2].weight)
        nn.init.zeros_(self.amp[2].bias)

    def forward(self, D_t, nu_t, z_t, q_t, K):
        """(B,) D_t, (B,) nu_t, (B,d) z_t, (B,q) q_t -> (B, K) exponent."""
        B = z_t.shape[0]
        k = torch.arange(1, K + 1, device=z_t.device, dtype=z_t.dtype)
        kk = torch.stack([k / float(K),
                          torch.log1p(k) / math.log1p(float(K))], dim=-1)
        kk = kk.unsqueeze(0).expand(B, -1, -1)
        base = torch.cat([z_t, D_t.unsqueeze(-1), nu_t.unsqueeze(-1), q_t], dim=-1)
        inp = torch.cat([base.unsqueeze(1).expand(-1, K, -1), kk], dim=-1)
        a_tk = self.amp(inp).squeeze(-1)                 # (B, K)
        g_t = torch.sigmoid(self.gate(q_t))              # (B, 1)
        return torch.clamp(g_t * a_tk, -self.clip, self.clip)

    def rates(self, nu_age, D_t, nu_t, z_t, q_t, K):
        return nu_age * torch.exp(self.forward(D_t, nu_t, z_t, q_t, K))
