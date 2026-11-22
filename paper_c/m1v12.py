"""Paper C M1 v1.2: corrected causal baseline + per-condition baseline."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from paper_c.models import GRUModel


class CADFv12(nn.Module):
    """Corrected CADF.

    mode='win': per-cluster causal running mean over the first K0 valid
    occurrences inside the window.
    mode='eng': per-cluster mean of the engine's first K0 cycles (causal).
    mode='static': same baseline as 'win' but no EMA/gate.
    """

    def __init__(self, dim=14, K0=15, mode='win', alpha_init=0.5):
        super().__init__()
        self.K0 = K0
        self.mode = mode
        self.alpha_logit = nn.Parameter(
            torch.full((dim,), math.log(alpha_init / (1.0 - alpha_init))))
        self.proj = nn.Linear(3 * dim, dim)
        self.gate = nn.Linear(3 * dim, dim)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, x, mask, cluster, eng_base=None):
        B, L, D = x.shape
        if self.mode == 'eng':
            c = cluster.clamp(min=0, max=5)
            b = torch.gather(eng_base, 1, c.unsqueeze(-1).expand(-1, -1, D))
        else:
            sums = torch.zeros(B, 6, D, device=x.device)
            counts = torch.zeros(B, 6, device=x.device)
            bs = []
            for t in range(L):
                c = cluster[:, t].clamp(min=0, max=5)
                v = mask[:, t]
                onehot = torch.nn.functional.one_hot(c, 6).float() * v.unsqueeze(-1)
                active = (counts < self.K0).float()
                add = onehot * active
                sums = sums + add.unsqueeze(-1) * x[:, t].unsqueeze(1)
                counts = counts + add
                denom = counts.clamp(min=1).unsqueeze(-1)
                cur = (sums / denom).gather(1, c.view(-1, 1, 1).expand(-1, 1, D)).squeeze(1)
                bs.append(cur)
            b = torch.stack(bs, 1)
        delta = (x - b) * mask.unsqueeze(-1)
        if self.mode == 'static':
            z = torch.cat([x, delta, torch.zeros_like(delta)], -1)
            return x + self.proj(z)
        alpha = torch.sigmoid(self.alpha_logit).view(1, D)
        es = []
        prev = torch.zeros_like(delta[:, 0])
        for t in range(L):
            prev = torch.where(mask[:, t].unsqueeze(-1) > 0,
                               alpha * prev + (1.0 - alpha) * delta[:, t], prev)
            es.append(prev)
        e = torch.stack(es, 1)
        z = torch.cat([x, delta, e], -1)
        g = torch.sigmoid(self.gate(z))
        return x + g * self.proj(z)


class M1v12Model(nn.Module):
    def __init__(self, mode='win', K0=15, input_dim=14, **kw):
        super().__init__()
        self.front = CADFv12(dim=input_dim, K0=K0, mode=mode)
        self.back = GRUModel(input_dim=input_dim, **kw)

    def forward(self, x, mask, cluster, eng_base=None):
        return self.back(self.front(x, mask, cluster, eng_base), mask)


def build_m1v12(name: str, **kw):
    if name == 'm1_v12_win':
        return M1v12Model(mode='win', **kw)
    if name == 'm1_v12_eng':
        return M1v12Model(mode='eng', **kw)
    if name == 'm1_v12_static':
        return M1v12Model(mode='static', **kw)
    raise ValueError(name)
