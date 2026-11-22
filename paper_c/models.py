"""Paper C backbones and M1 CADF (causal adaptive degradation front-end)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUModel(nn.Module):
    def __init__(self, input_dim=14, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout)
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def seq_features(self, x, mask):
        h, _ = self.gru(self.inp(x))
        return h

    def features(self, x, mask):
        h = self.seq_features(x, mask)
        lengths = mask.sum(1).long().clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(1, idx).squeeze(1)

    def forward(self, x, mask):
        return self.head(self.ln(self.features(x, mask))).squeeze(-1)


class CausalBlock(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        y = F.pad(x, (self.pad, 0))
        y = self.conv(y)
        y = F.relu(y)
        y = self.drop(y)
        y = y + x
        return self.norm(y.transpose(1, 2)).transpose(1, 2)


class TCNModel(nn.Module):
    def __init__(self, input_dim=14, ch=64, blocks=4, kernel=3, dropout=0.2):
        super().__init__()
        self.inp = nn.Conv1d(input_dim, ch, 1)
        self.blocks = nn.ModuleList([
            CausalBlock(ch, kernel, dilation=2 ** i, dropout=dropout)
            for i in range(blocks)
        ])
        self.head = nn.Linear(ch, 1)

    def forward(self, x, mask):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        lengths = mask.sum(1).long().clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(1))
        last = h.transpose(1, 2).gather(1, idx).squeeze(1)
        return self.head(last).squeeze(-1)


class CADF(nn.Module):
    """Causal adaptive degradation front-end.

    b_t is the mean of the first min(t+1, K0) valid cycles; delta = x - b;
    a learnable per-sensor EMA smooths delta; a zero-init residual gate fuses
    [x, delta, ema] back to the input space.
    """

    def __init__(self, dim=14, K0=15, static=False, alpha_init=0.5):
        super().__init__()
        self.K0 = K0
        self.static = static
        import math
        a0 = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.full((dim,), a0))
        self.proj = nn.Linear(3 * dim, dim)
        self.gate = nn.Linear(3 * dim, dim)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, x, mask):
        B, L, D = x.shape
        xm = x * mask.unsqueeze(-1)
        cs = torch.cumsum(xm, 1)
        idx = torch.arange(L, device=x.device)
        gidx = torch.minimum(idx, torch.tensor(self.K0 - 1, device=x.device))
        gidx = gidx.view(1, L, 1).expand(B, L, D)
        denom = torch.minimum(idx + 1, torch.tensor(self.K0, device=x.device)).float()
        b = cs.gather(1, gidx) / denom.view(1, L, 1)
        delta = x - b
        if self.static:
            z = torch.cat([x, delta, torch.zeros_like(delta)], -1)
            return x + self.proj(z)
        alpha = torch.sigmoid(self.alpha_logit).view(1, D)
        es = []
        prev = torch.zeros_like(delta[:, 0])
        for t in range(L):
            prev = alpha * prev + (1.0 - alpha) * delta[:, t]
            es.append(prev)
        e = torch.stack(es, 1)
        z = torch.cat([x, delta, e], -1)
        g = torch.sigmoid(self.gate(z))
        return x + g * self.proj(z)


class M1Model(nn.Module):
    def __init__(self, static=False, K0=15, alpha_init=0.5, **kw):
        super().__init__()
        self.front = CADF(static=static, K0=K0, alpha_init=alpha_init)
        self.back = GRUModel(**kw)

    def forward(self, x, mask):
        return self.back(self.front(x, mask), mask)


def build(name: str, **kw):
    if name == 'gru':
        return GRUModel(**kw)
    if name == 'gru_cap':
        return GRUModel(hidden=64, layers=3, **kw)
    if name == 'tcn':
        return TCNModel(**kw)
    if name == 'm1':
        return M1Model(static=False, **kw)
    if name == 'm1_static':
        return M1Model(static=True, **kw)
    if name == 'm1_k5':
        return M1Model(static=False, K0=5, **kw)
    if name == 'm1_k10':
        return M1Model(static=False, K0=10, **kw)
    if name == 'm1_a8':
        return M1Model(static=False, alpha_init=0.8, **kw)
    raise ValueError(name)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
