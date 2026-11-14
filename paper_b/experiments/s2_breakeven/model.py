"""S2: DSPT with a one-dimensional total-life auxiliary input.

The auxiliary value is concatenated to the pooled representation immediately
before the regression head. Everything else is the frozen F_full_basic_hi1
architecture: nothing else in the network changes, so a difference between
configurations is attributable to the auxiliary input alone.

The class subclasses the frozen DSPT and re-implements ``forward`` only to
insert the concat; the encoder, switches and loss stay untouched.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from paper_b.models.dspt import SHORT_SPAN, DSPT, masked_mean


class DSPTWithLife(DSPT):
    """Frozen DSPT whose regression head also sees one scalar."""

    def __init__(self, *args, d_model: int = 64, **kwargs):
        super().__init__(*args, d_model=d_model, **kwargs)
        # same two-layer shape as the original head, one extra input column
        self.reg_head = nn.Sequential(nn.Linear(d_model + 1, d_model), nn.GELU(),
                                      nn.Linear(d_model, 1))

    def _head(self, h, life):
        if life is None:
            return self.reg_head(torch.cat([h, torch.zeros(h.size(0), 1, device=h.device,
                                                          dtype=h.dtype)], dim=-1))
        return self.reg_head(torch.cat([h, life.reshape(-1, 1)], dim=-1))

    def forward(self, x, mask=None, baseline=None, cycle=None, cond=None, mode='dual', life=None):
        B, L, N = x.shape
        if mask is None:
            mask = torch.ones(B, L, device=x.device)
        if baseline is None:
            baseline = x[:, 0]
        if cycle is None:
            cycle = torch.zeros(B, device=x.device)
        h = self.encode(x, baseline, mask, cycle, cond)
        dist = torch.arange(L - 1, -1, -1, device=x.device)
        valid = mask.bool()
        short = valid & (dist.unsqueeze(0) < SHORT_SPAN)
        h_A = masked_mean(h, short.to(h.dtype))
        h_B = masked_mean(h, valid.to(h.dtype))
        m = mask.sum(dim=1, keepdim=True)
        g = torch.clamp((m - SHORT_SPAN) / SHORT_SPAN, 0.0, 1.0)
        h_f = g * h_B + (1.0 - g) * h_A

        out = {
            'y_A': self._head(h_A, life),
            'y_B': self._head(h_B, life),
            'prediction': (self._head(h_f, life) if mode == 'dual'
                           else self._head(h_B, life) if mode == 'long'
                           else self._head(h_A, life)),
            'gate': g,
            'embedding': h_f,
        }
        if self.hi_source != 'none':
            hsel = {'hf': h_f, 'hB': h_B, 'hA': h_A}[self.hi_source]
            out['hi'] = self.hi_head(hsel).reshape(-1)
            out['hi_seq'] = self.hi_head(h).squeeze(-1)
        return out
