"""Paper C backbones: minimal GRU and causal TCN (no cycle input)."""
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

    def forward(self, x, mask):
        h, _ = self.gru(self.inp(x))
        lengths = mask.sum(1).long().clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        last = h.gather(1, idx).squeeze(1)
        return self.head(self.ln(last)).squeeze(-1)


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


def build(name: str, **kw):
    if name == 'gru':
        return GRUModel(**kw)
    if name == 'tcn':
        return TCNModel(**kw)
    raise ValueError(name)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
