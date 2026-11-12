"""Independent Vanilla Transformer baseline (Paper B).

BASE: normalised sensor sequence -> Linear embedding -> fixed sinusoidal PE
      -> 2 x TransformerEncoderLayer (padding mask) -> masked mean pool -> RUL head.
No DSPT module is used: no 3-channel features, no absolute cycle, no conv,
no cross-sensor attention, no learned sensor pooling, no dual-span gate.

Controls for fair attribution (not replacements for DSPT):
  input_mode='dsp'  same 3-channel raw/deviation/causal-EMA representation
  use_cycle=True    same absolute-cycle Fourier encoding
These two switches let us separate "feature representation" from
"architecture" when comparing with DSPT.
"""
import math

import torch
import torch.nn as nn

from .dspt import masked_mean


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, h):
        return h + self.pe[:, :h.size(1)]


class VanillaTransformer(nn.Module):
    dual_span = False

    def __init__(self, num_sensors=14, d_model=64, num_layers=2, dropout=0.2,
                 max_span=70, input_mode='raw', use_cycle=False, n_feat_cycle=4):
        super().__init__()
        assert input_mode in ('raw', 'dsp')
        self.input_mode = input_mode
        self.use_cycle = use_cycle
        self.d_model = d_model
        in_dim = num_sensors if input_mode == 'raw' else 3 * num_sensors
        self.embed = nn.Linear(in_dim, d_model)
        self.pe = SinusoidalPE(d_model, max_len=max(256, max_span))
        if use_cycle:
            self.register_buffer('periods', torch.tensor((25.0, 50.0, 125.0, 250.0)))
            self.cycle_proj = nn.Linear(2 * n_feat_cycle, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.reg_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, 1))

    def _dsp_features(self, x, baseline, mask):
        dev = x - baseline.unsqueeze(1)
        valid = mask.bool()
        first = valid.float().argmax(dim=1)
        ema = x[torch.arange(x.size(0), device=x.device), first]
        out = []
        for t in range(x.size(1)):
            ema = torch.where(valid[:, t:t + 1], 0.5 * ema + 0.5 * x[:, t], ema)
            out.append(ema)
        trend = torch.stack(out, dim=1)
        return torch.cat([x, dev, trend], dim=-1)

    def forward(self, x, mask=None, baseline=None, cycle=None, cond=None, mode='dual'):
        B, L, N = x.shape
        if mask is None:
            mask = torch.ones(B, L, device=x.device)
        if baseline is None:
            baseline = x[:, 0]
        if cycle is None:
            cycle = torch.zeros(B, device=x.device)
        if self.input_mode == 'raw':
            inp = x
        else:
            inp = self._dsp_features(x, baseline, mask)
        h = self.pe(self.embed(inp))
        if self.use_cycle:
            ang = (2.0 * math.pi) * (cycle * 250.0).unsqueeze(1) / self.periods.view(1, -1)
            pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
            h = h + self.cycle_proj(pe).unsqueeze(1)
        pad = ~mask.bool()
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=pad)
        h = self.norm(h)
        pooled = masked_mean(h, mask)
        y = self.reg_head(pooled)
        return {'prediction': y, 'y_A': y, 'y_B': y, 'embedding': pooled}


def build_vanilla(num_sensors=14, d_model=64, num_layers=2, dropout=0.2, max_span=70,
                  input_mode='raw', use_cycle=False):
    return VanillaTransformer(num_sensors=num_sensors, d_model=d_model, num_layers=num_layers,
                              dropout=dropout, max_span=max_span, input_mode=input_mode,
                              use_cycle=use_cycle)
