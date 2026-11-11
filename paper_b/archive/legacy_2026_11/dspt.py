"""DSPT-v2: Dual-Span Pooled Transformer.

One shared encoder runs once over the full 100-step history; two nested suffix
pools read it (last 50 steps, and all valid steps) and a zero-parameter gate
mixes them. Parameter count equals the single-stream baseline of the same
temporal_pe width, so any gain cannot come from capacity.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SPAN = 100
SHORT_SPAN = 50


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """h (B,L,D), mask (B,L) with 1 = valid. Mean over valid steps only."""
    m = mask.unsqueeze(-1).to(h.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (h * m).sum(dim=1) / denom


class DSPT(nn.Module):
    def __init__(self, num_sensors=14, d_model=64, num_layers=2, dropout=0.2,
                 pe_periods=(25.0, 50.0, 125.0, 250.0), max_span=MAX_SPAN, instnorm=False,
                 use_ms=True, use_sa=True, use_pad_mask=True):
        super().__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model
        self.max_span = max_span
        self.instnorm = bool(instnorm)
        self.use_ms = bool(use_ms)
        self.use_sa = bool(use_sa)
        self.use_pad_mask = bool(use_pad_mask)
        self.n_feat = 4 if self.instnorm else 3
        self.sensor_in = nn.Linear(self.n_feat, d_model)
        self.multi_conv = nn.ModuleList([nn.Conv1d(d_model, d_model, k, padding=k//2, groups=d_model) for k in (3, 5, 7)])
        self.mscale_proj = nn.Linear(d_model, d_model)
        self.time_norm = nn.LayerNorm(d_model)
        self.register_buffer('pe_periods', torch.tensor(pe_periods))
        self.pe_proj = nn.Linear(2 * len(pe_periods), d_model)
        # learned positional table indexed by DISTANCE FROM THE END (0 = newest)
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_span, d_model))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(num_layers)])
        self.sensor_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=2*d_model, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.sensor_score = nn.Linear(d_model, 1)
        self.slot_score = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.hi_head = nn.Linear(d_model, 1)
        self.reg_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def _features(self, x, baseline, mask):
        """[raw, raw-baseline, causal EMA]; EMA recurses only over valid steps."""
        B, L, N = x.shape
        dev = x - baseline.unsqueeze(1)
        ema = x[:, 0].clone()
        out = []
        for t in range(L):
            ema = torch.where(mask[:, t:t + 1].bool(), 0.5 * ema + 0.5 * x[:, t], ema)
            out.append(ema)
        trend = torch.stack(out, dim=1)
        if not self.instnorm:
            return torch.stack([x, dev, trend], dim=-1)  # (B,L,N,3)
        # per-engine instance normalisation: remove this engine scale as well as its offset
        m = mask.unsqueeze(-1).to(x.dtype)                       # (B,L,1)
        den = m.sum(dim=1).clamp_min(1.0)                        # (B,1)
        mu = (x * m).sum(dim=1) / den                            # (B,N)
        var = (((x - mu.unsqueeze(1)) ** 2) * m).sum(dim=1) / den
        sc = var.sqrt().clamp_min(1e-3).unsqueeze(1)             # (B,1,N)
        norm = dev / sc
        return torch.stack([x, dev, norm, trend], dim=-1)  # (B,L,N,4)

    def encode(self, x, baseline, mask, cycle):
        B, L, N = x.shape
        feats = self._features(x, baseline, mask)
        h = self.sensor_in(feats)                                  # (B,L,N,D)
        hh = h.permute(0, 2, 3, 1).reshape(B * N, self.d_model, L)
        # replicate-pad the left-padded region with the first valid frame so that
        # zero padding cannot leak into the valid region through the convolution
        hh4 = hh.reshape(B, N, self.d_model, L)
        valid = mask.bool()
        first = valid.float().argmax(dim=1)
        fv = hh4[torch.arange(B, device=x.device), :, :, first]          # (B,N,D)
        m4 = valid.unsqueeze(1).unsqueeze(1)                             # (B,1,1,L)
        hh = torch.where(m4, hh4, fv.unsqueeze(-1)).reshape(B * N, self.d_model, L)
        if self.use_ms:
            ms = sum(F.gelu(c(hh)) for c in self.multi_conv) / 3.0
            hh = self.mscale_proj(ms.transpose(1, 2)).transpose(1, 2)
        else:
            hh = F.gelu(self.multi_conv[0](hh))
        # zero the convolution output on padding positions
        hh = hh * m4.expand(B, N, 1, L).reshape(B * N, 1, L)
        conv_out = hh.reshape(B, N, self.d_model, L).permute(0, 3, 1, 2)
        h = self.time_norm(h + conv_out)                            # (B,L,N,D)
        tokens = h.permute(0, 2, 1, 3).reshape(B * N, L, self.d_model)
        # distance-from-end positional table
        dist = torch.arange(L - 1, -1, -1, device=x.device)          # 0 = newest
        tokens = tokens + self.temporal_pe[:, dist]
        pad = ~mask.bool().unsqueeze(1).expand(B, N, L).reshape(B * N, L)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=pad) if self.use_pad_mask else layer(tokens)
        tokens = tokens.reshape(B, N, L, self.d_model).permute(0, 2, 1, 3)
        st = tokens.reshape(B * L, N, self.d_model)
        if self.use_sa:
            st = self.sensor_attn(st)
        tokens = st.reshape(B, L, N, self.d_model)
        score = torch.softmax(self.sensor_score(tokens).squeeze(-1), dim=-1)  # (B,L,N)
        pooled = (score.unsqueeze(-1) * tokens).sum(dim=2)           # (B,L,D)
        c_abs = cycle * 250.0
        ang = (2.0 * math.pi) * c_abs.unsqueeze(1) / self.pe_periods.view(1, -1)
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        pooled = pooled + self.pe_proj(pe).unsqueeze(1)
        return self.norm(pooled)                                     # (B,L,D)

    def slot_pool(self, h, mask):
        B, L, D = h.shape
        valid = mask.bool()
        m = valid.sum(dim=1).clamp_min(1)
        slots = [masked_mean(h, mask)]
        idx = torch.arange(L, device=h.device).unsqueeze(0).expand(B, L)
        start = L - m.unsqueeze(1)
        pos = idx - start
        seg_id = (pos.clamp_min(0) * 4) // m.unsqueeze(1)
        for k in range(4):
            seg = valid & (seg_id == k)
            slots.append(masked_mean(h, seg.to(h.dtype)))
        S = torch.stack(slots, dim=1)
        w = torch.softmax(self.slot_score(S).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * S).sum(dim=1), w

    def forward(self, x, mask=None, baseline=None, cycle=None, mode='dual'):
        B, L, N = x.shape
        if mask is None:
            mask = torch.ones(B, L, device=x.device)
        if baseline is None:
            baseline = x[:, 0]
        if cycle is None:
            cycle = torch.zeros(B, device=x.device)
        h = self.encode(x, baseline, mask, cycle)                    # (B,L,D)
        # distance-from-end index for each position
        dist = torch.arange(L - 1, -1, -1, device=x.device)          # (L,)
        valid = mask.bool()
        short = valid & (dist.unsqueeze(0) < SHORT_SPAN)
        h_A = masked_mean(h, short.to(h.dtype))
        h_B = masked_mean(h, valid.to(h.dtype))
        m = mask.sum(dim=1, keepdim=True)
        g = torch.clamp((m - SHORT_SPAN) / SHORT_SPAN, 0.0, 1.0)
        h_f = g * h_B + (1.0 - g) * h_A
        if mode == 'slot':
            h_s, w_slots = self.slot_pool(h, mask)
            return {'prediction': self.reg_head(h_s), 'y_A': self.reg_head(h_A),
                    'y_B': self.reg_head(h_B), 'gate': g, 'slot_w': w_slots, 'embedding': h_s}
        y = self.reg_head(h_f)
        hi_seq = self.hi_head(h).squeeze(-1)
        hi_pool = self.hi_head(h_f).reshape(-1)
        return {'prediction': y,
                'y_A': self.reg_head(h_A),
                'y_B': self.reg_head(h_B),
                'gate': g,
                'embedding': h_f, 'hi': hi_pool, 'hi_seq': hi_seq}


def build(num_sensors=14, d_model=64, num_layers=2, dropout=0.2, instnorm=False, max_span=MAX_SPAN, use_ms=True, use_sa=True, use_pad_mask=True):
    return DSPT(num_sensors=num_sensors, d_model=d_model, num_layers=num_layers, dropout=dropout, instnorm=instnorm, max_span=max_span, use_ms=use_ms, use_sa=use_sa, use_pad_mask=use_pad_mask)
