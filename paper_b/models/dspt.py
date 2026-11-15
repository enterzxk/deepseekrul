"""DSPT with auditable module switches (Paper B, ACTIVE model).

Design preserved from the original DSPT:
  raw / baseline-deviation / causal-EMA input representation, per-sensor
  projection, multi-scale depthwise temporal convolution, per-sensor temporal
  Transformer, optional cross-sensor Transformer (SA), learned sensor pooling,
  distance-from-end positional table, absolute-cycle Fourier encoding,
  dual-span pooling + gate, and optional HI auxiliary head.

Protocol fixes vs the legacy code (see paper_b/reports/PROTOCOL_AUDIT.md):
  P1  MS convolution uses replicate padding on both edges; the left padding
      region is filled with the first valid frame and its output is zeroed.
  P2  causal EMA is initialised at the first *valid* frame, not at x[:,0]
      (which is zero for left-padded windows).
  P3  modules are built conditionally, so parameter counts reflect active
      capacity (no dead parameters).
  P4  HI monotonic penalty is masked to adjacent valid pairs. The HI target is
      a decreasing health index, so the correct penalty punishes increases
      (the sign remains configurable for the attribution experiment).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SPAN = 100
SHORT_SPAN = 50


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class FastSensorAttention(nn.Module):
    """Pre-norm cross-sensor Transformer layer operating on (M, N, D).

    Numerically equivalent in structure to nn.TransformerEncoderLayer(norm_first=True)
    but implemented with three large batched ops, which is much faster when the
    batch dimension is B*L (tens of thousands of short sensor sequences).
    """

    def __init__(self, d_model, nhead=2, dropout=0.2):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.dh = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_keep=None):
        """key_keep: (M, N) bool, True = the sensor may be attended to.

        A masked sensor is removed from the Key/Value set by an additive -inf
        bias. SDPA's *float* mask is additive; a boolean mask would mean the
        opposite of a key-padding mask, so ~keep is never passed as a bool
        mask. None keeps the original call unchanged.
        """
        M, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(M, N, 3, self.nhead, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_bias = None
        if key_keep is not None:
            keep = key_keep.to(device=x.device, dtype=torch.bool)
            if keep.shape != (M, N):
                raise ValueError(f'key_keep must be {(M, N)}, got {tuple(keep.shape)}')
            attn_bias = torch.zeros((M, 1, 1, N), dtype=q.dtype, device=q.device)
            attn_bias = attn_bias.masked_fill(~keep[:, None, None, :], float('-inf'))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias,
                                           dropout_p=self.drop.p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(M, N, D)
        x = x + self.drop(self.out(o))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DSPT(nn.Module):
    dual_span = True

    def __init__(self, num_sensors=14, d_model=64, num_layers=2, dropout=0.2,
                 pe_periods=(25.0, 50.0, 125.0, 250.0), max_span=MAX_SPAN,
                 instnorm=False, ms_mode='full', sa_mode='off', hi_source='none',
                 hi_mono='none', use_pad_mask=True, use_cycle=True, n_cond=6,
                 legacy_ms_pad=False, legacy_ema=False):
        super().__init__()
        assert ms_mode in ('off', 'full', 'single_proj', 'multi_noproj', 'multi3', 'linear_only')
        assert sa_mode in ('off', 'basic', 'identity', 'cond', 'identity_cond', 'mean', 'static')
        assert hi_source in ('none', 'hf', 'hB', 'hA')
        assert hi_mono in ('none', 'correct', 'wrong')
        self.num_sensors = num_sensors
        self.d_model = d_model
        self.max_span = max_span
        self.instnorm = bool(instnorm)
        self.ms_mode = ms_mode
        self.sa_mode = sa_mode
        self.hi_source = hi_source
        self.hi_mono = hi_mono
        self.use_pad_mask = bool(use_pad_mask)
        self.use_cycle = bool(use_cycle)
        self.legacy_ms_pad = bool(legacy_ms_pad)
        self.legacy_ema = bool(legacy_ema)
        self.n_feat = 4 if self.instnorm else 3

        self.sensor_in = nn.Linear(self.n_feat, d_model)

        # ---- multi-scale temporal convolution --------------------------------
        if ms_mode != 'linear_only':
            self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model,
                                   padding_mode='replicate')
        if ms_mode in ('full', 'multi_noproj'):
            self.conv5 = nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model,
                                   padding_mode='replicate')
            self.conv7 = nn.Conv1d(d_model, d_model, 7, padding=3, groups=d_model,
                                   padding_mode='replicate')
        if ms_mode == 'multi3':
            self.conv3b = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model,
                                    padding_mode='replicate')
            self.conv3c = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model,
                                    padding_mode='replicate')
        if ms_mode in ('full', 'single_proj', 'multi3', 'linear_only'):
            self.ms_proj = nn.Linear(d_model, d_model)

        self.time_norm = nn.LayerNorm(d_model)
        self.register_buffer('pe_periods', torch.tensor(pe_periods))
        if self.use_cycle:
            self.pe_proj = nn.Linear(2 * len(pe_periods), d_model)

        # learned positional table indexed by distance from the end (0 = newest)
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_span, d_model))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(num_layers)])

        # ---- cross-sensor attention (SA) ------------------------------------
        if sa_mode in ('basic', 'identity', 'cond', 'identity_cond'):
            self.sensor_attn = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=2, dim_feedforward=2 * d_model, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True)
        if sa_mode in ('identity', 'identity_cond'):
            self.sensor_id = nn.Parameter(torch.zeros(1, num_sensors, d_model))
        if sa_mode in ('cond', 'identity_cond'):
            self.cond_emb = nn.Embedding(n_cond, d_model)

        if sa_mode in ('off', 'basic', 'identity', 'cond', 'identity_cond'):
            self.sensor_score = nn.Linear(d_model, 1)
        if sa_mode == 'static':
            self.static_score = nn.Parameter(torch.zeros(num_sensors))
        self.norm = nn.LayerNorm(d_model)
        self.reg_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, 1))
        if hi_source != 'none':
            self.hi_head = nn.Linear(d_model, 1)

    # ------------------------------------------------------------------ utils
    def _features(self, x, baseline, mask):
        """[raw, raw - baseline, causal EMA] with EMA recursing over valid steps."""
        dev = x - baseline.unsqueeze(1)
        valid = mask.bool()
        first = valid.float().argmax(dim=1)                       # (B,)
        if self.legacy_ema:
            ema = x[:, 0].clone()
        else:
            ema = x[torch.arange(x.size(0), device=x.device), first]  # (B,N)
        out = []
        for t in range(x.size(1)):
            ema = torch.where(valid[:, t:t + 1], 0.5 * ema + 0.5 * x[:, t], ema)
            out.append(ema)
        trend = torch.stack(out, dim=1)
        if not self.instnorm:
            return torch.stack([x, dev, trend], dim=-1)
        m = mask.unsqueeze(-1).to(x.dtype)
        den = m.sum(dim=1).clamp_min(1.0)
        mu = (x * m).sum(dim=1) / den
        var = (((x - mu.unsqueeze(1)) ** 2) * m).sum(dim=1) / den
        sc = var.sqrt().clamp_min(1e-3).unsqueeze(1)
        norm = dev / sc
        return torch.stack([x, dev, norm, trend], dim=-1)

    def _ms(self, hh):
        """hh: (B*N, D, L) -> (B*N, D, L)."""
        if self.ms_mode == 'linear_only':
            return self.ms_proj(hh.transpose(1, 2)).transpose(1, 2)
        if self.ms_mode == 'off':
            return F.gelu(self.conv3(hh))
        if self.ms_mode == 'single_proj':
            return self.ms_proj(F.gelu(self.conv3(hh)).transpose(1, 2)).transpose(1, 2)
        if self.ms_mode == 'multi_noproj':
            return (F.gelu(self.conv3(hh)) + F.gelu(self.conv5(hh)) + F.gelu(self.conv7(hh))) / 3.0
        if self.ms_mode == 'multi3':
            ms = (F.gelu(self.conv3(hh)) + F.gelu(self.conv3b(hh)) + F.gelu(self.conv3c(hh))) / 3.0
            return self.ms_proj(ms.transpose(1, 2)).transpose(1, 2)
        ms = (F.gelu(self.conv3(hh)) + F.gelu(self.conv5(hh)) + F.gelu(self.conv7(hh))) / 3.0
        return self.ms_proj(ms.transpose(1, 2)).transpose(1, 2)

    # ---------------------------------------------------------------- forward
    def encode(self, x, baseline, mask, cycle, cond=None, sensor_keep=None):
        """Encode a window.

        sensor_keep: (B, N) bool, True = the sensor is available. None runs the
        original, unmodified path: no sanitising, no masking and no extra
        random numbers. A hidden sensor is removed *before the first
        cross-sensor mixing* (the SA block) and again after it, and every
        sensor pooling re-normalises over the available set, so no hidden
        value can enter the prediction through a residual, a statistic or a
        pooling weight.
        """
        B, L, N = x.shape
        keep = None
        if sensor_keep is not None:
            keep = sensor_keep.to(device=x.device, dtype=torch.bool)
            if keep.shape != (B, N):
                raise ValueError(f'sensor_keep must be {(B, N)}, got {tuple(keep.shape)}')
            if not bool(keep.any(dim=1).all()):
                raise ValueError('sensor_keep must retain at least one sensor per sample')
            # Replace hidden channels with a finite placeholder before any
            # feature is computed: a NaN/Inf in an unavailable channel must not
            # reach a convolution, a variance or a normalisation. A non-finite
            # value on a channel declared available is an error, not a zero.
            finite_x = torch.where(keep[:, None, :], torch.isfinite(x),
                                   torch.ones_like(x, dtype=torch.bool))
            finite_b = torch.where(keep, torch.isfinite(baseline),
                                   torch.ones_like(baseline, dtype=torch.bool))
            if not (bool(finite_x.all()) and bool(finite_b.all())):
                raise ValueError('non-finite sensor value on a channel marked available')
            x = torch.where(keep[:, None, :], x, torch.zeros_like(x))
            baseline = torch.where(keep, baseline, torch.zeros_like(baseline))
        feats = self._features(x, baseline, mask)
        h = self.sensor_in(feats)                                  # (B,L,N,D)
        hh4 = h.permute(0, 2, 3, 1)                                # (B,N,D,L)
        valid = mask.bool()
        first = valid.float().argmax(dim=1)
        fv = hh4[torch.arange(B, device=x.device), :, :, first]     # (B,N,D)
        m4 = valid.unsqueeze(1).unsqueeze(1)                        # (B,1,1,L)
        if self.legacy_ms_pad:
            hh = hh4.reshape(B * N, self.d_model, L)
        else:
            hh = torch.where(m4, hh4, fv.unsqueeze(-1)).reshape(B * N, self.d_model, L)
        conv = self._ms(hh)
        if not self.legacy_ms_pad:
            conv = conv * m4.expand(B, N, 1, L).reshape(B * N, 1, L)
        conv_out = conv.reshape(B, N, self.d_model, L).permute(0, 3, 1, 2)
        h = self.time_norm(h + conv_out)                            # (B,L,N,D)

        tokens = h.permute(0, 2, 1, 3).reshape(B * N, L, self.d_model)
        dist = torch.arange(L - 1, -1, -1, device=x.device)
        tokens = tokens + self.temporal_pe[:, dist]
        pad = ~mask.bool().unsqueeze(1).expand(B, N, L).reshape(B * N, L)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=pad) if self.use_pad_mask else layer(tokens)
        tokens = tokens.reshape(B, N, L, self.d_model).permute(0, 2, 1, 3)

        if keep is not None:                       # hidden tokens are exactly 0
            tokens = torch.where(keep[:, None, :, None], tokens, torch.zeros_like(tokens))

        if self.sa_mode in ('basic', 'identity', 'cond', 'identity_cond'):
            st = tokens.reshape(B * L, N, self.d_model)
            if self.sa_mode in ('identity', 'identity_cond'):
                st = st + self.sensor_id
            if self.sa_mode in ('cond', 'identity_cond'):
                cc = cond if cond is not None else torch.zeros(B, dtype=torch.long, device=x.device)
                cc = cc.unsqueeze(1).expand(B, L).reshape(B * L)
                st = st + self.cond_emb(cc).unsqueeze(1)
            if keep is None:
                st = self.sensor_attn(st)
            else:
                # A hidden sensor may not be used as Key/Value. After attention
                # its own Query output is zeroed again: a key-padding mask does
                # not remove the hidden rows' outputs by itself.
                keep_bl = keep[:, None, :].expand(B, L, N).reshape(B * L, N)
                if self.use_pad_mask:
                    st = self.sensor_attn(st, src_key_padding_mask=~keep_bl)
                else:
                    st = self.sensor_attn(st)
                st = torch.where(keep_bl[..., None], st, torch.zeros_like(st))
            tokens = st.reshape(B, L, N, self.d_model)

        if keep is None:
            if self.sa_mode == 'mean':
                pooled = tokens.mean(dim=2)                                        # (B,L,D)
            elif self.sa_mode == 'static':
                w = torch.softmax(self.static_score, dim=0).view(1, 1, -1, 1)      # (1,1,N,1)
                pooled = (w * tokens).sum(dim=2)                                   # (B,L,D)
            else:
                score = torch.softmax(self.sensor_score(tokens).squeeze(-1), dim=-1)  # (B,L,N)
                pooled = (score.unsqueeze(-1) * tokens).sum(dim=2)                     # (B,L,D)
        else:
            # Pooling re-normalises over the available sensors only. Static
            # logits are masked *before* the softmax (never softmax-then-
            # delete); mean pooling divides by the number actually kept.
            kf = keep[:, None, :].to(tokens.dtype)                              # (B,1,N)
            if self.sa_mode == 'mean':
                pooled = (tokens * kf.unsqueeze(-1)).sum(dim=2) / kf.sum(dim=2, keepdim=True).clamp_min(1.0)
            elif self.sa_mode == 'static':
                logits = self.static_score.view(1, 1, -1).expand(B, L, N)
                logits = logits.masked_fill(~keep[:, None, :], float('-inf'))
                w = torch.softmax(logits, dim=-1).to(tokens.dtype)
                pooled = (w.unsqueeze(-1) * tokens).sum(dim=2)
            else:
                logits = self.sensor_score(tokens).squeeze(-1)                  # (B,L,N)
                logits = logits.masked_fill(~keep[:, None, :], float('-inf'))
                score = torch.softmax(logits, dim=-1)
                pooled = (score.unsqueeze(-1) * tokens).sum(dim=2)
        if self.use_cycle:
            c_abs = cycle * 250.0
            ang = (2.0 * math.pi) * c_abs.unsqueeze(1) / self.pe_periods.view(1, -1)
            pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
            pooled = pooled + self.pe_proj(pe).unsqueeze(1)
        return self.norm(pooled)

    def forward(self, x, mask=None, baseline=None, cycle=None, cond=None, mode='dual',
                sensor_keep=None):
        B, L, N = x.shape
        if mask is None:
            mask = torch.ones(B, L, device=x.device)
        if baseline is None:
            baseline = x[:, 0]
        if cycle is None:
            cycle = torch.zeros(B, device=x.device)
        h = self.encode(x, baseline, mask, cycle, cond, sensor_keep=sensor_keep)
        dist = torch.arange(L - 1, -1, -1, device=x.device)
        valid = mask.bool()
        short = valid & (dist.unsqueeze(0) < SHORT_SPAN)
        h_A = masked_mean(h, short.to(h.dtype))
        h_B = masked_mean(h, valid.to(h.dtype))
        m = mask.sum(dim=1, keepdim=True)
        g = torch.clamp((m - SHORT_SPAN) / SHORT_SPAN, 0.0, 1.0)
        h_f = g * h_B + (1.0 - g) * h_A

        out = {
            'y_A': self.reg_head(h_A),
            'y_B': self.reg_head(h_B),
            'prediction': self.reg_head(h_f) if mode == 'dual' else (
                self.reg_head(h_B) if mode == 'long' else self.reg_head(h_A)),
            'gate': g,
            'embedding': h_f,
        }
        if self.hi_source != 'none':
            hsel = {'hf': h_f, 'hB': h_B, 'hA': h_A}[self.hi_source]
            out['hi'] = self.hi_head(hsel).reshape(-1)
            out['hi_seq'] = self.hi_head(h).squeeze(-1)
        return out


def build_dspt(num_sensors=14, d_model=64, num_layers=2, dropout=0.2, max_span=MAX_SPAN,
               instnorm=False, ms_mode='full', sa_mode='off', hi_source='none',
               hi_mono='none', use_pad_mask=True, use_cycle=True, n_cond=6,
               legacy_ms_pad=False, legacy_ema=False):
    return DSPT(num_sensors=num_sensors, d_model=d_model, num_layers=num_layers,
                dropout=dropout, max_span=max_span, instnorm=instnorm, ms_mode=ms_mode,
                sa_mode=sa_mode, hi_source=hi_source, hi_mono=hi_mono,
                use_pad_mask=use_pad_mask, use_cycle=use_cycle, n_cond=n_cond,
                legacy_ms_pad=legacy_ms_pad, legacy_ema=legacy_ema)
