"""MSFB: multi-scale sensor feature bank with feature-type attention.

The frozen DSPT input only has [raw, raw-baseline, causal EMA]. This module
adds a causally computed sensor feature bank (velocity, acceleration, 5/10/20
slopes, CUSUM, fast-slow EMA difference, cross-sensor contrast) and a
feature-type attention over those families. The base path is exactly DSPT's
linear sensor projection; the extra path is zero-initialised at the fusion
layer, so at step 0 the model is a DSPT with the same base features.

Everything is computed from x / baseline / mask inside the model; no labels and
no future frames are used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.dspt import DSPT


def _ema(x, valid, alpha):
    out = []
    e = x[:, 0].clone()
    for t in range(x.size(1)):
        e = torch.where(valid[:, t:t + 1], alpha * e + (1.0 - alpha) * x[:, t], e)
        out.append(e)
    return torch.stack(out, dim=1)


class SensorFeatureBank(nn.Module):
    """8 extra causal channels per sensor."""

    def forward(self, x, baseline, mask):
        B, L, N = x.shape
        valid = mask.bool()
        m = valid.unsqueeze(-1)
        dev = x - baseline.unsqueeze(1)

        dx = torch.zeros_like(x)
        dx[:, 1:] = x[:, 1:] - x[:, :-1]
        dx = dx * m

        ddx = torch.zeros_like(x)
        ddx[:, 1:] = dx[:, 1:] - dx[:, :-1]
        ddx = ddx * m

        def slope(h):
            out = torch.zeros_like(x)
            if L > h:
                out[:, h:] = (x[:, h:] - x[:, :-h]) / float(h)
            return out * m

        s5, s10, s20 = slope(5), slope(10), slope(20)

        cs = torch.cumsum(dev * m, dim=1)
        t = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        cs = cs / t.sqrt().clamp_min(1.0)

        ediff = (_ema(x, valid, 0.5) - _ema(x, valid, 0.1)) * m

        med = x.median(dim=2, keepdim=True).values
        contrast = (x - med) * m

        return torch.stack([dx, ddx, s5, s10, s20, cs, ediff, contrast], dim=-1)


class FeatureTypeFusion(nn.Module):
    """Base DSPT projection + zero-init fused extra-feature path."""

    def __init__(self, n_base, d_model, hidden=16):
        super().__init__()
        self.n_base = int(n_base)
        self.base_proj = nn.Linear(n_base, d_model)
        dims = [1, 1, 3, 1, 1, 1]
        self.family_proj = nn.ModuleList([nn.Linear(k, d_model) for k in dims])
        self.gate = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))
        self.fuse = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, feats):
        base = feats[..., :self.n_base]
        extra = feats[..., self.n_base:]
        hs = []
        start = 0
        for proj, k in zip(self.family_proj, [1, 1, 3, 1, 1, 1]):
            hs.append(proj(extra[..., start:start + k]))
            start += k
        h_fam = torch.stack(hs, dim=3)              # (B,L,N,F,D)
        score = self.gate(h_fam).squeeze(-1)        # (B,L,N,F)
        alpha = torch.softmax(score, dim=-1)
        fused = (alpha.unsqueeze(-1) * h_fam).sum(dim=3)
        return self.base_proj(base) + self.fuse(fused)


class DSPT_MSFB(DSPT):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.feat_bank = SensorFeatureBank()
        self.sensor_in = FeatureTypeFusion(self.n_feat, self.d_model)

    def _features(self, x, baseline, mask):
        base = super()._features(x, baseline, mask)
        extra = self.feat_bank(x, baseline, mask)
        return torch.cat([base, extra], dim=-1)


def build_msfb(num_sensors=14, d_model=64, num_layers=2, dropout=0.2,
               max_span=100, instnorm=False, ms_mode='full', sa_mode='off',
               hi_source='none', hi_mono='none', use_pad_mask=True,
               use_cycle=True, cycle_pos=None, n_cond=6,
               legacy_ms_pad=False, legacy_ema=False):
    import inspect
    kw = dict(num_sensors=num_sensors, d_model=d_model, num_layers=num_layers,
              dropout=dropout, max_span=max_span, instnorm=instnorm,
              ms_mode=ms_mode, sa_mode=sa_mode, hi_source=hi_source,
              hi_mono=hi_mono, use_pad_mask=use_pad_mask, use_cycle=use_cycle,
              n_cond=n_cond, legacy_ms_pad=legacy_ms_pad,
              legacy_ema=legacy_ema)
    # The a40/3090 Stage D snapshot predates cycle_pos; the local Paper B
    # version has it. Pass it only when the installed DSPT accepts it.
    if 'cycle_pos' in inspect.signature(DSPT.__init__).parameters:
        kw['cycle_pos'] = cycle_pos
    return DSPT_MSFB(**kw)
