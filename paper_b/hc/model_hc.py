"""Stage C models: the frozen DSPT backbone with a swappable cycle path.

Hard constraint from the task book (§6): the backbone and the direct RUL
regression head stay exactly as frozen; **only the cycle path changes**. The
class below therefore subclasses ``paper_b.models.dspt.DSPT`` instead of
copying it: the encoder, multi-scale convolution, temporal Transformer,
cross-sensor attention, sensor pooling, dual-span gate and regression head are
all the frozen implementations. The subclass builds the base with
``use_cycle=False`` (so the base allocates no ``pe_proj`` and adds nothing) and
injects its own cycle term at one fixed place, which makes A2/A4/A5 mutually
comparable and keeps parameter accounting honest.

A3a is the frozen Fourier *form*; the frozen *injection point* (pre-norm, inside
``DSPT.encode``) is available as variant ``A3p`` for a one-run check that the
injection point is not what drives any difference.
"""
import torch
import torch.nn as nn

from ..models.dspt import DSPT
from .cycle_encoder import build_cycle_encoder


class DSPT_HC(DSPT):
    """Frozen DSPT + an explicit cycle encoder added after the shared encoder."""

    def __init__(self, cycle_mode='none', period_bank='frozen', **kw):
        super().__init__(use_cycle=False, **kw)
        self.cycle_mode = cycle_mode
        self.period_bank = period_bank
        self.cycle_enc = build_cycle_encoder(cycle_mode, self.d_model, period_bank)
        self.needs_cycle = self.cycle_enc is not None

    def encode(self, x, baseline, mask, cycle, cond=None):
        h = super().encode(x, baseline, mask, cycle, cond)
        if self.cycle_enc is None:
            return h
        # h is (B, L, D); the cycle term is one vector per sample, broadcast
        # over the time axis. Missing this unsqueeze is what the first smoke
        # run caught (RuntimeError: 70 vs 512).
        return h + self.cycle_enc(cycle).unsqueeze(1)


class DSPT_HC_Frozen(DSPT):
    """Reference: the frozen model, cycle added inside encode() as originally.

    A3p must differ from A3a *only* in the injection point, so the scalar has to
    match the frozen pipeline exactly. ``data_hc`` hands every encoder the cycle
    in **cycles** (0..250 after the frozen clip), whereas the frozen
    ``DSPT.encode`` multiplies by 250 because the frozen data layer stores
    ``min(c/250, 1)``. Passing raw cycles straight through would scale every
    angle by 250 and turn A3p into a different, meaningless control; the
    conversion is therefore done at this boundary. The injection point itself --
    pre-LayerNorm, inside ``DSPT.encode`` -- is untouched.
    """

    def __init__(self, **kw):
        super().__init__(use_cycle=True, **kw)
        self.needs_cycle = True

    def encode(self, x, baseline, mask, cycle, cond=None):
        if cycle is None:
            return super().encode(x, baseline, mask, None, cond)
        # Upper bound as well as lower: anything above 250 is outside the frozen
        # encoding's domain, and silently wrapping it would make A3p incomparable.
        c = torch.clamp(cycle, min=0.0, max=250.0) / 250.0
        return super().encode(x, baseline, mask, c, cond)


class CycleOnly(nn.Module):
    """A0: how much of the label is explained by absolute age alone.

    Deliberately tiny and sensor-free. It is not a competitor; it is the
    reference line that says whether the benchmark is solvable from age.
    """

    dual_span = False
    use_cycle = True
    needs_cycle = True

    def __init__(self, d_model=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, 1))

    def forward(self, x=None, mask=None, baseline=None, cycle=None, cond=None, mode='dual'):
        c = (torch.clamp(cycle, min=0.0) / 250.0).unsqueeze(1)
        y = self.net(c)
        return {'prediction': y, 'y_A': y, 'y_B': y,
                'gate': torch.zeros_like(y)}


def build_hc(cfg):
    """Build the model for one variant from an effective config dict."""
    if cfg['variant'] == 'A0':
        return CycleOnly(d_model=cfg['d_model'])
    bb = cfg['backbone']
    common = dict(num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
                  dropout=cfg['dropout'], max_span=cfg['span'],
                  ms_mode=bb['ms_mode'], sa_mode=bb['sa_mode'],
                  hi_source=bb['hi_source'], hi_mono=bb['hi_mono'],
                  use_pad_mask=True, n_cond=cfg['n_cond'])
    # A2 ruling: A3p is the frozen injection point, driven by the config flag so
    # the variant table stays the single source of truth.
    if cfg.get('frozen_injection'):
        return DSPT_HC_Frozen(**common)
    return DSPT_HC(cycle_mode=cfg['cycle_mode'],
                   period_bank=cfg.get('period_bank', 'frozen'), **common)
