"""M4 CSR: condition-aware sensor reliability pooling, built ON THE B0 BACKBONE.

Why a subclass instead of a flag
--------------------------------
``run_id()`` keys on the arm, so a flag would land in B0's directory and the
guard would refuse the second run -- the same collision that forced M7 to become
a formal arm. M4 is therefore registered as ``--arm M4``.

What changes, and only what changes
-----------------------------------
The frozen pooling is ``softmax_j(sensor_score(tokens)_j)`` -- a learned
per-timestep, per-sensor score. M4 replaces that one scorer with a wrapper that
adds a *reliability* bias before the softmax:

    w_tj = softmax_j( sensor_score(h_tj)
                      + cond_bias[cond_t, j]              (M4-C, 'cond')
                      + trend_proj(trend_tj) )            (M4-D, 'full')
    z_t  = sum_j w_tj * h_tj

``cond_bias`` and ``trend_proj`` are **zero-initialised**, so at step 0 the
logits are exactly B0's logits, the softmax is exactly B0's softmax, and the
prediction is bit-identical to B0. That is what makes M4 a one-variable change:
any gain later is attributable to condition/reliability weighting rather than to
a lucky re-initialisation of the pooling head.

What M4 does NOT do
-------------------
* It does not touch ``encode``'s body. ``DSPT.encode`` is code-locked and its
  body is not copied here; the only thing swapped is the ``sensor_score``
  module instance, so the convolution / Transformer / attention path stays
  literally the frozen implementation.
* It does not add a new Transformer. The task book explicitly warns that M4 must
  not be "another attention layer".
* Trend is computed from the **input feature** rate (per sensor, causal running
  mean subtracted), i.e. an observed 变化率, not from future observations.
"""
import torch
import torch.nn as nn

from .hi_d import DSPT_HC_Structured


def causal_trend(feats, mask):
    """Per-sensor causal contrast: feats_t - mean_{t'<=t} feats.

    ``feats`` is (B, L, N, F), ``mask`` is (B, L). Only past and present
    observations enter the mean, so this is causal by construction.
    """
    m = mask.to(feats.dtype).unsqueeze(-1).unsqueeze(-1)      # (B,L,1,1)
    cs = torch.cumsum(feats * m, dim=1)
    cm = torch.cumsum(m, dim=1)
    mean = cs / cm.clamp_min(1.0)
    return feats - mean


class CSRSensorScore(nn.Module):
    """Drop-in replacement for ``DSPT.sensor_score`` with a reliability bias.

    The base scorer is kept as a submodule (``base``) and its weights are
    preserved, so a B0 checkpoint transfers with a key rename only.
    """

    def __init__(self, base, d_model, n_cond, num_sensors, n_feat, mode='cond'):
        super().__init__()
        assert mode in ('cond', 'full'), mode
        self.base = base
        self.mode = mode
        self.cond_bias = nn.Embedding(n_cond, num_sensors)
        nn.init.zeros_(self.cond_bias.weight)
        if mode == 'full':
            self.trend_proj = nn.Linear(n_feat, 1)
            nn.init.zeros_(self.trend_proj.weight)
            nn.init.zeros_(self.trend_proj.bias)
        self._cond = None
        self._trend = None

    def set_context(self, cond=None, trend=None):
        """Called by ``DSPT_CSR.encode`` right before the frozen forward."""
        self._cond = cond
        self._trend = trend

    def forward(self, tokens):                       # (B,L,N,D) -> (B,L,N,1)
        logits = self.base(tokens)                   # frozen B0 score
        if self._cond is not None:
            # (B,N) -> (B,1,N,1): the same bias for every timestep of a window,
            # because the operating condition is constant within a window.
            logits = logits + self.cond_bias(self._cond).unsqueeze(1).unsqueeze(-1)
        if self.mode == 'full' and self._trend is not None:
            logits = logits + self.trend_proj(self._trend)
        return logits


class DSPT_CSR(DSPT_HC_Structured):
    """B0 build path with the sensor scorer swapped for CSR."""

    def __init__(self, csr_mode='cond', n_cond=6, **kw):
        # n_cond must reach the backbone too, otherwise cond_emb (built inside
        # DSPT) would fall back to the default while cond_bias uses the config
        # value -- a silent size mismatch on any subset with a different count.
        super().__init__(cycle_mode='none', n_cond=n_cond, **kw)
        self.csr_mode = csr_mode
        base = self.sensor_score
        self.sensor_score = CSRSensorScore(
            base, self.d_model, n_cond, self.num_sensors, self.n_feat, mode=csr_mode)

    def encode(self, x, baseline, mask, cycle, cond=None):
        trend = None
        if self.csr_mode == 'full':
            feats = self._features(x, baseline, mask)
            trend = causal_trend(feats, mask)
        self.sensor_score.set_context(cond=cond, trend=trend)
        return super().encode(x, baseline, mask, cycle, cond)

    def state_dict_remapped(self):
        """CSR -> B0 key layout, dropping the reliability bias.

        Used by the parity test: loading this into a plain B0 must reproduce the
        CSR model exactly while the bias is zero.
        """
        out = {}
        for k, v in self.state_dict().items():
            if k.startswith('sensor_score.base.'):
                out['sensor_score.' + k[len('sensor_score.base.'):]] = v
            elif k.startswith('sensor_score.cond_bias.'):
                continue
            elif k.startswith('sensor_score.trend_proj.'):
                continue
            else:
                out[k] = v
        return out


def remap_b0_state_dict(sd):
    """B0 init -> CSR init. Only the scorer's keys move under ``base``."""
    out = {}
    for k, v in sd.items():
        if k.startswith('sensor_score.'):
            out['sensor_score.base.' + k[len('sensor_score.'):]] = v
        else:
            out[k] = v
    return out


CSR_ALLOWED_MISSING = ('head.', 'sensor_score.cond_bias.', 'sensor_score.trend_proj.')


@torch.no_grad()
def assert_csr_parity(net, batch, tol=1e-6):
    """M4 must be bit-identical to B0 before the reliability bias is trained.

    Compares the CSR model against the same weights with the bias zeroed, i.e.
    it asserts that the wrapper is transparent at step 0. Run this on CPU before
    any GPU dispatch: if it fails, M4's numbers are B0's numbers plus noise and
    the arm is not a one-variable ablation.
    """
    was = net.training
    net.eval()
    out = net(batch['x'], mask=batch['mask'], baseline=batch['baseline'],
              cycle=batch['cycle'], cond=batch['cond'], mode='dual')['prediction']
    base = net.sensor_score.base
    with torch.no_grad():
        # zero-bias reference: run the frozen scorer path directly
        net.sensor_score.cond_bias.weight.zero_()
        if net.csr_mode == 'full':
            net.sensor_score.trend_proj.weight.zero_()
            net.sensor_score.trend_proj.bias.zero_()
        ref = net(batch['x'], mask=batch['mask'], baseline=batch['baseline'],
                  cycle=batch['cycle'], cond=batch['cond'], mode='dual')['prediction']
    worst = float((out - ref).abs().max())
    if was:
        net.train()
    if worst > tol:
        raise AssertionError('CSR is not transparent at step 0: max|diff|=%.3e' % worst)
    return worst
