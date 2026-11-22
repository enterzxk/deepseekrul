"""Paper C: plug CADF v1.2 in front of an arbitrary backbone.

Contract (Gate 1 is a paired comparison on the SAME backbone and SAME split):
    y_hat = Backbone( CADF_v12(x, mask, cluster) )

Two silent-failure modes this file exists to prevent
----------------------------------------------------
1. **K0 is an absolute cycle count, but the backbones' windows are 30/45/50.**
   With `K0 = 15` and a 30-cycle window, the "baseline" is computed from the first
   half of the window, not from a healthy period -- that is a different module,
   not the same module on a different backbone. So K0 is never hard-coded here:
   `K0Policy.absolute(k)` and `K0Policy.fraction(f)` are separate, explicit
   choices, the resolved value is recorded in `k0_resolved`, and the two modes
   must be run as separate arms. Never mix them in one table.

2. **CADF must only touch sensor channels.** It strips a per-condition baseline;
   doing that to operating variables (speed/temperature set points) is wrong,
   because the operating point is not a sensor offset. Backbones that keep the
   operating inputs (e.g. Two-Phase) therefore pass `sensor_idx`; channels
   outside it are passed through untouched.

Identity at initialisation
--------------------------
`CADFv12.proj` and `.gate` are zero-initialised, so `x + sigmoid(gate(z)) * proj(z)`
equals `x` exactly in IEEE arithmetic at step 0. Two consequences, both asserted
by `assert_identity`:

  * `cadf_on=False` and `cadf_on=True` agree bit-for-bit at initialisation, so the
    "off" arm is the V0 arm and no separate code path is needed;
  * the backbone must be constructed BEFORE the CADF module. `CADFv12` allocates
    three Linear layers, which would consume RNG and shift every downstream
    initialisation. `build_wrapped` therefore builds the backbone first and
    restores the RNG state afterwards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from paper_c.m1v12 import CADFv12


@dataclass(frozen=True)
class K0Policy:
    """How the causal-baseline warm-up length is derived.

    `absolute` is the original M1 definition (a fixed number of cycles).
    `fraction` is window-relative, `ceil(f * L)`, and is the only form that keeps
    the baseline semantically the same across backbones with different windows.
    """

    kind: str                 # 'absolute' | 'fraction'
    value: float

    def resolve(self, window_len: int) -> int:
        if self.kind == 'absolute':
            return int(self.value)
        if self.kind == 'fraction':
            return max(1, int(math.ceil(self.value * window_len)))
        raise ValueError(f'unknown K0 policy {self.kind!r}')

    def label(self) -> str:
        return f'{self.kind}:{self.value:g}'


class CADFWrapper(nn.Module):
    """Backbone with an optional CADF front-end on the sensor channels only."""

    def __init__(self, backbone: nn.Module, cadf: Optional[CADFv12],
                 sensor_idx: Optional[Sequence[int]] = None,
                 n_channels: Optional[int] = None,
                 cadf_on: bool = True,
                 k0_policy: Optional[K0Policy] = None):
        super().__init__()
        self.backbone = backbone
        self.cadf = cadf
        self.cadf_on = bool(cadf_on)
        self.sensor_idx = None if sensor_idx is None else list(int(i) for i in sensor_idx)
        self.n_channels = int(n_channels) if n_channels is not None else None
        self.k0_policy = k0_policy
        self.k0_resolved: dict[int, int] = {}
        if self.cadf_on:
            if cadf is None:
                raise ValueError('cadf_on=True but cadf is None')
            if self.sensor_idx is None or self.n_channels is None:
                # Fail loudly: silently running CADF over every channel is exactly
                # failure mode 2 above.
                raise ValueError('cadf_on=True requires sensor_idx and n_channels; '
                                 'CADF must not touch non-sensor channels')

    def _apply_cadf(self, x, mask, cluster, eng_base):
        L = x.shape[1]
        if L not in self.k0_resolved and self.k0_policy is not None:
            self.k0_resolved[L] = self.k0_policy.resolve(L)
        idx = torch.as_tensor(self.sensor_idx, device=x.device)
        xs = x.index_select(-1, idx)
        ys = self.cadf(xs, mask, cluster, eng_base)
        if ys.shape != xs.shape:
            raise RuntimeError(f'CADF changed shape {tuple(xs.shape)} -> {tuple(ys.shape)}')
        out = x.clone()
        out.index_copy_(-1, idx, ys)
        return out

    def forward(self, x, mask, cluster, eng_base=None):
        if self.cadf_on:
            x = self._apply_cadf(x, mask, cluster, eng_base)
        return self.backbone(x, mask)


def build_wrapped(backbone_factory: Callable[[], nn.Module],
                  cadf_factory: Callable[[], CADFv12],
                  *, seed: int, sensor_idx=None, n_channels=None,
                  cadf_on: bool = True, k0_policy: Optional[K0Policy] = None,
                  preserve_rng: bool = True) -> CADFWrapper:
    """Build backbone first so its weights match V0, then CADF.

    Order is the whole point: `CADFv12` allocates three Linear layers. If it were
    built first it would consume RNG and every backbone weight would differ from
    the frozen V0 run -- a silent, total loss of comparability.
    """
    torch.manual_seed(seed)
    backbone = backbone_factory()
    state = torch.get_rng_state()
    cadf = cadf_factory() if cadf_on else None
    if preserve_rng:
        torch.set_rng_state(state)
    return CADFWrapper(backbone, cadf, sensor_idx=sensor_idx, n_channels=n_channels,
                       cadf_on=cadf_on, k0_policy=k0_policy)


def reference_backbone(backbone_factory: Callable[[], nn.Module], seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return backbone_factory()


@torch.no_grad()
def assert_identity(wrapped: CADFWrapper, backbone_factory, seed: int,
                    x, mask, cluster, eng_base=None) -> None:
    """Three assertions that make 'CADF off == V0' a checked fact, not a claim."""
    ref = reference_backbone(backbone_factory, seed)
    a, b = wrapped.backbone.state_dict(), ref.state_dict()
    if list(a) != list(b):
        raise AssertionError('backbone state_dict keys differ from V0')
    for k in a:
        if not torch.equal(a[k], b[k]):
            raise AssertionError(f'backbone parameter {k} differs from V0 at init')

    # The backbones carry dropout, so two forward passes in train mode differ even
    # with identical weights. An identity check must run in eval mode; otherwise it
    # reports a failure that is pure dropout noise.
    modes = (wrapped.training, ref.training)
    wrapped.eval()
    ref.eval()
    try:
        y_ref = ref(x, mask)
        was = wrapped.cadf_on
        try:
            wrapped.cadf_on = False
            y_off = wrapped(x, mask, cluster, eng_base)
            if not torch.equal(y_off, y_ref):
                raise AssertionError('cadf_on=False does not reproduce the backbone output')
            if was:
                wrapped.cadf_on = True
                y_on = wrapped(x, mask, cluster, eng_base)
                if not torch.equal(y_on, y_ref):
                    raise AssertionError(
                        'CADF is not the identity at initialisation; check that proj/gate '
                        'are zero-initialised')
        finally:
            wrapped.cadf_on = was
    finally:
        wrapped.train(modes[0])
        ref.train(modes[1])


def param_delta(wrapped: CADFWrapper) -> dict:
    total = sum(p.numel() for p in wrapped.parameters())
    back = sum(p.numel() for p in wrapped.backbone.parameters())
    cadf = sum(p.numel() for p in wrapped.cadf.parameters()) if wrapped.cadf is not None else 0
    return dict(total=total, backbone=back, cadf=cadf,
                cadf_share=round(cadf / total, 4) if total else 0.0)
