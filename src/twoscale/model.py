"""Two-level (engine x window) scale for conformal RUL.

Candidate E's mechanism: the per-window half-width of the frozen CQR carries almost
no information about an engine's error magnitude (rho ~ +0.09, negative on FD001),
while the oracle optimal width tracks it strongly (rho = 0.727) and pooled
trajectory features can extract part of it (rho = 0.558).

This module adds an ENGINE-level scale on top of the existing per-window head:

    s_i = sigma_eng[e(i)] * sigma_win_i * kappa

Identifiability: sigma_win is normalised to have mean 1 inside each engine, so the
product cannot trade off between the two factors.  sigma_eng is constant within an
engine by construction (it is produced by pooling all of that engine's windows).

CPU-testable; no training loop here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class TwoScaleOutput:
    y_hat: torch.Tensor
    sigma_win: torch.Tensor   # (N,) within-engine mean 1
    sigma_eng: torch.Tensor   # (N,) constant within an engine
    scale: torch.Tensor       # (N,) sigma_eng * sigma_win


def _segment_mean(values: torch.Tensor, ids: torch.Tensor, n: int) -> torch.Tensor:
    """Mean of `values` per id, broadcast back to the original shape."""
    s = torch.zeros(n, dtype=values.dtype, device=values.device)
    c = torch.zeros(n, dtype=values.dtype, device=values.device)
    s.index_add_(0, ids, values)
    c.index_add_(0, ids, torch.ones_like(values))
    mean = s / c.clamp_min(1.0)
    return mean[ids]


def _segment_pool(h: torch.Tensor, ids: torch.Tensor, n: int) -> torch.Tensor:
    """Mean-pool the representation per engine -> (n, d)."""
    d = h.shape[-1]
    out = torch.zeros(n, d, dtype=h.dtype, device=h.device)
    cnt = torch.zeros(n, dtype=h.dtype, device=h.device)
    out.index_add_(0, ids, h)
    cnt.index_add_(0, ids, torch.ones(h.shape[0], dtype=h.dtype, device=h.device))
    return out / cnt.clamp_min(1.0).unsqueeze(-1)


class TwoScaleModel(nn.Module):
    """Frozen backbone + per-window scale head + per-engine scale head.

    Args:
        backbone: frozen IDA-PE model exposing encode(x[, adjacency]) -> (h, y).
        d_model: width of the pooled representation.
        use_engine: ablation switch (False -> sigma_eng == 1 for every engine).
        use_window: ablation switch (False -> sigma_win == 1 for every window).
        eps: numerical floor.
    """

    def __init__(self, backbone: nn.Module, d_model: int, *, hidden: int = 64,
                 use_engine: bool = True, use_window: bool = True, eps: float = 1e-3) -> None:
        super().__init__()
        self.backbone = backbone
        self.d_model = int(d_model)
        self.use_engine = bool(use_engine)
        self.use_window = bool(use_window)
        self.eps = float(eps)

        self.win_head = nn.Sequential(nn.Linear(self.d_model, hidden), nn.SiLU(),
                                      nn.Linear(hidden, 1))
        self.eng_head = nn.Sequential(nn.Linear(self.d_model, hidden), nn.SiLU(),
                                      nn.Linear(hidden, 1))

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "TwoScaleModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def _encode(self, x, adjacency):
        with torch.no_grad():
            if hasattr(self.backbone, "encode"):
                try:
                    h, y = self.backbone.encode(x, adjacency)
                except TypeError:
                    h, y = self.backbone.encode(x)
            else:
                out = self.backbone(x) if adjacency is None else self.backbone(x, adjacency)
                y = out[0] if isinstance(out, (tuple, list)) else out
                h = out[1] if isinstance(out, (tuple, list)) else None
                if h is None:
                    raise RuntimeError("backbone must expose encode() or return (y, h)")
        return h.detach(), y.detach().reshape(-1)

    def forward(self, x, engine_ids: torch.Tensor, adjacency=None,
                n_engines: Optional[int] = None) -> TwoScaleOutput:
        h, y_base = self._encode(x, adjacency)
        ids = engine_ids.reshape(-1).long()
        n = int(n_engines) if n_engines is not None else int(ids.max().item()) + 1

        if self.use_window:
            win = torch.nn.functional.softplus(self.win_head(h)).reshape(-1) + self.eps
            win = win / _segment_mean(win, ids, n)          # within-engine mean == 1
        else:
            win = torch.ones_like(y_base)

        if self.use_engine:
            pooled = _segment_pool(h, ids, n)                # (n, d)
            eng = torch.nn.functional.softplus(self.eng_head(pooled)).reshape(-1) + self.eps
            eng = eng / eng.mean()                           # global scale is not identifiable
            eng = eng[ids]
        else:
            eng = torch.ones_like(y_base)

        return TwoScaleOutput(y_hat=y_base, sigma_win=win, sigma_eng=eng, scale=win * eng)
