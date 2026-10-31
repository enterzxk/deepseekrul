"""Sensor-evidence likelihood over RUL bins (ECS design section 1.2).

The backbone is the project's frozen encoder (arm A configuration: cycle channel on,
baseline/EMA features on, graph and dual head off) - unchanged, so that any difference
between this model and arm A is attributable to the fusion, not to the encoder. Only
the regression head is replaced by a 32-way bin classifier.

Training target: cross-entropy on the bin containing the window's remaining life
(capped at 125, the registered convention).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from candidates.model import CandidateModel  # noqa: E402

CAP = 125.0
N_BINS = 32


def rul_to_bin(rul, cap=CAP, n_bins=N_BINS):
    r = np.clip(np.asarray(rul, dtype=float), 0.0, cap)
    idx = np.floor(r / cap * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def bin_centers(cap=CAP, n_bins=N_BINS):
    edges = np.linspace(0.0, cap, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


class BinLikelihood(nn.Module):
    """Encoder (arm A) + 32-way head. forward returns logits (B, 32)."""

    def __init__(self, n_bins=N_BINS, d_model=64, num_layers=1, dropout=0.15):
        super().__init__()
        self.backbone = CandidateModel(
            num_sensors=14, window_size=50, d_model=d_model, num_layers=num_layers,
            dropout=dropout, use_baseline=True, use_ema_features=True, use_cycle=True,
            use_graph=False, use_dual_head=False, max_rul=CAP)
        self.head = nn.Linear(d_model, n_bins)
        self.n_bins = n_bins

    def forward(self, x, cond=None, extras=None):
        out = self.backbone(x, cond=cond, extras=extras)
        return self.head(out["embedding"])

    @torch.no_grad()
    def prob(self, x, cond=None, extras=None):
        logits = self.forward(x, cond, extras)
        return torch.softmax(logits, dim=-1)


def log_prob_np(model, x, extras, device="cpu"):
    model.eval()
    model.to(device)
    with torch.no_grad():
        lg = model(torch.from_numpy(x).to(device), cond=None,
                   extras=torch.from_numpy(extras).to(device))
        lp = torch.log_softmax(lg, dim=-1).cpu().numpy()
    return lp
