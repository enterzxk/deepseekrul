"""PCDT-RUL network.  All arms share the identical backbone, losses and data;
they differ ONLY in how the predicted lifetime scale enters the encoder:

  plain  absolute positions only (B1)
  rel    standard relative bias B=f(dt)                       (B2)
  tau    dt/Lambda appended as an input channel               (B3)
  rand   phase bias with batch-permuted Lambda-hat            (B4)
  oracle phase bias with the true T/c (NOT deployable)        (B5)
  full   phase bias B=phi(dt/Lambda-hat, log sigma-hat)       (F)

The scale head (GRU -> truncated-lognormal lifetime NLL) is trained in every
arm, so cross-arm differences cannot come from multi-task supervision.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_L_MIN, LOG_L_MAX = math.log(50.0), math.log(500.0)
SIG_MIN, SIG_MAX = 0.05, 1.5


class ScaleHead(nn.Module):
    """Causal GRU over observed rows -> lognormal (mu, sigma) of total life.

    Returns RAW mu: clamping must never sit on the loss path or the gradient
    is zeroed while mu is outside the clamp range (initialised at 0, log-range
    is [3.9, 6.2] -- a clamp here would freeze the head forever).  Clamping to
    a sane lambda happens only where lambda feeds the attention bias.
    """

    def __init__(self, nf: int, d: int = 64, kind: str = "gru"):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.gru = nn.GRU(nf, d, batch_first=True)
            self.out = nn.Linear(d, 2)
        elif kind == "attn":
            # single-query attention pooling over the history: the gate showed
            # lifetime information is spread across the prefix (corr rises with
            # prefix length), so pooling over rows beats reading the last state
            self.proj = nn.Linear(nf, d)
            self.query = nn.Parameter(torch.randn(d) * 0.02)
            self.ln = nn.LayerNorm(d)
            self.out = nn.Linear(d, 2)
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "gru":
            h, _ = self.gru(x)
            g = h[:, -1]
        else:
            k = self.ln(self.proj(x))                       # (N, S, d)
            w = (k @ self.query).softmax(dim=1)             # (N, S)
            g = (w.unsqueeze(-1) * k).sum(dim=1)            # (N, d)
        mu, log_sig = self.out(g).unbind(-1)
        sig = log_sig.clamp(math.log(0.02), math.log(5.0)).exp()
        return mu, sig


def trunc_lognorm_nll(mu, sig, life, cyc):
    """-log[ p(T) / P(T>c) ] for lognormal(mu, sig), rows with T>c."""
    z = (life.log() - mu) / sig
    log_pdf = -life.log() - sig.log() - 0.5 * z.square() - 0.5 * math.log(2 * math.pi)
    # survival: P(T > c) = 1 - Phi((ln c - mu)/sig)
    zc = (cyc.float().log() - mu) / sig
    surv = (1.0 - 0.5 * (1.0 + torch.erf(zc / math.sqrt(2.0)))).clamp_min(1e-6)
    return -(log_pdf - surv.log()).mean()


class BiasMLP(nn.Module):
    """Per-head additive attention bias from a 1- or 2-d coordinate."""

    def __init__(self, n_in: int, n_heads: int, hidden: int = 16):
        super().__init__()
        self.n_heads = n_heads
        self.net = nn.ModuleList(
            nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, 1))
            for _ in range(n_heads))

    def forward(self, coords):
        # coords: (N, L, L, n_in); tanh bounds the logit contribution so a
        # noisy lambda-hat cannot blow up the attention pattern
        out = torch.tanh(torch.stack([m(coords).squeeze(-1) for m in self.net], dim=1))
        return out  # (N, heads, L, L)


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, bias):
        N, L, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(N, L, self.n_heads, d // self.n_heads).transpose(1, 2)
        k = k.view(N, L, self.n_heads, d // self.n_heads).transpose(1, 2)
        v = v.view(N, L, self.n_heads, d // self.n_heads).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / math.sqrt(d // self.n_heads)
        if bias is not None:
            att = att + bias
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf"))
        att = att.softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(N, L, d)
        x = x + self.proj(y)
        x = x + self.ff(self.ln2(x))
        return x


class PCDT(nn.Module):
    def __init__(self, nf: int, d: int = 64, n_layers: int = 2, n_heads: int = 4,
                 ffn: int = 256, arm: str = "full", L: int = 40,
                 scale_head: str = "gru", detach_lam: bool = False):
        super().__init__()
        self.arm = arm
        self.L = L
        self.detach_lam = detach_lam
        n_in_bias = 3 if arm == "both" else (2 if arm in ("full", "rand", "oracle") else 1)
        self.scale = ScaleHead(nf, kind=scale_head)
        self.inp = nn.Linear(nf + (1 if arm == "tau" else 0), d)
        self.pos = nn.Embedding(L, d)
        self.bias = (BiasMLP(n_in_bias, n_heads)
                     if arm in ("rel", "full", "rand", "oracle", "both") else None)
        self.blocks = nn.ModuleList(Block(d, n_heads, ffn) for _ in range(n_layers))
        self.read = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, x, life=None, cyc=None, xs=None):
        N, L, _ = x.shape
        mu, sig = self.scale(xs if xs is not None else x)   # raw mu: loss path unclamped
        if self.detach_lam:
            # Lambda-hat is trained ONLY by the lifetime NLL.  Without the cut,
            # RUL-loss gradients flow through the bias path and the head can be
            # co-opted into producing a scale estimate that helps attention
            # instead of an honest one -- the mini-matrix showed that learned
            # Lambda-hat hurts MORE than random noise, which is the signature
            # of this co-opting, not of noise.
            mu = mu.detach()
        lam = mu.clamp(LOG_L_MIN, LOG_L_MAX).exp()   # clamp only the bias path
        coords_for_eval = {"mu": mu, "sig": sig}
        if self.arm == "oracle":
            # oracle lambda is the true TOTAL lifetime (diagnostic upper bound,
            # not deployable) -- not the current age; the phase coordinate is
            # dt normalised by the lifetime scale.
            assert life is not None
            lam = life.clamp(math.exp(LOG_L_MIN), math.exp(LOG_L_MAX))
        if self.arm == "rand":
            perm = torch.randperm(N, device=x.device)
            lam_u = lam[perm]
            sig_u = sig[perm]
        else:
            lam_u, sig_u = lam, sig
        if self.arm == "tau":
            pos = torch.arange(L, device=x.device).float().view(1, L)
            tau = (pos - (L - 1)) / lam_u.clamp_min(1.0).view(N, 1)  # (N, L)
            x = torch.cat([x, tau.unsqueeze(-1)], dim=-1)
        h = self.inp(x) + self.pos.weight.unsqueeze(0)
        bias = None
        if self.bias is not None:
            pos = torch.arange(L, device=x.device).float()
            dt = pos.view(1, L, 1) - pos.view(1, 1, L)            # (1, L, L)
            if self.arm == "rel":
                coords = dt.view(1, L, L, 1).expand(N, L, L, 1)
            elif self.arm == "both":
                coords = torch.stack([dt.expand(N, L, L),
                                      dt.expand(N, L, L) / lam_u.view(N, 1, 1),
                                      sig_u.view(N, 1, 1).expand(N, L, L).log()],
                                     dim=-1)
            else:
                coords = torch.stack([dt.expand(N, L, L) / lam_u.view(N, 1, 1),
                                      sig_u.view(N, 1, 1).expand(N, L, L).log()],
                                     dim=-1)
            bias = self.bias(coords)
        for blk in self.blocks:
            h = blk(h, bias)
        mu_rul, log_sig_rul = self.read(h[:, -1]).unbind(-1)
        return mu_rul, log_sig_rul.clamp(math.log(0.5), math.log(50.0)).exp(), \
            coords_for_eval


def gauss_nll(mu, sig, y):
    return (0.5 * ((y - mu) / sig).square() + sig.log() + 0.5 * math.log(2 * math.pi)).mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
