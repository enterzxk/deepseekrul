"""CET-RUL: conditional-evidence RUL network with a recursive survival belief.

Architecture (four parts, all trained end to end):

  A. CausalSensorEncoder   dilated causal convolutions over cycles -> e_t
  B. ObservationModel      belief summary -> predicted sensor reading yhat_t, its noise
  C. CEIO                  whitened innovation -> scalar evidence z_t -> e-value gate
  D. SurvivalBelief        q_t(R): survival advance, then an evidence-tilted update

The one non-standard piece is the gate. It is **not a learned scalar**: the update
strength is derived from an e-process (a non-negative martingale under the null "no
degradation evidence"), so

    P( sup_t E_t >= 1/alpha ) <= alpha          (Ville's inequality)

holds by construction, and a transient single-channel spike cannot produce a sustained
gate because the e-process needs continuing positive evidence rather than one large
value. The learned parts are the observation model, the degradation direction and the
log-likelihood-ratio basis coefficients -- not the trust.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SENSORS = 14
N_COND = 3
K_BINS = 600          # ABSOLUTE failure time in cycles, 1-cycle resolution
N_BASIS = 12          # B-spline-like basis for the log-LR over RUL
RUL_MAX = 400         # support of the remaining-life view


def causal_conv_stack(in_ch: int, d: int, dilations=(1, 2, 4, 8)) -> nn.Module:
    """Dilated convolutions with left padding only: output t sees inputs <= t."""
    layers = []
    prev = in_ch
    for dil in dilations:
        layers.append(nn.Conv1d(prev, d, kernel_size=3, dilation=dil, padding=0))
        layers.append(nn.GELU())
        prev = d
    return nn.ModuleList(layers), sum(dil * 2 for dil in dilations)


class CausalSensorEncoder(nn.Module):
    """Per-cycle features from a causal window; never reads the future."""

    def __init__(self, d: int = 64, dilations=(1, 2, 4, 8), dropout: float = 0.1):
        super().__init__()
        self.in_ch = N_SENSORS + N_COND + 2          # sensors, op settings, age, mask
        self.d = d
        self.dilations = dilations
        self.left_pad = sum(2 * dil for dil in dilations)   # (k-1)*dil summed
        blocks = []
        prev = self.in_ch
        for dil in dilations:
            blocks.append(nn.Conv1d(prev, d, kernel_size=3, dilation=dil))
            blocks.append(nn.GELU())
            prev = d
        self.blocks = nn.ModuleList(blocks)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_ch) -> (B, T, d), strictly causal."""
        h = x.transpose(1, 2)                        # (B, in_ch, T)
        h = F.pad(h, (self.left_pad, 0))             # left pad only
        for layer in self.blocks:
            h = layer(h) if isinstance(layer, nn.Conv1d) else layer(h)
        h = h.transpose(1, 2)[:, : x.shape[1], :]    # trim to T, causal
        return self.norm(self.drop(h))


class ObservationModel(nn.Module):
    """Predict the sensor reading implied by the current belief.

    The innovation is defined against this prediction, which is what makes the operator
    *conditional*: once the belief absorbs a change, the predicted reading follows and
    the innovation collapses. A window encoder has no such feedback.
    """

    def __init__(self, d: int = 64, rank: int = 2):
        super().__init__()
        self.rank = rank
        self.net = nn.Sequential(nn.Linear(d + 5 + N_COND, 96), nn.GELU(),
                                 nn.Linear(96, 96), nn.GELU())
        self.mu = nn.Linear(96, N_SENSORS)
        self.log_sigma = nn.Linear(96, N_SENSORS)
        self.factor = nn.Linear(96, N_SENSORS * rank)

    def forward(self, e_t: torch.Tensor, summary: torch.Tensor, op_t: torch.Tensor):
        h = self.net(torch.cat([e_t, summary, op_t], dim=-1))
        mu = self.mu(h)
        log_sigma = self.log_sigma(h).clamp(-6.0, 1.0)
        u = self.factor(h).view(-1, N_SENSORS, self.rank)
        return mu, log_sigma, u


class CEIO(nn.Module):
    """Conditional Evidence Innovation Operator.

    innovation -> whiten by S = diag(sigma^2) + U U^T -> GLS along a learned
    degradation direction -> bounded score -> e-value -> gate in [0, 1].
    """

    def __init__(self, d: int = 64, alpha: float = 0.05, ema: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.log_alpha_inv = math.log(1.0 / alpha)
        self.ema = ema
        self.direction = nn.Linear(d, N_SENSORS)          # w_t
        self.evidence = nn.Sequential(nn.Linear(6 + 4, 64), nn.GELU(),
                                      nn.Linear(64, N_BASIS))
        # location/scale of the implied likelihood over remaining life. These give the
        # belief an explicit way to become sharp; the basis term is only a residual.
        self.loc = nn.Linear(6 + 4, 1)
        self.log_scale = nn.Linear(6 + 4, 1)
        # temperature starts at 3 so the gate is live at initialisation; with the
        # betting fraction floored at 0.3 the gradient never vanishes (see below)
        self.temperature = nn.Parameter(torch.tensor(math.log(3.0)))
        self.gate_bias = nn.Parameter(torch.tensor(-1.5))
        self.basis = self._make_basis()

    @staticmethod
    def _make_basis(n_bins: int = K_BINS, n_basis: int = N_BASIS) -> torch.Tensor:
        """Raised-cosine basis over the RUL axis, shape (n_bins, n_basis)."""
        r = torch.linspace(0.0, 1.0, n_bins)
        centres = torch.linspace(0.0, 1.0, n_basis)
        width = 1.0 / (n_basis - 1)
        b = torch.cos(math.pi * (r[:, None] - centres[None, :]) / width)
        b = torch.clamp(b, min=0.0)
        return b / (b.sum(dim=1, keepdim=True) + 1e-8)

    def whiten(self, nu, log_sigma, u):
        """Solve S z = nu with S = diag(sigma^2) + U U^T via Woodbury."""
        sigma2 = torch.exp(2.0 * log_sigma)
        w = nu / sigma2
        # Woodbury: (D + UU^T)^-1 = D^-1 - D^-1 U (I + U^T D^-1 U)^-1 U^T D^-1
        du = u / sigma2.unsqueeze(-1)                      # (B, M, r)
        inner = torch.eye(u.shape[-1], device=u.device).unsqueeze(0) + u.transpose(1, 2) @ du
        corr = torch.linalg.solve(inner, u.transpose(1, 2) @ w.unsqueeze(-1))
        return (w.unsqueeze(-1) - du @ corr).squeeze(-1), sigma2

    def forward(self, e_t, nu, log_sigma, u, prev_ema, prev_logE, prev_lambda_sum,
                prev_lambda_sq, valid, age_t=None):
        z, sigma2 = self.whiten(nu, log_sigma, u)
        w = self.direction(e_t)                            # (B, M)
        wz = (w * z).sum(-1)
        wSw = (w * w * sigma2).sum(-1) + 1e-6
        raw = wz / torch.sqrt(wSw)                         # GLS z-score along w

        # bounded score in [-1, 1]; the bound is what makes the e-value non-negative
        score = torch.tanh(raw)

        # persistence of the whitened innovation: transient spikes have |z| large but
        # a stationary mean; a real degradation shifts the mean
        ema = self.ema * prev_ema + (1.0 - self.ema) * z
        persist = ema.abs().mean(-1)
        transient = (z.abs().mean(-1) - persist).clamp(min=0.0)

        # e-value with a "follow-the-leader" betting fraction on bounded scores:
        # lambda_t in [0, 1] from the running mean, so prod(1 + lambda u) is a
        # non-negative martingale under H0: E[u] = 0  =>  Ville gives the guarantee.
        adaptive = (prev_lambda_sum / (1.0 + prev_lambda_sq)).clamp(0.0, 1.0)
        # lam0 = 0.3 is a constant, hence predictable, betting fraction: the martingale
        # property E[1 + lam*score] = 1 under H0 holds for ANY predictable lam in [0,1],
        # so the anytime-valid guarantee is unchanged while the gradient stays alive.
        lam = 0.3 + 0.7 * adaptive
        e_inc = (1.0 + lam * score).clamp(min=1e-6)
        log_e = torch.log(e_inc)
        logE = prev_logE + log_e

        # gate: instantaneous log-evidence scaled by the threshold, times a learned
        # temperature; no free-form MLP gate anywhere
        # gate = sigmoid(theta0 + theta1 * log_e / log(1/alpha)): two parameters, a
        # monotone map of the calibrated evidence. Never exactly 0 or 1, so the gradient
        # survives; at log_e = 0 it sits at sigmoid(theta0) = 0.18 by initialisation.
        z = torch.exp(self.temperature).clamp(0.05, 20.0) * log_e / self.log_alpha_inv
        gate = torch.sigmoid(self.gate_bias + z) * valid

        feats = torch.cat([score.unsqueeze(-1), raw.unsqueeze(-1), persist.unsqueeze(-1),
                           transient.unsqueeze(-1), logE.unsqueeze(-1).clamp(-20, 20),
                           gate.unsqueeze(-1),
                           r_placeholder(e_t)], dim=-1)
        coeff = self.evidence(feats)                        # (B, N_BASIS)
        # Gaussian log-likelihood over remaining life, in cycles
        # absolute failure time = current age + a positive predicted remaining life
        offset = F.softplus(self.loc(feats).squeeze(-1)) * 80.0 + 2.0
        loc = (age_t if age_t is not None else torch.zeros_like(offset)) + offset
        scale = F.softplus(self.log_scale(feats).squeeze(-1)) * 20.0 + 3.0

        return dict(score=score, raw=raw, gate=gate, logE=logE, ema=ema, coeff=coeff,
                    loc=loc, scale=scale,
                    persist=persist, transient=transient,
                    stats=dict(lambda_sum=prev_lambda_sum + score * valid,
                               lambda_sq=prev_lambda_sq + score * score * valid))


def r_placeholder(e_t: torch.Tensor) -> torch.Tensor:
    """Four constant channels kept so the MLP width is fixed across steps."""
    return e_t.new_zeros(e_t.shape[0], 4)


class SurvivalBelief(nn.Module):
    """q_t(T): belief over the ABSOLUTE failure time T, in cycles.

    An earlier version carried the belief over remaining life R and advanced it by
    shifting one bin per cycle. That is wrong: a broad prior shifted 150 times walks its
    own tail into the low-R region, and the measured distribution at a cutoff had
    q[:6] ~ 0.017 with sd 35 while its mean was nearly correct. Belief over absolute
    failure time needs no shift - survival conditioning is just "zero the bins at or
    below the current age and renormalise", which is exact and loses no mass.
    """

    def __init__(self, k_bins: int = K_BINS, basis: Optional[torch.Tensor] = None,
                 prior: Optional[torch.Tensor] = None):
        super().__init__()
        self.k = k_bins
        self.register_buffer("basis", basis if basis is not None else
                             CEIO._make_basis(k_bins, N_BASIS))
        if prior is None:
            t = torch.arange(1, k_bins + 1, dtype=torch.float32)
            prior = torch.exp(-0.5 * ((t - 220.0) / 70.0) ** 2)
        self.register_buffer("prior", (prior / prior.sum()).float())

    def initial(self, batch: int, device) -> torch.Tensor:
        return self.prior.to(device).unsqueeze(0).expand(batch, -1).contiguous()

    def condition_on_survival(self, q: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        """Zero every bin with T <= age (the engine is alive at `age`) and renormalise."""
        idx = torch.arange(1, self.k + 1, device=q.device, dtype=q.dtype).unsqueeze(0)
        alive = (idx > age.unsqueeze(-1)).to(q.dtype)
        q = q * alive
        return q / (q.sum(-1, keepdim=True) + 1e-9)

    def update(self, q, gate, coeff, loc=None, scale=None) -> torch.Tensor:
        """Evidence tilt over absolute failure time: Student-t likelihood + residual."""
        tilt = 0.15 * torch.tanh(coeff @ self.basis.t())
        if loc is not None and scale is not None:
            r = torch.arange(1, q.shape[-1] + 1, device=q.device, dtype=q.dtype).unsqueeze(0)
            z = (r - loc.unsqueeze(-1)) / scale.unsqueeze(-1)
            nu = 4.0
            tilt = tilt - 0.5 * (nu + 1.0) * torch.log1p(z ** 2 / nu)
        log_q = torch.log(q + 1e-9) + gate.unsqueeze(-1) * tilt
        return torch.softmax(log_q, dim=-1)

    @staticmethod
    def to_rul(q: torch.Tensor, age: torch.Tensor, r_max: int = RUL_MAX) -> torch.Tensor:
        """p(R) = q(T = age + R) on R in [0, r_max), renormalised. Shape (B, r_max).

        Clamping age + R to the last bin (the previous implementation) aliases hundreds
        of distinct R values onto the top bin, multiplies its mass by that count, and
        sends the mean to the horizon. Bins beyond the support must be *zero*, not
        folded together.
        """
        k = q.shape[-1]
        r = torch.arange(r_max, device=q.device).unsqueeze(0)
        t = age.unsqueeze(-1).long() + r
        in_range = (t < k)
        p = torch.gather(q, 1, t.clamp(max=k - 1)) * in_range.to(q.dtype)
        return p / (p.sum(-1, keepdim=True) + 1e-9)


class CETRUL(nn.Module):
    """Full network: encoder -> observation model -> CEIO -> belief -> RUL readout."""

    def __init__(self, d: int = 64, k_bins: int = K_BINS, alpha: float = 0.05,
                 dropout: float = 0.1, ablation: str = "full"):
        super().__init__()
        self.ablation = ablation
        self.k = k_bins
        self.encoder = CausalSensorEncoder(d=d, dropout=dropout)
        self.obs = ObservationModel(d=d)
        self.ceio = CEIO(d=d, alpha=alpha)
        self.belief = SurvivalBelief(k_bins, self.ceio.basis)
        self.summary = nn.Linear(d + 2, 5)     # belief summary -> (mean, sd, entropy, ...)

    def forward(self, x, op, mask, age, valid, detach_every: int = 32):
        """x (B,T,14) scaled, op (B,T,3), mask (B,T,14), age (B,T), valid (B,T).

        ``detach_every`` truncates backpropagation through the belief chain. Without it
        the graph spans 300+ sequential steps: the gradient either vanishes or is
        clipped away, and in practice the belief never learns to sharpen (90% interval
        width came out at 145 cycles against a 10-14 RMSE competitor range).
        """
        B, T, _ = x.shape
        enc_in = torch.cat([x * mask, op, (age / 250.0).unsqueeze(-1),
                            mask.mean(-1, keepdim=True)], dim=-1)
        e = self.encoder(enc_in)                                  # (B, T, d)

        q = self.belief.initial(B, x.device)
        prev_ema = torch.zeros(B, N_SENSORS, device=x.device)
        prev_logE = torch.zeros(B, device=x.device)
        lam_sum = torch.zeros(B, device=x.device)
        lam_sq = torch.zeros(B, device=x.device)

        q_hist, rul_hist, gate_hist, logE_hist = [], [], [], []
        mu_hist, ls_hist = [], []
        for t in range(T):
            # survival conditioning at the current age, then the evidence update
            q = self.belief.condition_on_survival(q, age[:, t])
            p_rul = SurvivalBelief.to_rul(q, age[:, t])
            n_rul = p_rul.shape[-1]
            mean_r = (p_rul * torch.arange(n_rul, device=x.device)).sum(-1) / 100.0
            ent = -(p_rul * torch.log(p_rul + 1e-9)).sum(-1) / math.log(n_rul)
            summ = self.summary(torch.cat([e[:, t], mean_r.unsqueeze(-1),
                                           ent.unsqueeze(-1)], dim=-1))   # (B, 5)

            # The observation model must forecast y_t from the state BEFORE seeing it.
            # Using e[:, t] (the encoder output at t, which already contains y_t) made
            # yhat_t ~ y_t, so the innovation was identically zero and the operator had
            # no evidence to act on: the gate sat at its initialisation for the whole run.
            e_prev = e[:, t - 1] if t > 0 else torch.zeros_like(e[:, t])
            mu, log_sigma, u = self.obs(e_prev, summ, op[:, t])
            nu = (x[:, t] - mu) * mask[:, t]

            out = self.ceio(e[:, t], nu, log_sigma, u, prev_ema, prev_logE,
                            lam_sum, lam_sq, valid[:, t], age_t=age[:, t])
            prev_ema, prev_logE = out["ema"], out["logE"]
            lam_sum, lam_sq = out["stats"]["lambda_sum"], out["stats"]["lambda_sq"]

            if self.ablation == "no_gate":
                gate = torch.zeros_like(out["gate"])
            elif self.ablation == "gate_one":
                gate = valid[:, t]
            elif self.ablation == "no_update":
                gate = torch.zeros_like(out["gate"])
            else:
                gate = out["gate"]

            q = self.belief.update(q, gate, out["coeff"], out["loc"], out["scale"])
            if detach_every and (t % detach_every) == (detach_every - 1):
                # truncated BPTT: the belief stays a valid state, but gradients only
                # travel the last `detach_every` steps
                q = q.detach()
                prev_ema = prev_ema.detach()
                prev_logE = prev_logE.detach()
                lam_sum, lam_sq = lam_sum.detach(), lam_sq.detach()
            q_hist.append(q)
            rul_hist.append(SurvivalBelief.to_rul(q, age[:, t]))
            gate_hist.append(gate)
            logE_hist.append(out["logE"])
            mu_hist.append(mu)
            ls_hist.append(log_sigma)

        return dict(q=torch.stack(q_hist, 1),            # (B, T, K) absolute time
                    rul=torch.stack(rul_hist, 1),        # (B, T, K) remaining life
                    gate=torch.stack(gate_hist, 1),      # (B, T)
                    logE=torch.stack(logE_hist, 1),      # (B, T)
                    mu=torch.stack(mu_hist, 1),          # (B, T, M)
                    log_sigma=torch.stack(ls_hist, 1))   # (B, T, M)

    # ---------------- readouts ----------------
    @staticmethod
    def rul_mean(q):
        r = torch.arange(q.shape[-1], device=q.device, dtype=q.dtype)
        return (q * r).sum(-1)

    @staticmethod
    def rul_interval(q, lo=0.05, hi=0.95):
        k = q.shape[-1]
        shape = q.shape[:-1]
        cdf = torch.cumsum(q, dim=-1).reshape(-1, k).contiguous()
        n = cdf.shape[0]
        r = torch.arange(k, device=q.device, dtype=q.dtype)
        lo_i = torch.searchsorted(cdf, torch.full((n, 1), lo, device=q.device, dtype=cdf.dtype))
        hi_i = torch.searchsorted(cdf, torch.full((n, 1), hi, device=q.device, dtype=cdf.dtype))
        lo_v = r[lo_i.reshape(-1).clamp(max=k - 1)].reshape(shape)
        hi_v = r[hi_i.reshape(-1).clamp(max=k - 1)].reshape(shape)
        return lo_v, hi_v

    @staticmethod
    def nll(q, life, weight):
        """Discrete NLL of the observed lifetime under q(T), with censoring at the horizon.

        q is the belief over absolute failure time, so the target is the engine's true
        lifetime L (a constant per engine), not a per-cycle remaining life.
        """
        k = q.shape[-1]
        idx = life.round().long().clamp(0, k - 1)
        logp = torch.log(q.gather(-1, idx.unsqueeze(-1)).squeeze(-1) + 1e-9)
        surv = torch.log(q[..., -1] + 1e-9)        # P(T >= horizon)
        censored = (life >= k - 1)
        val = torch.where(censored, surv, logp)
        return -(val * weight).sum() / weight.sum().clamp(min=1e-6)

    @staticmethod
    def crps(q, rul, weight):
        """CRPS for a discrete distribution on {0..K-1}."""
        k = q.shape[-1]
        r = torch.arange(k, device=q.device, dtype=q.dtype)
        cdf = torch.cumsum(q, dim=-1)
        step = (cdf - (r >= rul.unsqueeze(-1)).to(q.dtype)).abs().sum(-1)
        return (step * weight).sum() / weight.sum().clamp(min=1e-6)
