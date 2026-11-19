"""RHSE-DSPT: differentiable RHSE-TSH front-end + DSPT-V0 backbone (Phase 2 M1).

The measurement matrix H and log_s are learnable; the front-end is a soft,
clean-preserving gate. A zero-initialized linear head injects low-dimensional
theta health features into the RUL prediction. This is a compact implementation
of the plan's M1 (input purification + theta head); attention-bias injection is
left for the next iteration.
"""
import torch
import torch.nn as nn

from paper_b.models import build_dspt


class RHSE_DSPT(nn.Module):
    def __init__(self, H_init, n_sensors=14, m=3, d_model=64, layers=2,
                 dropout=0.2, span=70, n_cond=6, iters=3, c=2.5, lam=1e-2,
                 kappa=5.0):
        super().__init__()
        self.backbone = build_dspt(
            num_sensors=n_sensors, d_model=d_model, num_layers=layers,
            dropout=dropout, max_span=span, ms_mode='full', sa_mode='basic',
            hi_source='hf', hi_mono='none', use_pad_mask=True,
            use_cycle=True, n_cond=n_cond)
        self.H = nn.Parameter(torch.tensor(H_init, dtype=torch.float32))
        self.log_s = nn.Parameter(torch.zeros(n_sensors))
        self.hi_head = nn.Linear(3 * m + 2, 1)
        nn.init.zeros_(self.hi_head.weight)
        nn.init.zeros_(self.hi_head.bias)
        self.iters = iters
        self.c = c
        self.lam = lam
        self.kappa = kappa

    def forward(self, x, mask, baseline, cycle, cond, mode='dual'):
        s = self.log_s.exp().clamp(0.1, 10.0)
        Hn = self.H / self.H.norm(dim=0, keepdim=True).clamp_min(1e-6)
        Hs = Hn / s[:, None]                      # (N,m)
        dy = (x - baseline[:, None, :]) / s        # (B,L,N)
        B, L, N = dy.shape
        w = mask[:, :, None].expand(B, L, N).clone()
        theta = None
        r = None
        for _ in range(self.iters):
            HtW = Hs.t()[None, None] * w.unsqueeze(-2)   # (B,L,m,N)
            A = HtW @ Hs + self.lam * torch.eye(Hs.size(1), device=x.device)
            b = HtW @ dy.unsqueeze(-1)
            theta = torch.linalg.solve(A, b).squeeze(-1)  # (B,L,m)
            r = dy - theta @ Hs.t()
            w = (1.0 / (1.0 + (r / self.c) ** 2)) * mask[:, :, None]
        dy_hat = theta @ Hs.t()
        x_rec = baseline[:, None, :] + s * dy_hat
        w_soft = torch.sigmoid(self.kappa * (1.0 - r.abs() / self.c))
        x_clean = w_soft * x + (1.0 - w_soft) * x_rec
        out = self.backbone(x_clean, mask=mask, baseline=baseline,
                            cycle=cycle, cond=cond, mode=mode)
        m = mask.unsqueeze(-1)
        denom = mask.sum(1).clamp_min(1.0)
        last = theta[:, -1]
        slope10 = theta[:, -1] - theta[:, max(0, L - 11)]
        slope30 = theta[:, -1] - theta[:, max(0, L - 31)]
        energy = ((r ** 2) * m).sum((1, 2)) / (denom * N)
        ent = -(w * (w + 1e-6).log() + (1 - w) * (1 - w + 1e-6).log()).mean((1, 2))
        f_hi = torch.cat([last, slope10, slope30, energy[:, None], ent[:, None]], 1)
        out['prediction'] = out['prediction'] + self.hi_head(f_hi)
        out['theta'] = theta
        out['w'] = w
        out['r'] = r
        return out
