"""RHSE: robust health-state estimation via differentiable weighted LS + IRLS.

Measurement model: dy_tj = H_j theta_t + noise, with per-sensor scale s_j and
Cauchy reliability weights w_tj. This is a post-hoc/diagnostic implementation of
the module described in the improvement plan; it has very few parameters
(N*m + N) and its reliability is anchored to measurement residuals, not to a
black-box RUL loss.
"""
import numpy as np
import torch
import torch.nn as nn


class RHSE(nn.Module):
    def __init__(self, n_sensors=14, m=3, iters=3, c=2.5, lam=1e-2,
                 H_init=None, detach_w=True):
        super().__init__()
        H = H_init.clone() if H_init is not None else 0.1 * torch.randn(n_sensors, m)
        self.H = nn.Parameter(H.float())
        self.log_s = nn.Parameter(torch.zeros(n_sensors))
        self.iters, self.c, self.lam, self.detach_w = iters, c, lam, detach_w

    def forward(self, dy, mask):
        # dy: (B,L,N); mask: (B,L)
        s = self.log_s.exp()
        Hn = self.H / self.H.norm(dim=0, keepdim=True).clamp_min(1e-6)
        Hs = Hn / s[:, None]
        z = dy / s
        I = torch.eye(Hs.size(1), device=dy.device, dtype=dy.dtype)
        w = torch.ones_like(z)
        theta = None
        for k in range(self.iters + 1):
            wk = w.detach() if (self.detach_w and k > 0) else w
            HtW = Hs.t()[None, None] * wk.unsqueeze(-2)          # (B,L,m,N)
            A = HtW @ Hs + self.lam * I                          # (B,L,m,m)
            b = HtW @ z.unsqueeze(-1)                            # (B,L,m,1)
            theta = torch.linalg.solve(A, b).squeeze(-1)         # (B,L,m)
            r = z - theta @ Hs.t()                               # (B,L,N)
            w = 1.0 / (1.0 + (r / self.c) ** 2)
        m_ = mask.unsqueeze(-1).to(dy.dtype)
        return theta * m_, w * m_, r * m_


def pca_H(dy, m=3):
    """PCA on standardized late-life dy; orient each column to late-life positive."""
    X = np.asarray(dy, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    H = vt[:m].T
    late = X[-max(1, len(X) // 10):].mean(0)
    for j in range(m):
        if np.dot(H[:, j], late) < 0:
            H[:, j] *= -1.0
    return H.astype(np.float32)
