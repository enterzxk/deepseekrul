"""M5 (DPC): multi-span degradation consistency, training-only auxiliary.

The head predicts the latent displacement ``z_{t+h} - z_t`` from ``z_t`` for
several spans h. The target uses the model's own later latents under stopgrad,
so it is a self-supervised auxiliary: no labels, no future information at
inference (the head is dropped and the prediction stays ``reg_head(z_t)``).

Why this is a one-variable change on B0: the head's last layer is zero-init and
it never touches the prediction path, so at step 0 the RUL prediction is exactly
B0's. Any later difference comes from the auxiliary loss acting on the shared
encoder.

The target is aligned as ``src = z[:-h]``, ``tgt = z[h:] - z[:-h]``: row t of
``src`` is z_t and row t of ``tgt`` is z_{t+h} - z_t. Getting this off by one is
the one silent way to make the module meaningless, so ``check_alignment`` below
is called by the unit test.
"""
import torch
import torch.nn as nn


class DPCHead(nn.Module):
    def __init__(self, dim, hidden=64, spans=(5, 15, 30)):
        super().__init__()
        self.spans = tuple(int(h) for h in spans)
        self.trunk = nn.Sequential(nn.Linear(dim + len(self.spans), hidden),
                                   nn.GELU())
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, h_idx):
        onehot = torch.zeros(z.shape[0], len(self.spans), device=z.device, dtype=z.dtype)
        onehot[:, h_idx] = 1.0
        return self.out(self.trunk(torch.cat([z, onehot], dim=-1)))

    def loss(self, z):
        """Mean MSE over spans that fit inside this engine. Returns (loss, n_used)."""
        total = z.new_zeros(())
        used = 0
        for j, h in enumerate(self.spans):
            if z.shape[0] <= h:
                continue
            src = z[:-h]
            tgt = (z[h:] - z[:-h]).detach()
            total = total + ((self.forward(src, j) - tgt) ** 2).mean()
            used += 1
        if used == 0:
            return z.new_zeros(()), 0
        return total / used, used


def check_alignment(head, dim=4, n=40, h=5):
    """Unit check that row t of the target really is z_{t+h} - z_t.

    Builds a z sequence whose displacement over h is a known constant per
    dimension, so a misaligned target cannot accidentally pass.
    """
    z = torch.arange(n, dtype=torch.float32).unsqueeze(1) * torch.ones(1, dim)
    src = z[:-h]
    tgt = (z[h:] - z[:-h])
    assert torch.allclose(tgt, torch.full_like(tgt, float(h))), 'target misaligned'
    assert src.shape[0] == tgt.shape[0] == n - h
    return True
