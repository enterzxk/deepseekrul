"""Train only the PU-Net heads; the backbone stays frozen."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import PUNetModel


def pinball(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    delta = target - pred
    return torch.mean(torch.maximum(q * delta, (q - 1.0) * delta))


def train_heads(
    model: PUNetModel,
    batches,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    lam: float = 0.5,
    gamma: float = 1e-4,
    delta: float = 0.1,
    device: str = "cpu",
    point_loss_fn=None,
) -> dict:
    """Fit the scale and bias heads.

    Args:
        model: a PUNetModel whose backbone is already frozen.
        batches: iterable of dicts with keys x, adjacency (optional), feats, y.
        point_loss_fn: optional callable (y_hat, y) -> loss; defaults to MSE.
    """
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("backbone must be frozen before training the heads")

    model.to(device)
    params = model.trainable_parameters()
    if not params:
        raise RuntimeError("no trainable parameters; both heads are disabled")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    history = []
    for epoch in range(1, epochs + 1):
        model.train(True)
        total = 0.0
        n_batches = 0
        for batch in batches:
            x = batch["x"].to(device)
            adjacency = batch.get("adjacency")
            adjacency = adjacency.to(device) if adjacency is not None else None
            feats = batch["feats"].to(device)
            y = batch["y"].to(device).reshape(-1)

            out = model(x, adjacency, feats)
            if point_loss_fn is None:
                point = F.mse_loss(out.y_hat, y)
            else:
                point = point_loss_fn(out.y_hat, y)

            normalised = (y - out.y_hat) / out.w
            q_lo, q_hi = delta / 2.0, 1.0 - delta / 2.0
            scale_loss = pinball(torch.zeros_like(normalised), normalised, q_lo) + pinball(
                torch.zeros_like(normalised), normalised, q_hi
            )
            reg = sum(p.pow(2).sum() for p in params)

            loss = point + lam * scale_loss + gamma * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach())
            n_batches += 1

        history.append({"epoch": epoch, "loss": total / max(1, n_batches)})
    return {"history": history, "final_loss": history[-1]["loss"] if history else float("nan")}


def fit_heads(
    h: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
    y: torch.Tensor,
    y_base: torch.Tensor,
    *,
    d_model: int,
    epochs: int = 200,
    lr: float = 1e-2,
    lam: float = 0.5,
    gamma: float = 1e-4,
    delta: float = 0.1,
    device: str = "cpu",
    seed: int = 0,
    use_scale_head: bool = True,
    use_bias_head: bool = True,
    scale_simple: bool = True,
):
    """Fit the two heads on precomputed representation tensors.

    This is the tensor-level counterpart of :func:`train_heads`; it is used by
    scripts/run_punet_stage.py once the frozen backbone has produced h.
    """
    from .heads import BiasHead, ScaleHead

    torch.manual_seed(seed)
    h = h.to(device)
    sigma = sigma.to(device)
    tau = tau.to(device)
    y = y.to(device).reshape(-1)
    y_base = y_base.to(device).reshape(-1)

    scale_head = ScaleHead(d_model, hidden=64, simple=scale_simple).to(device) if use_scale_head else None
    bias_head = BiasHead(d_model, hidden=64).to(device) if use_bias_head else None
    params = []
    if scale_head is not None:
        params += list(scale_head.parameters())
    if bias_head is not None:
        params += list(bias_head.parameters())
    if not params:
        raise ValueError("both heads are disabled")

    opt = torch.optim.AdamW(params, lr=lr)
    q_lo, q_hi = delta / 2.0, 1.0 - delta / 2.0
    history = []
    for epoch in range(1, epochs + 1):
        w = torch.ones_like(y) if scale_head is None else scale_head(h, sigma, tau)
        b = torch.zeros_like(y) if bias_head is None else bias_head(h, sigma)
        y_hat = y_base + b

        point = torch.mean((y_hat - y) ** 2)
        normalised = (y - y_hat) / w
        scale_loss = pinball(torch.zeros_like(normalised), normalised, q_lo) + pinball(
            torch.zeros_like(normalised), normalised, q_hi
        )
        reg = sum(p.pow(2).sum() for p in params)
        loss = point + lam * scale_loss + gamma * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))

    with torch.no_grad():
        w = torch.ones_like(y) if scale_head is None else scale_head(h, sigma, tau)
        b = torch.zeros_like(y) if bias_head is None else bias_head(h, sigma)
    return {
        "scale_head": scale_head,
        "bias_head": bias_head,
        "w": w.cpu(),
        "y_hat": (y_base + b).cpu(),
        "loss_history": history,
        "final_loss": history[-1] if history else float("nan"),
    }
