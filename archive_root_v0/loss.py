"""Physics-guided loss functions for safety-critical RUL regression.

The asymmetric term follows the C-MAPSS cost asymmetry. With

    d = y_pred - y_true,

a positive d means the model over-estimates remaining life ("late alarm"),
which can postpone maintenance and is treated as the dangerous error. The
loss is

    L_Asym = mean( gamma * d^2  if d > 0  else  1.0 * d^2 ),   gamma = 2.0.

The total objective is

    L_total = L_MSE + lambda_asym * L_Asym + lambda_mono * L_Mono,

with lambda_asym = 0.5 and lambda_mono = 0.1 by default. The monotonic term

    L_Mono = mean_{b,t} max(0, y_hat_{b,t} - y_hat_{b,t-1})

penalises a predicted RUL that increases as the engine ages. To compute it,
pass the dense trajectory returned by STG_MambaNet:

    prediction, trajectory = model(x, return_trajectory=True)
    loss = criterion(prediction, target, y_pred_traj=trajectory)

If only a scalar prediction is supplied, the monotonic term is exactly zero
because there is no temporal axis to regularise. It is never silently
approximated from shuffled batch order.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _as_prediction_views(y_pred: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
    """Return (terminal prediction (B, 1), optional trajectory (B, L))."""
    if not isinstance(y_pred, torch.Tensor):
        raise TypeError("y_pred must be a torch.Tensor.")
    if y_pred.ndim == 0:
        return y_pred.reshape(1, 1), None
    if y_pred.ndim == 1:
        return y_pred.reshape(-1, 1), None
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 1:
            return y_pred, None
        return y_pred[:, -1:], y_pred
    if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
        sequence = y_pred.squeeze(-1)
        return sequence[:, -1:], sequence
    raise ValueError(
        "y_pred must have shape (B,), (B, 1), (B, L), or (B, L, 1); "
        f"got {tuple(y_pred.shape)}."
    )


def _as_terminal_target(y_true: Tensor, batch_size: int) -> Tensor:
    """Reduce a scalar or trajectory target to its terminal (B, 1) value."""
    if not isinstance(y_true, torch.Tensor):
        raise TypeError("y_true must be a torch.Tensor.")
    if y_true.ndim == 0:
        return y_true.reshape(1, 1).expand(batch_size, 1)
    if y_true.ndim == 1:
        target = y_true.reshape(-1, 1)
    elif y_true.ndim == 2:
        target = y_true if y_true.shape[1] == 1 else y_true[:, -1:]
    elif y_true.ndim == 3 and y_true.shape[-1] == 1:
        target = y_true[..., 0][:, -1:]
    else:
        raise ValueError(
            "y_true must have shape (B,), (B, 1), (B, L), or (B, L, 1); "
            f"got {tuple(y_true.shape)}."
        )
    if target.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: prediction has {batch_size}, target has "
            f"{target.shape[0]}."
        )
    return target


def _as_trajectory(y_pred_traj: Tensor) -> Tensor:
    """Normalise a trajectory to (B, L) and reject ambiguous shapes."""
    if not isinstance(y_pred_traj, torch.Tensor):
        raise TypeError("y_pred_traj must be a torch.Tensor.")
    if y_pred_traj.ndim == 1:
        return y_pred_traj.reshape(1, -1)
    if y_pred_traj.ndim == 2:
        return y_pred_traj
    if y_pred_traj.ndim == 3 and y_pred_traj.shape[-1] == 1:
        return y_pred_traj.squeeze(-1)
    raise ValueError(
        "y_pred_traj must have shape (B, L) or (B, L, 1); "
        f"got {tuple(y_pred_traj.shape)}."
    )


class PhysicsGuidedRULLoss(nn.Module):
    """MSE + asymmetric late-alarm penalty + monotonic degradation prior.

    Parameters
    ----------
    use_physics:
        If False, return the standard MSE only. This is the w/o Physics-Loss
        ablation.
    gamma:
        Weight applied to squared over-estimation errors (d > 0). The paper's
        default is 2.0; under-estimation keeps weight 1.0.
    lambda_asym:
        Coefficient of L_Asym in the total objective (lambda_1).
    lambda_mono:
        Coefficient of L_Mono in the total objective (lambda_2).
    mse_weight:
        Coefficient of the standard MSE term. Kept at 1.0 for the published
        objective; exposed mainly for sensitivity studies.
    """

    def __init__(
        self,
        use_physics: bool = True,
        gamma: float = 2.0,
        lambda_asym: float = 0.5,
        lambda_mono: float = 0.1,
        mse_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if gamma <= 0.0:
            raise ValueError("gamma must be positive.")
        if lambda_asym < 0.0 or lambda_mono < 0.0:
            raise ValueError("lambda_asym and lambda_mono must be non-negative.")
        if mse_weight < 0.0:
            raise ValueError("mse_weight must be non-negative.")
        self.use_physics = bool(use_physics)
        self.gamma = float(gamma)
        self.lambda_asym = float(lambda_asym)
        self.lambda_mono = float(lambda_mono)
        self.mse_weight = float(mse_weight)

    @staticmethod
    def monotonic_violation(
        trajectory: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """Return mean max(0, r_t - r_{t-1}) over valid adjacent pairs.

        Parameters
        ----------
        trajectory:
            (B, L) or (B, L, 1) predicted RUL values ordered by increasing
            cycle.
        lengths:
            Optional (B,) valid sequence lengths for padded batches. Pairs in
            which either endpoint is padding are excluded from the mean.
        """
        values = _as_trajectory(trajectory)
        if values.shape[1] <= 1:
            return values.new_zeros(())

        increments = F.relu(values[:, 1:] - values[:, :-1])  # (B, L-1)
        if lengths is None:
            return increments.mean()

        if lengths.ndim != 1 or lengths.shape[0] != values.shape[0]:
            raise ValueError("lengths must have shape (B,).")
        lengths = lengths.to(device=values.device)
        positions = torch.arange(values.shape[1], device=values.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)
        pair_mask = valid[:, 1:] & valid[:, :-1]  # (B, L-1)
        denominator = pair_mask.sum().clamp_min(1)
        return (increments * pair_mask.to(increments.dtype)).sum() / denominator

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        y_pred_traj: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        return_components: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Compute the physics-guided objective.

        Parameters
        ----------
        y_pred:
            Terminal prediction, usually (B, 1). A (B, L) sequence is also
            accepted; its last column is used as the terminal prediction.
        y_true:
            Terminal target (B, 1) or (B,). A (B, L) target is also accepted
            and reduced to its last column.
        y_pred_traj:
            Optional dense (B, L) RUL trajectory used by L_Mono. This is the
            preferred way to activate the monotonic prior.
        lengths:
            Optional valid lengths for padded trajectories.
        return_components:
            If True, also return a detached dictionary of scalar diagnostics.

        Returns
        -------
        Tensor or (Tensor, dict[str, Tensor])
            Differentiable total loss, optionally with detached components.
        """
        terminal_pred, sequence_pred = _as_prediction_views(y_pred)
        terminal_target = _as_terminal_target(y_true, terminal_pred.shape[0])

        if terminal_pred.shape != terminal_target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch after alignment: "
                f"{tuple(terminal_pred.shape)} vs {tuple(terminal_target.shape)}."
            )
        if not torch.isfinite(terminal_pred).all():
            raise ValueError("y_pred contains NaN or infinite values.")
        if not torch.isfinite(terminal_target).all():
            raise ValueError("y_true contains NaN or infinite values.")

        terminal_target = terminal_target.to(
            device=terminal_pred.device, dtype=terminal_pred.dtype
        )
        mse = F.mse_loss(terminal_pred, terminal_target, reduction="mean")

        trajectory = None
        if y_pred_traj is not None:
            trajectory = _as_trajectory(y_pred_traj)
            if trajectory.shape[0] != terminal_pred.shape[0]:
                raise ValueError(
                    "y_pred_traj batch size does not match y_pred: "
                    f"{trajectory.shape[0]} vs {terminal_pred.shape[0]}."
                )
            if not torch.isfinite(trajectory).all():
                raise ValueError("y_pred_traj contains NaN or infinite values.")
            trajectory = trajectory.to(
                device=terminal_pred.device, dtype=terminal_pred.dtype
            )
        elif sequence_pred is not None:
            trajectory = sequence_pred.to(
                device=terminal_pred.device, dtype=terminal_pred.dtype
            )

        if self.use_physics:
            delta = terminal_pred - terminal_target
            # d > 0 is a late (dangerous) over-estimate; d <= 0 is an early,
            # conservative under-estimate.
            error_weight = torch.where(
                delta > 0.0,
                terminal_pred.new_full((), self.gamma),
                terminal_pred.new_full((), 1.0),
            )
            asymmetric = (error_weight * delta.square()).mean()

            if trajectory is None:
                monotonic = terminal_pred.new_zeros(())
                monotonic_available = 0.0
            else:
                monotonic = self.monotonic_violation(trajectory, lengths)
                monotonic_available = 1.0

            total = (
                self.mse_weight * mse
                + self.lambda_asym * asymmetric
                + self.lambda_mono * monotonic
            )
        else:
            # Exact standard MSE objective for the w/o Physics-Loss ablation.
            asymmetric = terminal_pred.new_zeros(())
            monotonic = terminal_pred.new_zeros(())
            monotonic_available = 0.0
            total = self.mse_weight * mse

        if not return_components:
            return total

        if self.use_physics:
            late_fraction = (delta > 0.0).to(terminal_pred.dtype).mean()
        else:
            late_fraction = terminal_pred.new_zeros(())
        components = {
            "total": total.detach(),
            "mse": mse.detach(),
            "asymmetric": asymmetric.detach(),
            "monotonic": monotonic.detach(),
            "late_fraction": late_fraction.detach(),
            "signed_error": (terminal_pred - terminal_target).mean().detach(),
            "monotonic_available": terminal_pred.new_tensor(monotonic_available),
        }
        return total, components

    def extra_repr(self) -> str:
        return (
            f"use_physics={self.use_physics}, gamma={self.gamma}, "
            f"lambda_asym={self.lambda_asym}, lambda_mono={self.lambda_mono}, "
            f"mse_weight={self.mse_weight}"
        )


__all__ = ["PhysicsGuidedRULLoss"]
