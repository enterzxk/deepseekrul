"""Physics-guided losses for remaining useful life regression.

Production loss for the C-MAPSS safety-critical RUL task.  It preserves the
original PhysicsGuidedLoss MSE + asymmetric late-penalty behaviour and adds an
optional trajectory monotonicity term.

    L_total = mse_weight * MSE
              + asymmetric_weight * L_Asym
              + lambda_mono * L_Mono

    L_Asym = mean( late_weight * d^2  if d > 0 else early_weight * d^2 )
    L_Mono = mean( max(0, r_t - r_{t-1}) )

where d = y_pred - y_true.  A positive d is a late (dangerous) over-estimate.
The monotonic term is computed only when a dense per-cycle trajectory is
supplied.  With a scalar prediction the term is exactly zero; it is never
approximated from shuffled batch order.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _as_prediction_views(
    y_pred: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return terminal prediction (B, 1) and optional trajectory (B, L)."""
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


def _as_terminal_target(y_true: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reduce scalar or trajectory targets to the terminal (B, 1) value."""
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


def _as_trajectory(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError("trajectory must be a torch.Tensor.")
    if values.ndim == 1:
        return values.reshape(1, -1)
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[-1] == 1:
        return values.squeeze(-1)
    raise ValueError(
        "trajectory must have shape (B, L) or (B, L, 1); "
        f"got {tuple(values.shape)}."
    )


class PhysicsGuidedLoss(nn.Module):
    """Composite RUL loss with asymmetric and monotonic safety terms.

    Parameters
    ----------
    mse_weight:
        Coefficient of the standard MSE term.
    asymmetric_weight:
        Coefficient of the asymmetric term (lambda_1 in the paper).
    late_weight:
        Weight gamma applied to squared late errors (y_pred > y_true).
    early_weight:
        Weight applied to squared early errors (y_pred <= y_true).
    reduction:
        'mean' (default) or 'sum' for the MSE and asymmetric terms.
    use_physics:
        If False, return pure MSE. This is the w/o Physics-Loss control.
    use_asym:
        If False, disable the asymmetric term. Used for Loss-Mono.
    use_mono:
        If False, disable the monotonic term. Used for Loss-Asym and MSE.
    lambda_mono:
        Coefficient lambda_2 of the monotonic term.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        asymmetric_weight: float = 1.0,
        late_weight: float = 2.0,
        early_weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        *,
        use_physics: bool = True,
        use_asym: bool = True,
        use_mono: bool = True,
        lambda_mono: float = 0.1,
    ) -> None:
        super().__init__()
        if mse_weight < 0.0 or asymmetric_weight < 0.0 or lambda_mono < 0.0:
            raise ValueError("Loss component weights must be non-negative.")
        if late_weight < 0.0 or early_weight < 0.0:
            raise ValueError("Error weights must be non-negative.")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'.")
        self.mse_weight = float(mse_weight)
        self.asymmetric_weight = float(asymmetric_weight)
        self.late_weight = float(late_weight)
        self.early_weight = float(early_weight)
        self.reduction = reduction
        self.use_physics = bool(use_physics)
        self.use_asym = bool(use_asym)
        self.use_mono = bool(use_mono)
        self.lambda_mono = float(lambda_mono)

    @property
    def gamma(self) -> float:
        """Alias for the asymmetric late-error weight."""
        return self.late_weight

    @property
    def needs_trajectory(self) -> bool:
        return self.use_physics and self.use_mono and self.lambda_mono > 0.0

    @staticmethod
    def _monotonic_terms(
        trajectory: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = _as_trajectory(trajectory)
        if values.shape[1] <= 1:
            zero = values.new_zeros(())
            return zero, zero

        increments = F.relu(values[:, 1:] - values[:, :-1])
        violations = (increments > 0.0).to(values.dtype)

        if lengths is None:
            return increments.mean(), violations.mean()

        if lengths.ndim != 1 or lengths.shape[0] != values.shape[0]:
            raise ValueError("lengths must have shape (B,).")
        lengths = lengths.to(device=values.device)
        positions = torch.arange(values.shape[1], device=values.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        pair_mask = valid[:, 1:] & valid[:, :-1]
        denominator = pair_mask.sum().clamp_min(1)
        mask = pair_mask.to(values.dtype)
        loss = (increments * mask).sum() / denominator
        rate = (violations * mask).sum() / denominator
        return loss, rate

    @staticmethod
    def monotonic_violation(
        trajectory: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return mean max(0, r_t - r_{t-1}) over valid adjacent pairs."""
        loss, _ = PhysicsGuidedLoss._monotonic_terms(trajectory, lengths)
        return loss

    @staticmethod
    def monotonic_violation_rate(
        trajectory: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return fraction of valid adjacent pairs with r_t > r_{t-1}."""
        _, rate = PhysicsGuidedLoss._monotonic_terms(trajectory, lengths)
        return rate

    @staticmethod
    def _align_and_flatten(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy scalar alignment helper retained for compatibility."""
        terminal_pred, _ = _as_prediction_views(y_pred)
        target = _as_terminal_target(y_true, terminal_pred.shape[0])
        if terminal_pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch after alignment: {tuple(terminal_pred.shape)} "
                f"vs {tuple(target.shape)}."
            )
        pred = terminal_pred.reshape(-1)
        target = target.reshape(-1)
        if pred.numel() == 0:
            raise ValueError("Loss input cannot be empty.")
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("Loss input contains NaN or infinite values.")
        return pred, target

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        return values.mean() if self.reduction == "mean" else values.sum()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_traj: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute the physics-guided objective.

        y_pred is normally (B, 1). A (B, L) sequence is accepted and its last
        column is used as the terminal prediction. y_pred_traj is the dense
        (B, L) trajectory used by L_Mono; when omitted, a sequence-shaped
        y_pred is reused for the monotonic term.
        """
        terminal_pred, sequence_pred = _as_prediction_views(y_pred)
        terminal_target = _as_terminal_target(y_true, terminal_pred.shape[0])
        if terminal_pred.numel() == 0 or terminal_target.numel() == 0:
            raise ValueError("Loss input cannot be empty.")
        if terminal_pred.shape != terminal_target.shape:
            raise ValueError(
                f"Shape mismatch after alignment: {tuple(terminal_pred.shape)} "
                f"vs {tuple(terminal_target.shape)}."
            )
        if not torch.isfinite(terminal_pred).all():
            raise ValueError("y_pred contains NaN or infinite values.")
        if not torch.isfinite(terminal_target).all():
            raise ValueError("y_true contains NaN or infinite values.")

        terminal_target = terminal_target.to(
            device=terminal_pred.device, dtype=terminal_pred.dtype
        )
        delta = terminal_pred - terminal_target
        squared_error = delta.square()
        mse = self._reduce(squared_error)

        if self.use_physics and self.use_asym and self.asymmetric_weight > 0.0:
            weights = torch.where(
                delta > 0.0,
                terminal_pred.new_full((), self.late_weight),
                terminal_pred.new_full((), self.early_weight),
            )
            asymmetric = self._reduce(weights * squared_error)
        else:
            asymmetric = terminal_pred.new_zeros(())

        trajectory: Optional[torch.Tensor] = sequence_pred
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

        if self.use_physics and self.use_mono and trajectory is not None:
            monotonic, mono_rate = self._monotonic_terms(trajectory, lengths)
            monotonic_available = 1.0
        else:
            monotonic = terminal_pred.new_zeros(())
            mono_rate = terminal_pred.new_zeros(())
            monotonic_available = 0.0

        if not self.use_physics:
            total = self.mse_weight * mse
        else:
            total = self.mse_weight * mse
            if self.use_asym:
                total = total + self.asymmetric_weight * asymmetric
            if self.use_mono:
                total = total + self.lambda_mono * monotonic

        if not return_components:
            return total

        late_mask = delta > 0.0
        components = {
            "total": total.detach(),
            "mse": mse.detach(),
            "asymmetric": asymmetric.detach(),
            "monotonic": monotonic.detach(),
            "mono_violation_rate": mono_rate.detach(),
            "monotonic_available": terminal_pred.new_tensor(monotonic_available),
            "late_fraction": late_mask.to(terminal_pred.dtype).mean().detach(),
            "severe_late_count_gt10": (delta > 10.0).sum().detach(),
            "signed_bias": delta.mean().detach(),
        }
        return total, components

    def extra_repr(self) -> str:
        return (
            f"use_physics={self.use_physics}, use_asym={self.use_asym}, "
            f"use_mono={self.use_mono}, mse_weight={self.mse_weight}, "
            f"asymmetric_weight={self.asymmetric_weight}, "
            f"late_weight={self.late_weight}, early_weight={self.early_weight}, "
            f"lambda_mono={self.lambda_mono}, reduction={self.reduction}"
        )




class StateAwarePhysicsLoss(nn.Module):
    """Risk-sensitive RUL loss in real cycle units.

    Predictions and targets are normalized to [0, 1] and multiplied by
    target_scale before computing SmoothL1 and asymmetric risk terms.
    """

    def __init__(
        self,
        target_scale: float = 125.0,
        gamma: float = 2.0,
        eta: float = 2.0,
        tau: float = 40.0,
        base_weight: float = 1.0,
        asymmetric_weight: float = 1.0,
        smoothl1_beta: float = 5.0,
        return_mean: bool = True,
        base_on_normalized: bool = False,
    ) -> None:
        super().__init__()
        if target_scale <= 0.0 or tau <= 0.0:
            raise ValueError("target_scale and tau must be positive.")
        if gamma < 0.0 or eta < 0.0:
            raise ValueError("gamma and eta must be non-negative.")
        if base_weight < 0.0 or asymmetric_weight < 0.0:
            raise ValueError("Loss component weights must be non-negative.")
        self.target_scale = float(target_scale)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.tau = float(tau)
        self.base_weight = float(base_weight)
        self.asymmetric_weight = float(asymmetric_weight)
        self.smoothl1_beta = float(smoothl1_beta)
        self.return_mean = bool(return_mean)
        self.base_on_normalized = bool(base_on_normalized)

    @staticmethod
    def _flatten(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred = prediction.reshape(-1)
        true = target.reshape(-1)
        if pred.numel() != true.numel() or pred.numel() == 0:
            raise ValueError("Prediction and target must have equal non-zero size.")
        if not torch.isfinite(pred).all() or not torch.isfinite(true).all():
            raise ValueError("Non-finite loss input.")
        return pred, true

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_norm, true_norm = self._flatten(prediction, target)
        pred = pred_norm * self.target_scale
        true = true_norm * self.target_scale
        delta = pred - true

        if self.base_on_normalized:
            base = F.smooth_l1_loss(
                pred_norm,
                true_norm,
                beta=self.smoothl1_beta / self.target_scale,
                reduction="mean",
            )
        else:
            base = F.smooth_l1_loss(
                pred,
                true,
                beta=self.smoothl1_beta,
                reduction="mean",
            )
        state_weight = 1.0 + self.eta * torch.exp(-true / self.tau)
        weights = torch.where(
            delta > 0.0,
            self.gamma * state_weight,
            torch.ones_like(state_weight),
        )
        asymmetric_sq = weights * delta.square()
        asymmetric = (
            asymmetric_sq.mean() if self.return_mean else asymmetric_sq.sum()
        )
        total = self.base_weight * base + self.asymmetric_weight * asymmetric

        if not return_components:
            return total
        late = delta > 0.0
        components = {
            "total": total.detach(),
            "base_smoothl1": base.detach(),
            "asymmetric": asymmetric.detach(),
            "late_fraction": late.to(delta.dtype).mean().detach(),
            "severe_late_fraction_gt10": (delta > 10.0).to(delta.dtype).mean().detach(),
            "mean_bias_cycles": delta.mean().detach(),
        }
        return total, components


def composite_validation_score(
    rmse: float,
    nasa_score: float,
    lambda_selection: float = 0.5,
) -> float:
    """Composite model-selection criterion used for early stopping."""
    return float(rmse + lambda_selection * np.log1p(max(float(nasa_score), 0.0)))


def risk_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Risk-oriented metrics in real RUL cycles."""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.size != pred.size or true.size == 0:
        raise ValueError("y_true and y_pred must have equal non-zero size.")
    delta = pred - true
    return {
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "mae": float(np.mean(np.abs(delta))),
        "mean_bias": float(np.mean(delta)),
        "late_ratio": float(np.mean(delta > 0.0)),
        "severe_late_ratio": float(np.mean(delta > 10.0)),
        "nasa_score": float(
            np.sum(
                np.where(
                    delta < 0.0,
                    np.expm1(-delta / 13.0),
                    np.expm1(delta / 10.0),
                )
            )
        ),
    }


__all__ = ["PhysicsGuidedLoss", "StateAwarePhysicsLoss", "composite_validation_score", "risk_metrics"]
