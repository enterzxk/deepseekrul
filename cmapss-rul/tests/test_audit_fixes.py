"""Audit regression tests for the Fix-1..Fix-7 items (Stage 2 review)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candidates.model import soft_nasa_loss  # noqa: E402
from model import CRGraphTransformer  # noqa: E402


def test_soft_nasa_zero_at_origin() -> None:
    d = torch.zeros(8, 1, requires_grad=True)
    loss = soft_nasa_loss(d)
    assert abs(float(loss)) < 1e-6, loss


def test_soft_nasa_monotone_and_asymmetric() -> None:
    def at(value: float) -> float:
        return float(soft_nasa_loss(torch.tensor([[value]])))

    # U-shaped and strictly increasing in |d| on each side
    assert at(0.0) < at(10.0) < at(20.0) < at(40.0)
    assert at(0.0) < at(-10.0) < at(-20.0) < at(-40.0)
    # over-estimating RUL (late) must cost more than under-estimating (early)
    for magnitude in (10.0, 20.0, 40.0):
        assert at(magnitude) > at(-magnitude), (magnitude, at(magnitude), at(-magnitude))


def test_soft_nasa_severe_error_keeps_gradient() -> None:
    for magnitude in (120.0, 200.0):
        d = torch.tensor([[magnitude]], requires_grad=True)
        loss = soft_nasa_loss(d)
        loss.backward()
        assert torch.isfinite(loss), loss
        assert d.grad is not None and float(d.grad.abs()) > 0.0, d.grad


def _make_model(tail_mode: str) -> CRGraphTransformer:
    model = CRGraphTransformer(
        num_sensors=14, window_size=50, d_model=32, num_heads=4,
        num_transformer_layers=1, dropout=0.0, use_crgcn=False, use_spatial=False,
        use_synergy=True, tail_window=5, tail_mode=tail_mode,
    )
    model.eval()
    with torch.no_grad():
        # strongly favour k = 0 (the newest step)
        model.tail_logits.copy_(torch.tensor([20.0, -5.0, -5.0, -5.0, -5.0]))
        model.tail_beta.fill_(-20.0)  # sigmoid -> ~0, so no -k correction
    return model


def test_tail_smoothing_uses_the_newest_step() -> None:
    model = _make_model("fixed")
    # old -> new, decreasing degradation: newest value is 60
    traj = torch.tensor([[100.0, 90.0, 80.0, 70.0, 60.0]])
    terminal = model._smooth_terminal(traj)
    assert abs(float(terminal) - 60.0) < 1e-3, float(terminal)


def test_tail_smoothing_fixed_slope_direction() -> None:
    model = _make_model("fixed")
    with torch.no_grad():
        # equal weights, no beta: each y~_{L-k} must lose exactly k cycles
        model.tail_logits.zero_()
    traj = torch.tensor([[100.0, 90.0, 80.0, 70.0, 60.0]])
    terminal = model._smooth_terminal(traj)
    # equal weights, no beta: y~_{L-k} - k for k = 4,3,2,1,0
    #   = (100-4) + (90-3) + (80-2) + (70-1) + (60-0), averaged
    expected = (96.0 + 87.0 + 78.0 + 69.0 + 60.0) / 5.0
    assert abs(float(terminal) - expected) < 1e-3, (float(terminal), expected)


if __name__ == "__main__":
    test_soft_nasa_zero_at_origin()
    test_soft_nasa_monotone_and_asymmetric()
    test_soft_nasa_severe_error_keeps_gradient()
    test_tail_smoothing_uses_the_newest_step()
    test_tail_smoothing_fixed_slope_direction()
    print("all audit tests passed")
