"""Sensor-identity probe (ORDER_01 Fix 3).

If the shared per-sensor encoder plus a symmetric readout is permutation
invariant, permuting sensor channels together with their baseline must not
change the prediction. Adding a learned sensor-ID embedding must break that
invariance. Both statements are asserted here as documenting properties.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candidates.model import CandidateModel  # noqa: E402


def _build(use_sensor_id: bool) -> CandidateModel:
    torch.manual_seed(0)
    model = CandidateModel(
        num_sensors=14, window_size=50, d_model=32, num_heads=4, num_layers=1,
        dropout=0.0, use_baseline=True, use_ema_features=True, use_cycle=True,
        use_graph=False, use_dual_head=False, use_sensor_id=use_sensor_id,
    )
    model.eval()
    return model


def _permuted_inputs(seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 50, 14, generator=generator)
    baseline = x[:, :5].mean(dim=1)
    cycle = torch.full((4, 1), 0.4)
    ref_count = torch.ones(4, 1)
    extras = torch.cat([baseline, cycle, ref_count], dim=-1)
    perm = torch.tensor([3, 0, 11, 7, 1, 13, 5, 2, 9, 4, 12, 6, 10, 8])
    x_p = x[:, :, perm]
    extras_p = extras.clone()
    extras_p[:, :14] = extras[:, :14][:, perm]
    return x, extras, x_p, extras_p


def main() -> None:
    x, extras, x_p, extras_p = _permuted_inputs()

    shared = _build(use_sensor_id=False)
    with torch.no_grad():
        a = shared(x, extras=extras)["prediction"]
        b = shared(x_p, extras=extras_p)["prediction"]
    delta_shared = float((a - b).abs().max())

    identified = _build(use_sensor_id=True)
    with torch.no_grad():
        c = identified(x, extras=extras)["prediction"]
        d = identified(x_p, extras=extras_p)["prediction"]
    delta_id = float((c - d).abs().max())

    print(f"shared-encoder permutation delta: {delta_shared:.6f}")
    print(f"sensor-id encoder permutation delta: {delta_id:.6f}")
    assert delta_shared < 1e-4, delta_shared  # documented invariance
    assert delta_id > 1e-4, delta_id          # identity embedding breaks it
    print("sensor permutation probe passed")


if __name__ == "__main__":
    main()
