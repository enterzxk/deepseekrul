"""Self-check gate for STG-MambaNet and PhysicsGuidedRULLoss.

Checks:
1. All four ablation configurations run forward, loss, and backward.
2. Prediction is exactly (B, 1); trajectory is exactly (B, L).
3. The adaptive adjacency is (N, N), non-negative, and row-stochastic.
4. E1/E2 gradients exist and are non-zero for graph-enabled variants.
5. Losses and all parameter gradients are finite.
6. Asymmetric late errors are penalised more than equal early errors.
7. The monotonic term responds to an increasing RUL trajectory.

Run from the project root:
    python sanity_check.py --device cuda
"""

from __future__ import annotations

import argparse

import torch

from loss import PhysicsGuidedRULLoss
from model import STG_MambaNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STG-MambaNet sanity check.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--num-sensors", type=int, default=14)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def check_config(
    name: str,
    use_gcn: bool,
    use_ssm: bool,
    use_physics: bool,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    d_model: int,
) -> dict:
    torch.manual_seed(42)
    model = STG_MambaNet(
        num_sensors=x.shape[-1],
        seq_len=x.shape[1],
        d_model=d_model,
        use_gcn=use_gcn,
        use_ssm=use_ssm,
        dropout=0.1,
    ).to(device)
    model.train()

    criterion = PhysicsGuidedRULLoss(use_physics=use_physics).to(device)
    prediction, trajectory = model(x, return_trajectory=True)
    total, components = criterion(
        prediction,
        y,
        y_pred_traj=trajectory,
        return_components=True,
    )

    assert tuple(prediction.shape) == (x.shape[0], 1), (
        name,
        prediction.shape,
    )
    assert tuple(trajectory.shape) == (x.shape[0], x.shape[1]), (
        name,
        trajectory.shape,
    )
    assert torch.isfinite(prediction).all(), name
    assert torch.isfinite(trajectory).all(), name
    assert torch.isfinite(total), name

    total.backward()

    grad_count = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            grad_count += 1
    assert grad_count > 0, name

    adjacency = None
    e1_grad_norm = None
    if use_gcn:
        adjacency = model.get_adjacency_matrix(detach=True)
        assert tuple(adjacency.shape) == (x.shape[-1], x.shape[-1]), name
        assert torch.all(adjacency >= 0.0), name
        row_sums = adjacency.sum(dim=-1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6
        ), (name, row_sums)
        assert model.spatial_gcn.E1.grad is not None, name
        assert model.spatial_gcn.E2.grad is not None, name
        e1_grad_norm = float(model.spatial_gcn.E1.grad.norm())
        assert e1_grad_norm > 0.0, (name, e1_grad_norm)
    else:
        try:
            model.get_adjacency_matrix()
        except RuntimeError:
            pass
        else:
            raise AssertionError("w/o GCN model unexpectedly exported an adjacency.")

    if use_physics:
        assert float(components["monotonic_available"]) > 0.0, name
    else:
        assert float(components["asymmetric"]) == 0.0, name
        assert float(components["monotonic"]) == 0.0, name

    result = {
        "name": name,
        "prediction": tuple(prediction.shape),
        "trajectory": tuple(trajectory.shape),
        "adjacency": None if adjacency is None else tuple(adjacency.shape),
        "loss": float(total.detach()),
        "mse": float(components["mse"]),
        "asym": float(components["asymmetric"]),
        "mono": float(components["monotonic"]),
        "e1_grad_norm": e1_grad_norm,
        "parameters": model.num_parameters,
    }
    model.zero_grad(set_to_none=True)
    return result


def check_loss_properties(device: torch.device) -> None:
    target = torch.tensor([[50.0]], device=device)
    criterion = PhysicsGuidedRULLoss(use_physics=True)

    early = criterion(torch.tensor([[45.0]], device=device), target)
    late = criterion(torch.tensor([[55.0]], device=device), target)
    assert float(late) > float(early), (early, late)

    # MSE 25 + lambda_asym 0.5 * (gamma 2 * 25) = 50 for a late error;
    # 25 + 0.5 * 25 = 37.5 for the equal-magnitude early error.
    assert torch.isclose(early, torch.tensor(37.5, device=device)), early
    assert torch.isclose(late, torch.tensor(50.0, device=device)), late

    mse_only = PhysicsGuidedRULLoss(use_physics=False)
    assert torch.isclose(
        mse_only(late.new_tensor([[55.0]]), target),
        torch.tensor(25.0, device=device),
    )

    scalar = torch.tensor([[60.0]], device=device)
    flat = torch.tensor([[60.0, 60.0, 60.0]], device=device)
    increasing = torch.tensor([[50.0, 55.0, 60.0]], device=device)
    mono_flat = criterion.monotonic_violation(flat)
    mono_increasing = criterion.monotonic_violation(increasing)
    assert float(mono_flat) == 0.0, mono_flat
    assert float(mono_increasing) > 0.0, mono_increasing
    total_flat, _ = criterion(
        scalar, torch.tensor([[60.0]], device=device), y_pred_traj=flat,
        return_components=True,
    )
    total_inc, _ = criterion(
        scalar, torch.tensor([[60.0]], device=device), y_pred_traj=increasing,
        return_components=True,
    )
    assert float(total_inc) > float(total_flat), (total_flat, total_inc)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(42)
    x = torch.randn(args.batch_size, args.seq_len, args.num_sensors, device=device)
    y = torch.rand(args.batch_size, 1, device=device) * 125.0

    configs = [
        ("Full", True, True, True),
        ("w/o A-GCN", False, True, True),
        ("w/o SSM", True, False, True),
        ("w/o Physics", True, True, False),
    ]
    rows = [
        check_config(
            name=name,
            use_gcn=use_gcn,
            use_ssm=use_ssm,
            use_physics=use_physics,
            x=x,
            y=y,
            device=device,
            d_model=args.d_model,
        )
        for name, use_gcn, use_ssm, use_physics in configs
    ]
    check_loss_properties(device)

    header = (
        f"{'variant':<14} {'pred':<10} {'traj':<10} {'adj':<9} "
        f"{'loss':>9} {'mse':>8} {'asym':>8} {'mono':>8} "
        f"{'E1grad':>9} {'params':>8}"
    )
    print(f"device={device}  batch={args.batch_size}  seq_len={args.seq_len}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<14} {str(row['prediction']):<10} "
            f"{str(row['trajectory']):<10} {str(row['adjacency']):<9} "
            f"{row['loss']:9.4f} {row['mse']:8.4f} {row['asym']:8.4f} "
            f"{row['mono']:8.4f} "
            f"{'n/a' if row['e1_grad_norm'] is None else format(row['e1_grad_norm'], '.4f'):>9} "
            f"{row['parameters']:8d}"
        )
    print("All sanity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
