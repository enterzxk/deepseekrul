"""Regression tests for STG-MambaNet and PhysicsGuidedLoss."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loss import PhysicsGuidedLoss  # noqa: E402
from model import STG_MambaNet  # noqa: E402


class STGMambaNetTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = STG_MambaNet(
            num_sensors=14,
            window_size=30,
            d_model=32,
            graph_embed_dim=8,
            state_dim=8,
            kernels=(3, 5),
            num_heads=4,
            dropout=0.1,
        )
        self.inputs = torch.randn(3, 30, 14)

    def test_forward_shapes_and_diagnostics(self) -> None:
        self.model.eval()
        with torch.no_grad():
            prediction, diagnostics = self.model(
                self.inputs, return_diagnostics=True
            )
        self.assertEqual(tuple(prediction.shape), (3, 1))

        adjacency = diagnostics["adjacency"]
        self.assertEqual(tuple(adjacency.shape), (14, 14))
        self.assertTrue(torch.all(adjacency >= 0.0))
        torch.testing.assert_close(
            adjacency.sum(dim=-1), torch.ones(14), atol=1e-6, rtol=1e-6
        )

        scale_weights = diagnostics["temporal_scale_weights"]
        self.assertEqual(tuple(scale_weights.shape), (3, 2))
        torch.testing.assert_close(
            scale_weights.sum(dim=-1), torch.ones(3), atol=1e-6, rtol=1e-6
        )

        branch_weights = diagnostics["fusion_branch_weights"]
        self.assertEqual(tuple(branch_weights.shape), (3, 3))
        torch.testing.assert_close(
            branch_weights.sum(dim=-1), torch.ones(3), atol=1e-6, rtol=1e-6
        )

        gate = diagnostics["fusion_channel_gate"]
        self.assertEqual(tuple(gate.shape), (3, 30, 32))
        self.assertTrue(torch.all(gate >= 0.0) and torch.all(gate <= 1.0))

    def test_unbatched_forward(self) -> None:
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(self.inputs[0])
        self.assertEqual(tuple(prediction.shape), (1,))

    def test_invalid_input_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.randn(3, 29, 14))
        with self.assertRaises(ValueError):
            self.model(torch.randn(3, 30, 13))

    def test_backward_reaches_graph_embeddings_and_ssm_parameters(self) -> None:
        self.model.train()
        prediction = self.model(self.inputs)
        target = torch.full_like(prediction, 60.0)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()

        self.assertIsNotNone(self.model.spatial_branch.E1.grad)
        self.assertIsNotNone(self.model.spatial_branch.E2.grad)
        self.assertTrue(torch.isfinite(self.model.spatial_branch.E1.grad).all())
        self.assertTrue(torch.isfinite(self.model.spatial_branch.E2.grad).all())

        first_ssm = self.model.temporal_branch.branches[0]
        self.assertIsNotNone(first_ssm.a_log.grad)
        self.assertTrue(torch.isfinite(first_ssm.a_log.grad).all())
        self.assertGreater(float(first_ssm.a_log.grad.abs().sum()), 0.0)

    def test_ablation_variants_forward(self) -> None:
        cases = [
            ("independent", "ssm", False, True),
            ("graph", "lstm", True, False),
        ]
        for spatial_mode, temporal_mode, has_adj, has_scale in cases:
            with self.subTest(spatial=spatial_mode, temporal=temporal_mode):
                model = STG_MambaNet(
                    num_sensors=14,
                    window_size=30,
                    d_model=32,
                    graph_embed_dim=8,
                    state_dim=8,
                    kernels=(3, 5),
                    num_heads=4,
                    spatial_mode=spatial_mode,
                    temporal_mode=temporal_mode,
                )
                model.eval()
                with torch.no_grad():
                    prediction, diagnostics = model(
                        self.inputs, return_diagnostics=True
                    )
                self.assertEqual(tuple(prediction.shape), (3, 1))
                self.assertEqual("adjacency" in diagnostics, has_adj)
                self.assertEqual(
                    "temporal_scale_weights" in diagnostics, has_scale
                )


    def test_trajectory_last_matches_terminal_prediction(self) -> None:
        self.model.eval()
        with torch.no_grad():
            prediction, trajectory = self.model(
                self.inputs, return_trajectory=True
            )
        self.assertEqual(tuple(prediction.shape), (3, 1))
        self.assertEqual(tuple(trajectory.shape), (3, 30))
        torch.testing.assert_close(
            trajectory[:, -1], prediction[:, 0], atol=1e-6, rtol=1e-6
        )

    def test_graph_modes_and_identity_prior(self) -> None:
        self.model.eval()
        adjacency = self.model.get_adjacency_matrix(detach=True)
        torch.testing.assert_close(
            adjacency.sum(dim=-1), torch.ones(14), atol=1e-6, rtol=1e-6
        )
        alpha = self.model.spatial_branch.graph_alpha
        self.assertIsNotNone(alpha)
        self.assertGreater(float(alpha), 0.0)
        self.assertLess(float(alpha), 1.0)

        identity = STG_MambaNet(
            num_sensors=14,
            window_size=30,
            d_model=32,
            graph_embed_dim=8,
            graph_mode="identity",
        )
        identity.eval()
        identity_adjacency = identity.get_adjacency_matrix(detach=True)
        torch.testing.assert_close(
            identity_adjacency, torch.eye(14), atol=1e-6, rtol=1e-6
        )

        random_graph = STG_MambaNet(
            num_sensors=14,
            window_size=30,
            d_model=32,
            graph_embed_dim=8,
            graph_mode="random_fixed",
        )
        random_graph.eval()
        random_adjacency = random_graph.get_adjacency_matrix(detach=True)
        torch.testing.assert_close(
            random_adjacency.sum(dim=-1), torch.ones(14), atol=1e-6, rtol=1e-6
        )

    def test_parameter_matched_lstm_is_within_three_percent(self) -> None:
        full = STG_MambaNet(num_sensors=14, window_size=30, d_model=64)
        lstm = STG_MambaNet(
            num_sensors=14,
            window_size=30,
            d_model=64,
            temporal_mode="lstm",
            lstm_hidden_size=70,
        )
        difference = abs(full.num_parameters - lstm.num_parameters)
        self.assertLessEqual(difference / full.num_parameters, 0.03)


class PhysicsGuidedLossTests(unittest.TestCase):
    def test_late_prediction_is_penalized_more(self) -> None:
        criterion = PhysicsGuidedLoss(
            mse_weight=1.0,
            asymmetric_weight=1.0,
            late_weight=2.0,
            early_weight=1.0,
        )
        target = torch.tensor([[50.0]])
        early = criterion(torch.tensor([[45.0]]), target)
        late = criterion(torch.tensor([[55.0]]), target)
        self.assertGreater(float(late), float(early))
        self.assertAlmostEqual(float(early), 50.0, places=6)
        self.assertAlmostEqual(float(late), 75.0, places=6)

    def test_component_report_and_shape_alignment(self) -> None:
        criterion = PhysicsGuidedLoss()
        prediction = torch.tensor([[50.0], [40.0], [70.0]])
        target = torch.tensor([50.0, 50.0, 50.0])
        loss, components = criterion(
            prediction, target, return_components=True
        )
        self.assertEqual(loss.ndim, 0)
        self.assertIn("mse", components)
        self.assertIn("asymmetric", components)
        self.assertIn("late_fraction", components)
        self.assertAlmostEqual(float(components["late_fraction"]), 1.0 / 3.0, places=6)

    def test_empty_input_raises(self) -> None:
        criterion = PhysicsGuidedLoss()
        with self.assertRaises(ValueError):
            criterion(torch.empty(0), torch.empty(0))


    def test_monotonic_term_and_violation_rate(self) -> None:
        criterion = PhysicsGuidedLoss(
            use_physics=True,
            use_asym=False,
            use_mono=True,
            lambda_mono=0.1,
        )
        prediction = torch.zeros(1, 1)
        target = torch.zeros(1, 1)
        trajectory = torch.tensor([[10.0, 20.0, 30.0]])
        total, components = criterion(
            prediction,
            target,
            y_pred_traj=trajectory,
            return_components=True,
        )
        self.assertAlmostEqual(float(components["monotonic"]), 10.0, places=6)
        self.assertAlmostEqual(
            float(components["mono_violation_rate"]), 1.0, places=6
        )
        self.assertAlmostEqual(float(total), 1.0, places=6)

    def test_use_physics_false_is_pure_mse(self) -> None:
        criterion = PhysicsGuidedLoss(use_physics=False)
        loss = criterion(
            torch.tensor([[55.0]]),
            torch.tensor([[50.0]]),
        )
        self.assertAlmostEqual(float(loss), 25.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
