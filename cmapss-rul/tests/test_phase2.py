"""Phase II tests that never read official test files."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baselines import BASELINE_REGISTRY, build_baseline  # noqa: E402
from dataset import (  # noqa: E402
    OperatingConditionScaler,
    load_train_only,
    make_engine_folds,
    prepare_development_fold,
)
from loss import (  # noqa: E402
    StateAwarePhysicsLoss,
    composite_validation_score,
    risk_metrics,
)
from model import Lite_STG_Mamba  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "CMAPSS"


class FoldAndDataTests(unittest.TestCase):
    def test_five_engine_folds_are_disjoint(self) -> None:
        folds = make_engine_folds(np.arange(1, 101), n_splits=5, seed=42)
        self.assertEqual(len(folds), 5)
        all_validation = []
        for train, validation in folds:
            self.assertEqual(len(train), 80)
            self.assertEqual(len(validation), 20)
            self.assertFalse(set(train) & set(validation))
            all_validation.extend(validation)
        self.assertEqual(sorted(all_validation), list(range(1, 101)))

    @unittest.skipUnless(DATA_DIR.is_dir(), "C-MAPSS data missing")
    def test_development_fold_uses_scaled_targets_and_train_only(self) -> None:
        bundle = prepare_development_fold(
            DATA_DIR, "FD001", fold_index=0, scaler_type="minmax",
            window_size=30, target_scale=125.0
        )
        self.assertEqual(bundle.train.windows.shape[1:], (30, 14))
        self.assertGreater(float(bundle.train.targets.max()), 0.9)
        self.assertLessEqual(float(bundle.train.targets.max()), 1.0)
        self.assertLessEqual(float(bundle.validation.targets.max()), 1.0)

    @unittest.skipUnless(DATA_DIR.is_dir(), "C-MAPSS data missing")
    def test_operating_condition_scaler_fd002(self) -> None:
        frame = load_train_only(DATA_DIR, "FD002")
        scaler = OperatingConditionScaler(n_clusters=6, random_state=42)
        values = scaler.fit_transform(frame)
        self.assertEqual(values.shape, (len(frame), 14))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(len(np.unique(scaler.kmeans.labels_)), 6)

    def test_blind_test_files_unreadable(self) -> None:
        if not DATA_DIR.is_dir():
            self.skipTest("C-MAPSS data missing")
        for name in ("test_FD001.txt", "RUL_FD001.txt"):
            self.assertFalse(os.access(DATA_DIR / name, os.R_OK))


class LossTests(unittest.TestCase):
    def test_state_aware_late_penalty_increases_near_failure(self) -> None:
        criterion = StateAwarePhysicsLoss(
            target_scale=125.0, gamma=2.0, eta=2.0, tau=40.0,
            base_weight=0.0, asymmetric_weight=1.0
        )
        near_true = torch.tensor([10.0 / 125.0])
        near_late = torch.tensor([20.0 / 125.0])
        healthy_true = torch.tensor([100.0 / 125.0])
        healthy_late = torch.tensor([110.0 / 125.0])
        near_loss = criterion(near_late, near_true)
        healthy_loss = criterion(healthy_late, healthy_true)
        self.assertGreater(float(near_loss), float(healthy_loss))

    def test_composite_and_risk_metrics(self) -> None:
        true = np.asarray([10.0, 50.0, 100.0])
        pred = np.asarray([5.0, 60.0, 90.0])
        metrics = risk_metrics(true, pred)
        self.assertEqual(metrics["rmse"], float(np.sqrt((25 + 100 + 100) / 3)))
        self.assertEqual(metrics["late_ratio"], 1.0 / 3.0)
        self.assertGreater(
            composite_validation_score(metrics["rmse"], metrics["nasa_score"]),
            metrics["rmse"],
        )


class ModelShapeTests(unittest.TestCase):
    def test_lite_shape_and_diagnostics(self) -> None:
        model = Lite_STG_Mamba(d_model=32, window_size=30, state_dim=8, kernels=(3, 5), dropout=0.1)
        model.eval()
        x = torch.randn(4, 30, 14)
        with torch.no_grad():
            prediction, diagnostics = model(x, return_diagnostics=True)
        self.assertEqual(tuple(prediction.shape), (4, 1))
        self.assertEqual(tuple(diagnostics["adjacency"].shape), (14, 14))
        self.assertEqual(tuple(diagnostics["channel_gate"].shape), (4, 30, 32))

    def test_all_baseline_shapes(self) -> None:
        x = torch.randn(3, 30, 14)
        for name in BASELINE_REGISTRY:
            with self.subTest(name=name):
                model = build_baseline(name, d_model=32, dropout=0.1)
                model.eval()
                with torch.no_grad():
                    output = model(x)
                self.assertEqual(tuple(output.shape), (3, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
