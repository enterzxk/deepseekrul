"""Regression tests for the C-MAPSS Step 1 preprocessing protocol."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import (  # noqa: E402
    SELECTED_SENSOR_COLUMNS,
    SensorMinMaxScaler,
    build_piecewise_linear_rul,
    compute_nasa_score,
    compute_rmse,
    load_cmapss_files,
    prepare_cmapss_datasets,
)

DATA_DIR = PROJECT_ROOT / "data" / "CMAPSS"

EXPECTED = {
    "FD001": {
        "train_windows": 17731,
        "test_windows": 100,
        "train_units": 100,
        "test_units": 100,
        "short_test": 0,
        "test_rul": (7.0, 125.0),
    },
    "FD002": {
        "train_windows": 46219,
        "test_windows": 259,
        "train_units": 260,
        "test_units": 259,
        "short_test": 6,
        "test_rul": (6.0, 125.0),
    },
    "FD003": {
        "train_windows": 21820,
        "test_windows": 100,
        "train_units": 100,
        "test_units": 100,
        "short_test": 0,
        "test_rul": (6.0, 125.0),
    },
    "FD004": {
        "train_windows": 54028,
        "test_windows": 248,
        "train_units": 249,
        "test_units": 248,
        "short_test": 11,
        "test_rul": (6.0, 125.0),
    },
}

TEST_FILES_READABLE = all(
    os.access(DATA_DIR / f"test_{subset}.txt", os.R_OK)
    and os.access(DATA_DIR / f"RUL_{subset}.txt", os.R_OK)
    for subset in EXPECTED
)


class MetricTests(unittest.TestCase):
    def test_rmse_known_value(self) -> None:
        value = compute_rmse([0.0, 10.0], [1.0, 14.0])
        self.assertAlmostEqual(value, float(np.sqrt(8.5)), places=12)

    def test_nasa_score_sign_convention(self) -> None:
        # d = pred - true = +1 and +4, both late predictions.
        value = compute_nasa_score([0.0, 10.0], [1.0, 14.0])
        expected = float(np.expm1(0.1) + np.expm1(0.4))
        self.assertAlmostEqual(value, expected, places=12)

    def test_identity_predictions_score_zero(self) -> None:
        y = np.asarray([1.0, 5.0, 125.0])
        self.assertEqual(compute_rmse(y, y), 0.0)
        self.assertEqual(compute_nasa_score(y, y), 0.0)

    def test_metric_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_rmse([1.0, 2.0], [1.0])


class LabelAndScalerTests(unittest.TestCase):
    def test_piecewise_label(self) -> None:
        frame = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2],
                "cycle": [1, 2, 200, 1, 50],
            }
        )
        labels = build_piecewise_linear_rul(frame, max_rul=125)
        np.testing.assert_allclose(labels, [125.0, 125.0, 0.0, 49.0, 0.0])

    def test_scaler_uses_training_statistics_only(self) -> None:
        scaler = SensorMinMaxScaler(["a", "b"])
        scaler.fit(np.asarray([[0.0, 10.0], [2.0, 10.0]]))
        transformed = scaler.transform(np.asarray([[4.0, 10.0]]))
        np.testing.assert_allclose(transformed, [[2.0, 0.0]])

    def test_zero_range_feature_is_mapped_to_zero(self) -> None:
        scaler = SensorMinMaxScaler(["constant"])
        scaler.fit(np.asarray([[3.0], [3.0]]))
        np.testing.assert_allclose(scaler.transform(np.asarray([[3.0]])), [[0.0]])


@unittest.skipUnless(
    DATA_DIR.is_dir() and TEST_FILES_READABLE,
    "Official test files are sealed (or data missing); unseal for dataset tests.",
)
class RealCmapssRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundles = {
            subset: prepare_cmapss_datasets(DATA_DIR, subset)
            for subset in EXPECTED
        }

    def test_expected_dataset_shapes_and_counts(self) -> None:
        for subset, expected in EXPECTED.items():
            with self.subTest(subset=subset):
                bundle = self.bundles[subset]
                self.assertEqual(
                    len(bundle.train), expected["train_windows"]
                )
                self.assertEqual(len(bundle.test), expected["test_windows"])
                self.assertEqual(
                    np.unique(bundle.train.unit_ids).size, expected["train_units"]
                )
                self.assertEqual(
                    np.unique(bundle.test.unit_ids).size, expected["test_units"]
                )
                self.assertEqual(bundle.train.windows.shape[1:], (30, 14))
                self.assertEqual(bundle.test.windows.shape[1:], (30, 14))

    def test_training_targets_are_aligned_with_window_end(self) -> None:
        for subset, bundle in self.bundles.items():
            with self.subTest(subset=subset):
                raw_train, _, _ = load_cmapss_files(DATA_DIR, subset)
                last_cycle = raw_train.groupby("unit")["cycle"].max()
                expected = np.minimum(
                    125.0,
                    last_cycle.loc[bundle.train.unit_ids].to_numpy()
                    - bundle.train.end_cycles,
                )
                np.testing.assert_allclose(bundle.train.targets, expected)
                self.assertEqual(float(bundle.train.targets.min()), 0.0)
                self.assertEqual(float(bundle.train.targets.max()), 125.0)

    def test_scaler_is_fitted_only_on_train_rows(self) -> None:
        for subset, bundle in self.bundles.items():
            with self.subTest(subset=subset):
                raw_train, _, _ = load_cmapss_files(DATA_DIR, subset)
                sensors = raw_train.loc[:, SELECTED_SENSOR_COLUMNS].to_numpy(
                    dtype=np.float64
                )
                np.testing.assert_allclose(bundle.scaler.data_min_, sensors.min(axis=0))
                np.testing.assert_allclose(bundle.scaler.data_max_, sensors.max(axis=0))

    def test_test_targets_and_short_sequence_padding(self) -> None:
        for subset, expected in EXPECTED.items():
            with self.subTest(subset=subset):
                bundle = self.bundles[subset]
                self.assertEqual(
                    float(bundle.test.targets.min()), expected["test_rul"][0]
                )
                self.assertEqual(
                    float(bundle.test.targets.max()), expected["test_rul"][1]
                )
                short = bundle.test.observed_lengths < 30
                self.assertEqual(int(short.sum()), expected["short_test"])
                if short.any():
                    idx = int(np.flatnonzero(short)[0])
                    pad_len = 30 - int(bundle.test.observed_lengths[idx])
                    first_real_frame = bundle.test.windows[idx, pad_len]
                    np.testing.assert_allclose(
                        bundle.test.windows[idx, :pad_len],
                        np.repeat(first_real_frame[None, :], pad_len, axis=0),
                    )

    def test_raw_test_targets_available_for_appendix(self) -> None:
        raw = prepare_cmapss_datasets(DATA_DIR, "FD001", clip_test_rul=False)
        self.assertEqual(float(raw.test.targets.max()), 145.0)

    def test_no_nonfinite_values(self) -> None:
        for subset, bundle in self.bundles.items():
            with self.subTest(subset=subset):
                self.assertTrue(np.isfinite(bundle.train.windows).all())
                self.assertTrue(np.isfinite(bundle.test.windows).all())
                self.assertTrue(np.isfinite(bundle.train.targets).all())
                self.assertTrue(np.isfinite(bundle.test.targets).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)