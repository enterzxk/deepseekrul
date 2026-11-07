"""Tests for engine bootstrap and stratified coverage (numpy only)."""

import unittest

import numpy as np

from src.punet.diagnostics import engine_bootstrap_ci, stratified_coverage


class TestBootstrap(unittest.TestCase):
    def test_ci_contains_truth(self):
        rng = np.random.default_rng(0)
        units = np.repeat(np.arange(50), 10)
        covered = (rng.random(500) < 0.9).astype(float)
        lo, hi = engine_bootstrap_ci(units, covered, n_boot=500, seed=0)
        truth = covered.mean()
        self.assertLessEqual(lo, truth + 1e-9)
        self.assertGreaterEqual(hi, truth - 1e-9)
        self.assertLess(lo, hi)


class TestStratified(unittest.TestCase):
    def test_bins_are_reported(self):
        rng = np.random.default_rng(1)
        n = 200
        units = np.repeat(np.arange(20), 10)
        cutoffs = rng.integers(1, 50, n).astype(float)
        mu = rng.uniform(60, 120, n)
        y = rng.uniform(0, 125, n)
        lo = y - rng.uniform(0, 20, n)
        hi = y + rng.uniform(0, 20, n)
        out = stratified_coverage(units, cutoffs, mu, y, lo, hi, n_bins=4)
        self.assertGreater(len(out["stage_bins"]), 0)
        self.assertGreater(len(out["misfit_bins"]), 0)
        for row in out["stage_bins"]:
            self.assertGreaterEqual(row["coverage"], 0.0)
            self.assertLessEqual(row["coverage"], 1.0)
