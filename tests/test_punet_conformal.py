"""Tests for the block split and the conformal calibrator (numpy only)."""

import unittest

import numpy as np

from src.punet.conformal import BlockConformalCalibrator, block_split
from src.punet.metrics import engine_coverage, picp, pinaw, winkler


class TestBlockSplit(unittest.TestCase):
    def test_disjoint_and_temporal(self):
        units = np.repeat(np.arange(1, 21), 5)
        cutoffs = np.tile(np.arange(5), 20)
        split = block_split(units, cutoffs, seed=0)
        blocks = [set(split.fit.tolist()), set(split.tune.tolist()), set(split.calib.tolist()), set(split.test.tolist())]
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                self.assertFalse(blocks[i] & blocks[j])

    def test_no_engine_leak(self):
        units = np.repeat(np.arange(1, 11), 4)
        cutoffs = np.tile(np.arange(4), 10)
        split = block_split(units, cutoffs, seed=1)
        fit_units = set(units[split.fit].tolist())
        calib_units = set(units[split.calib].tolist())
        self.assertFalse(fit_units & calib_units)


class TestCalibrator(unittest.TestCase):
    def test_coverage_on_gaussian(self):
        rng = np.random.default_rng(0)
        n = 4000
        y = rng.normal(size=n)
        y_hat = np.zeros(n)
        w = np.ones(n)
        scores = BlockConformalCalibrator.scores(y[:2000], y_hat[:2000], w[:2000])
        cal = BlockConformalCalibrator(delta=0.1).fit(scores)
        lo, hi = cal.interval(y_hat[2000:], w[2000:])
        cov = picp(y[2000:], lo, hi)
        self.assertGreater(cov, 0.87)
        self.assertLess(cov, 0.93)

    def test_scale_shrinks_width(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        cal = BlockConformalCalibrator(delta=0.1).fit(scores)
        lo1, hi1 = cal.interval(np.array([0.0]), np.array([1.0]))
        lo2, hi2 = cal.interval(np.array([0.0]), np.array([2.0]))
        self.assertAlmostEqual((hi1 - lo1)[0] * 2, (hi2 - lo2)[0])


class TestMetrics(unittest.TestCase):
    def test_basic_values(self):
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.5, 1.5, 2.5])
        hi = np.array([1.5, 2.5, 3.5])
        self.assertEqual(picp(y, lo, hi), 1.0)
        self.assertGreater(pinaw(y, lo, hi), 0)
        self.assertGreater(winkler(y, lo, hi), 0)

    def test_engine_coverage(self):
        units = np.array([1, 1, 2, 2])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lo = np.array([0.0, 0.0, 0.0, 0.0])
        hi = np.array([10.0, 10.0, 10.0, 0.5])
        out = engine_coverage(units, y, lo, hi)
        self.assertEqual(out["n_engines"], 2)
        self.assertAlmostEqual(out["min"], 0.5)
