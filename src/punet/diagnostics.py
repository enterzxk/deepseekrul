"""Engine-level coverage diagnostics (RQ3)."""

from __future__ import annotations

import numpy as np


def engine_bootstrap_ci(units, covered, n_boot: int = 5000, seed: int = 42, alpha: float = 0.05):
    """Cluster bootstrap CI for mean coverage, resampling whole engines.

    Mirrors the engine-cluster bootstrap used by docs/FINAL_PROTOCOL.md so the
    numbers are comparable with Gate 0.
    """
    units = np.asarray(units).reshape(-1)
    covered = np.asarray(covered, dtype=np.float64).reshape(-1)
    if units.size != covered.size:
        raise ValueError("units and covered must share a length")

    uniq, inverse = np.unique(units, return_inverse=True)
    sums = np.bincount(inverse, weights=covered, minlength=uniq.size)
    counts = np.bincount(inverse, minlength=uniq.size).astype(np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, uniq.size, size=(n_boot, uniq.size))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def stratified_coverage(units, cutoffs, mu, y, lo, hi, n_bins: int = 5):
    """Coverage per degradation-stage bin and per prior-misfit bin.

    Returns a dict with two lists of dicts: stage_bins and misfit_bins.
    """
    units = np.asarray(units).reshape(-1)
    cutoffs = np.asarray(cutoffs, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)

    covered = ((y >= lo) & (y <= hi)).astype(np.float64)
    tau = cutoffs / np.where(mu > 0, mu, 1.0)
    misfit = np.abs(mu - y)

    def bin_report(values, label):
        edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        out = []
        for i in range(len(edges) - 1):
            mask = (values >= edges[i]) & (values <= edges[i + 1] if i == len(edges) - 2 else values < edges[i + 1])
            if not np.any(mask):
                continue
            out.append(
                {
                    label: [float(edges[i]), float(edges[i + 1])],
                    "n": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                }
            )
        return out

    return {
        "stage_bins": bin_report(tau, "tau"),
        "misfit_bins": bin_report(misfit, "abs_mu_minus_y"),
        "n_samples": int(covered.size),
        "n_engines": int(np.unique(units).size),
    }
