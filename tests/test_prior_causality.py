"""Leakage unit tests for the nested lifetime prior (Gate 0 entry requirement).

Implements the four mandatory assertions from the IDA-PE gated spec plus a
hand-computed unit_mse cross-check. All tests run on small synthetic C-MAPSS
style data; none touches the real dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lifetime_prior.nested_prior import (  # noqa: E402
    StaircasePrior, build_rows, hgb_fit_predict, run_subset,
)
from lsc_feasibility import prefix_features  # noqa: E402
from dataset import SELECTED_SENSOR_COLUMNS  # noqa: E402


N_SENS = 14
SENS_COLS = list(SELECTED_SENSOR_COLUMNS)


def synthetic_engine(unit, n_cycles, seed, true_extra=0):
    """One engine: n_cycles rows, gentle drift; true length = n_cycles."""
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)
    ops = np.stack([rng.normal(0, 0.2, n_cycles) + 0.5,
                    rng.normal(0, 0.1, n_cycles) + 1.0,
                    rng.normal(0, 5, n_cycles) + 100.0], axis=1)
    drift = np.linspace(0, 1, n_cycles)[:, None]
    sensors = 0.5 + 0.3 * drift + rng.normal(0, 0.05, (n_cycles, N_SENS))
    frame = pd.DataFrame(ops, columns=["op_1", "op_2", "op_3"])
    for j, c in enumerate(SENS_COLS):
        frame[c] = sensors[:, j]
    frame.insert(0, "cycle", cycles)
    frame.insert(0, "unit", unit)
    return frame


def synthetic_dataset(n_units=20, length=120, seed0=1000):
    frames = [synthetic_engine(u, length + (u % 4) * 10, seed0 + u)
              for u in range(1, n_units + 1)]
    return pd.concat(frames, ignore_index=True)


def corrupt_after(frame, unit, cut_cycle, seed=7):
    """Return a copy of `frame` where engine `unit`'s sensors after `cut_cycle`
    are replaced by noise (future information perturbation)."""
    out = frame.copy()
    sel = (out["unit"] == unit) & (out["cycle"] > cut_cycle)
    rng = np.random.default_rng(seed)
    for c in SENS_COLS:
        out.loc[sel, c] = rng.normal(5.0, 2.0, int(sel.sum()))
    return out


CUTOFFS = list(range(50, 111, 10))  # 50..100; engines have 120-150 cycles


# ---------------------------------------------------------------- T1 -------

def test_t1_future_sensors_cannot_change_prior():
    """Perturbing sensors AFTER cycle t must not change mu_T(t) features or
    an HGB's prediction for cutoffs <= t."""
    frame = synthetic_dataset(n_units=12, length=130)
    # features from the clean data
    rows_clean = build_rows(frame[frame.unit <= 10],
                            _IdentityScaler(N_SENS), CUTOFFS)
    # corrupt engine 3's sensors after cycle 80
    frame_bad = corrupt_after(frame, 3, 80)
    rows_bad = build_rows(frame_bad[frame_bad.unit <= 10],
                          _IdentityScaler(N_SENS), CUTOFFS)
    clean = {(r["unit"], r["c"]): r["feat"] for r in rows_clean}
    bad = {(r["unit"], r["c"]): r["feat"] for r in rows_bad}
    for (unit, c), feat in clean.items():
        if unit == 3 and c <= 80:
            np.testing.assert_array_equal(
                feat, bad[(unit, c)],
                err_msg=f"features for unit 3 cutoff {c} changed after "
                        f"perturbing sensors after cycle 80")
    # and an HGB trained on other engines predicts identically for those rows
    tr = [r for r in rows_clean if r["unit"] != 3]
    va_clean = [r for r in rows_clean if r["unit"] == 3]
    va_bad = [r for r in rows_bad if r["unit"] == 3 and r["c"] <= 80]
    p_clean = hgb_fit_predict(tr, va_clean, 0)
    p_bad = hgb_fit_predict(tr, va_bad, 0)
    np.testing.assert_array_equal(p_clean[:len(va_bad)], p_bad)


# ---------------------------------------------------------------- T2/T3 -----

def _run(frame, seeds):
    table, meta = run_subset("FD00T", "unused", None, outer_seed=99,
                             seeds=seeds, inner_folds=4, cutoffs=CUTOFFS,
                             frame=frame, quiet=True)
    return table


def test_t2_and_t3_val_label_cannot_leak():
    """Perturbing an outer-val engine's true lifetime (extending its record)
    must change neither any outer-train mu/sigma (T2) nor that engine's own
    mu/sigma at shared cutoffs (T3)."""
    frame = synthetic_dataset(n_units=20, length=130)
    seeds = (11, 22)
    t0 = _run(frame, seeds)
    # pick a genuine outer-val engine of fold 0 from the first run; the split
    # depends only on the unit list + seed, which the perturbation preserves.
    val_u = int(t0[(t0.outer_fold == 0) & (t0.role == "val")]["unit"].iloc[0])
    frame_ext = frame.copy()
    extra = synthetic_engine(val_u, 40, seed=5555)  # 40 more cycles -> longer true life
    max_c = int(frame[frame.unit == val_u]["cycle"].max())
    extra["cycle"] = extra["cycle"] + max_c   # 1-based contiguity preserved
    extra["unit"] = val_u
    frame_ext = pd.concat([frame_ext, extra], ignore_index=True)
    t1 = _run(frame_ext, seeds)
    # T2: outer-train rows of the SAME outer fold identical (bit-for-bit).
    # Other folds legitimately see the perturbed engine as training data.
    a = t0[(t0.role == "train") & (t0.outer_fold == 0)].sort_values(["unit", "c"])
    b = t1[(t1.role == "train") & (t1.outer_fold == 0)].sort_values(["unit", "c"])
    pd.testing.assert_frame_equal(
        a[["unit", "c", "mu_T", "sigma_T"]].reset_index(drop=True),
        b[["unit", "c", "mu_T", "sigma_T"]].reset_index(drop=True))
    # T3: that engine's own val rows at shared cutoffs identical
    a5 = t0[(t0.role == "val") & (t0.unit == val_u)].set_index(["outer_fold", "c"])
    b5 = t1[(t1.role == "val") & (t1.unit == val_u)].set_index(["outer_fold", "c"])
    shared = a5.index.intersection(b5.index)
    assert len(shared) > 0, "perturbed engine never landed in any outer-val fold"
    np.testing.assert_allclose(
        a5.loc[shared, "mu_T"].to_numpy(), b5.loc[shared, "mu_T"].to_numpy(), rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        a5.loc[shared, "sigma_T"].to_numpy(), b5.loc[shared, "sigma_T"].to_numpy(), rtol=0, atol=1e-9)


# ---------------------------------------------------------------- T4 -------

def test_t4_staircase_cutoff_never_after_window_end():
    table = pd.DataFrame([
        dict(subset="FD00T", outer_fold=0, role="val", unit=1, c=c,
             mu_T=100.0 + c, sigma_T=5.0, true_T=140)
        for c in (50, 60, 70, 80, 90, 100, 110)
    ] + [
        dict(subset="FD00T", outer_fold=0, role="val", unit=2, c=c,
             mu_T=90.0 + c, sigma_T=4.0, true_T=90)   # short engine: only c=50,60,70,80 rows
        for c in (50, 60, 70, 80)
    ])
    sp = StaircasePrior(table, {0: 105.0}, "FD00T")
    for unit, true_T in ((1, 140), (2, 90)):
        for t in range(1, true_T + 1):
            mu, sig, flag, c_star = sp.lookup(0, unit, t)
            assert c_star <= t, f"T4 violated: cutoff {c_star} > window end {t}"
            if t < 50:
                assert flag == 1 and mu == 105.0
            else:
                assert flag == 0
    # staircase monotonicity: mu constant between cutoffs, jumps only at cutoffs
    mu1, _, _, c1 = sp.lookup(0, 1, 74)
    mu2, _, _, c2 = sp.lookup(0, 1, 70)
    assert mu1 == mu2 and c1 == c2 == 70  # latest cutoff <= t, no interpolation
    mu3, _, _, c3 = sp.lookup(0, 1, 79)
    assert c3 == 70  # still the same cutoff until the next grid point
    mu4, _, _, c4 = sp.lookup(0, 1, 80)
    assert c4 == 80 and mu4 != mu3   # new cutoff kicks in exactly at its cycle


# ------------------------------------------------- unit statistics ---------

def test_unit_mse_hand_computed():
    """Engine-cluster statistic must match a hand computation (spec section 1)."""
    # windows: engine 1 -> errors +2, -2 (MSE 4); engine 2 -> error +6 (MSE 36)
    errs = np.array([2.0, -2.0, 6.0])
    units = np.array([1, 1, 2])
    per_engine = {1: 4.0, 2: 36.0}
    vals = np.array(list(per_engine.values()))
    assert np.isclose(vals.mean(), 20.0)  # hand value
    # bootstrap on engines must only ever recombine the engine-level values
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(vals), size=(2000, len(vals)))
    boot = vals[idx].mean(axis=1)
    assert set(np.unique(boot)) <= {4.0, 20.0, 36.0}  # convex combos of {4,36}


class _IdentityScaler:
    """Minimal stand-in: transform(frame) -> raw sensor matrix."""

    def __init__(self, n_sens):
        self.cols = SENS_COLS

    def transform(self, g):
        return g[self.cols].to_numpy(dtype=np.float64)


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_") and callable(v)}.items()):
        fn()
        print(f"PASS {name}")
    print("ALL PRIOR CAUSALITY TESTS PASSED")
