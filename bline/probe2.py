"""Probe B2: is the condition map universal, or does each engine have its own?

Probe B1 established that on FD002/FD004 the condition effect on the mean
degradation curve is a per-channel affine map, exactly, and that a time-warp
reading is wrong by two orders of magnitude.  That is a population statement:
the curves were averaged over engines.  The question that decides whether any
method is needed at all is whether the same affine map applies to *every*
engine.

Two nested models per engine e, fitted on its own rows only:

  separable    x_{e,c}(r) = a_e + b_e * H_c(r)          one gain per engine
  interaction  x_{e,c}(r) = a_{e,c} + b_{e,c} * H_c(r)  one gain per cell

where H_c is the condition-aligned population curve from B1.  If ``separable``
already reaches the noise floor, the condition effect is a rank-one-in-engines
nuisance that can be inverted exactly, and there is no representation-learning
problem left on this benchmark.  If it does not, the residual is the real
problem, and its size says how much room a method has.

The noise floor per engine comes from splitting that engine's rows of a cell
into two interleaved halves, so the comparison is again held-out rather than
in-sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data as D
import probe as B1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "bline"
MIN_ROWS = 12


def engine_cell_curve(xs, rul, lab, unit, eng, cond, grid):
    m = (unit == eng) & (lab == cond)
    d = xs.shape[1]
    if m.sum() < 4:
        return None, None
    r, x = rul[m], xs[m]
    order = np.argsort(r)
    r, x = r[order], x[order]
    # two interleaved halves in cycle order, each interpolated on the grid
    out = []
    for sel in (np.arange(len(r)) % 2 == 0, np.arange(len(r)) % 2 == 1):
        o = np.full((len(grid), d), np.nan)
        ok = (grid >= r[sel].min()) & (grid <= r[sel].max())
        if ok.sum() < 5:
            return None, None
        o[ok] = np.stack([np.interp(grid[ok], r[sel], x[sel, j]) for j in range(d)], axis=1)
        out.append(o)
    return out[0], out[1]


def fit_shared(curves: list[np.ndarray], bases: list[np.ndarray]):
    """One (a,b) pair shared across cells: stack all anchors of all cells."""
    ys = np.concatenate([c for c in curves], axis=1)
    xs_ = np.concatenate([b for b in bases], axis=1)
    return B1.fit_diag(ys, xs_)


def main() -> dict:
    out = {}
    for sub in ("FD002", "FD004"):
        tr = D.load(sub, "train")
        sc = D.setting_scale([tr])
        cen = D.fit_conditions([tr], k=6)
        lab = D.assign_conditions(tr, cen, sc)
        xs, mu, sd = B1.standardise(tr, tr)
        engines = tr.engines()
        grid = B1.GRID

        # population condition maps, fitted on everything (this is the H_c basis)
        base = B1.pooled_curve(xs, tr.rul, lab, tr.unit, engines, grid)
        grid_ok = np.isfinite(base).all(1)
        H = {}
        for c in range(6):
            Cc, _ = B1.condition_curve(xs, tr.rul, lab, tr.unit, engines, c, grid)
            ok = np.isfinite(Cc).all(1) & grid_ok
            a, b = B1.fit_diag(Cc[ok].T, base[ok].T)
            H[c] = B1.apply_diag(base.T, a, b)          # aligned population curve, full grid

        rows = []
        gains = []
        for e in engines:
            cells_A, cells_B, bases = [], [], []
            for c in range(6):
                ca, cb = engine_cell_curve(xs, tr.rul, lab, tr.unit, e, c, grid)
                if ca is None:
                    continue
                keep = grid_ok & np.isfinite(ca[:, 0]) & np.isfinite(cb[:, 0])
                if keep.sum() < 8:
                    continue
                cells_A.append(ca[keep].T)
                cells_B.append(cb[keep].T)
                bases.append(H[c][:, keep])
            if len(cells_A) < 4:
                continue
            noise = float(np.mean([np.mean((A - B) ** 2) / 2 for A, B in zip(cells_A, cells_B)])) + 1e-12
            ref = float(np.mean([np.mean((b - B) ** 2) for b, B in zip(bases, cells_B)])) + 1e-12

            a, b = fit_shared(cells_A, bases)
            sep = [B1.apply_diag(bb, a, b) for bb in bases]
            r_sep = float(np.mean([np.mean((p - B) ** 2) for p, B in zip(sep, cells_B)]))

            inter = [B1.apply_diag(bb, *B1.fit_diag(AA, bb)) for AA, bb in zip(cells_A, bases)]
            r_int = float(np.mean([np.mean((p - B) ** 2) for p, B in zip(inter, cells_B)]))

            # cross-cell transfer: fit (a,b) on one cell, apply to the others
            r_xfer = []
            for k in range(len(cells_A)):
                ak, bk = B1.fit_diag(cells_A[k], bases[k])
                r_xfer.append(np.mean([np.mean((B1.apply_diag(bases[j], ak, bk) - cells_B[j]) ** 2)
                                       for j in range(len(cells_B)) if j != k]))
            rows.append({"engine": int(e), "cells": len(cells_A),
                         "sep_explained": 1 - r_sep / ref,
                         "inter_explained": 1 - r_int / ref,
                         "xfer_explained": 1 - float(np.mean(r_xfer)) / ref,
                         "sep_z": float(np.sqrt(r_sep / noise)),
                         "inter_z": float(np.sqrt(r_int / noise)),
                         "xfer_z": float(np.sqrt(np.mean(r_xfer) / noise))})
            gains.append(b)

        gains = np.stack(gains)
        q = np.percentile(gains, [5, 50, 95], axis=0)
        out[sub] = {
            "n_engines": len(rows),
            "sep_explained_median": float(np.median([r["sep_explained"] for r in rows])),
            "inter_explained_median": float(np.median([r["inter_explained"] for r in rows])),
            "xfer_explained_median": float(np.median([r["xfer_explained"] for r in rows])),
            "sep_z_median": float(np.median([r["sep_z"] for r in rows])),
            "inter_z_median": float(np.median([r["inter_z"] for r in rows])),
            "xfer_z_median": float(np.median([r["xfer_z"] for r in rows])),
            "gain_p5": q[0].round(3).tolist(), "gain_median": q[1].round(3).tolist(),
            "gain_p95": q[2].round(3).round(3).tolist(),
            "gain_cv": (gains.std(0) / np.maximum(np.abs(gains.mean(0)), 1e-9)).round(3).tolist(),
        }
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = main()
    for sub, r in res.items():
        print(f"\n=== {sub}  engines with >=4 usable cells: {r['n_engines']}")
        print(f"  explained  separable {r['sep_explained_median']:.3f}   "
              f"interaction {r['inter_explained_median']:.3f}   "
              f"fit-one-cell-apply-others {r['xfer_explained_median']:.3f}")
        print(f"  z          separable {r['sep_z_median']:.1f}   "
              f"interaction {r['inter_z_median']:.1f}   "
              f"cross-cell {r['xfer_z_median']:.1f}")
        print(f"  engine gain CV across engines: {r['gain_cv']}")
    (OUT / "probe_b2.json").write_text(json.dumps(res, indent=1))
    print(f"\nwritten {OUT / 'probe_b2.json'}")
