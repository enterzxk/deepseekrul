"""Probe B5: can the condition map be interpolated in u to an unseen setpoint?

Feasibility gate for CAI.  The mechanism promises that a(u), b(u) are smooth in
the three settings, so a setpoint absent from training can be reached by
interpolation.  If that is false, CAI has nothing the baselines lack, and the
line stops before any training run.

Protocol, for each held-out condition c*:

1. fit the per-channel (a,b) maps on the other five, against the pooled curve of
   those five;
2. regress each map coefficient on the settings (linear and quadratic), predict
   c*'s map;
3. compare the *rendered trajectory*, not the coefficients: push real rows of
   every seen condition through the predicted map and through the true map, and
   take the RMS difference.  Both renderings target the same space, so no gauge
   argument is needed.

The reference numbers are the intrinsic re-rendering error from B4 (0.024-0.031
standardised units), which is the floor any map can reach.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data as D
import probe as B1
import probe3 as B3
import probe4 as B4

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "bline"


def design(u: np.ndarray, order: int) -> np.ndarray:
    cols = [np.ones(len(u)), u[:, 0], u[:, 1], u[:, 2]]
    if order >= 2:
        cols += [u[:, 0] ** 2, u[:, 1] ** 2, u[:, 2] ** 2,
                 u[:, 0] * u[:, 1], u[:, 0] * u[:, 2], u[:, 1] * u[:, 2]]
    return np.column_stack(cols)


def pooled(xs, tr, lab, conds, grid):
    parts, weights = [], []
    for c in conds:
        m, s = B1.condition_curve(xs, tr.rul, lab, tr.unit, tr.engines(), c, grid)
        parts.append(m)
        weights.append(s.astype(float))
    parts, weights = np.stack(parts), np.stack(weights)
    return np.nansum(parts * weights[:, :, None], axis=0) / np.maximum(weights.sum(0), 1.0)[:, None]


def maps_against(xs, tr, lab, conds, base, grid):
    ok = np.isfinite(base).all(1)
    out = {}
    for c in conds:
        Cc, _ = B1.condition_curve(xs, tr.rul, lab, tr.unit, tr.engines(), c, grid)
        k = ok & np.isfinite(Cc).all(1)
        out[c] = B1.fit_diag(Cc[k].T, base[k].T)
    return out


def render(xs, tr, lab, src, a_s, b_s, a_d, b_d, grid):
    m = lab == src
    z = a_d + b_d * ((xs[m] - a_s) / b_s)
    r_src = tr.rul[m]
    out = np.full((len(grid), xs.shape[1]), np.nan)
    for i, rr in enumerate(grid):
        k = np.abs(r_src - rr) < 1.0
        if k.sum() >= B4.MIN_ROWS:
            out[i] = z[k].mean(0)
    return out


def main() -> dict:
    res = {}
    for sub in ("FD002", "FD004"):
        tr = D.load(sub, "train")
        sc = D.setting_scale([tr])
        cen = D.fit_conditions([tr], k=6)
        lab = D.assign_conditions(tr, cen, sc)
        mu, sd = tr.x.mean(0), np.where(tr.x.std(0) > 1e-9, tr.x.std(0), 1.0)
        xs = (tr.x - mu) / sd
        grid = B1.GRID
        real = B4.curves_by_condition(xs, tr.rul, lab, grid)

        rows = []
        for hold in range(6):
            keep = [c for c in range(6) if c != hold]
            base5 = pooled(xs, tr, lab, keep, grid)
            maps5 = maps_against(xs, tr, lab, keep, base5, grid)
            # truth: map of c* against the same base5, fitted on its own rows
            Cc, _ = B1.condition_curve(xs, tr.rul, lab, tr.unit, tr.engines(), hold, grid)
            ok = np.isfinite(base5).all(1) & np.isfinite(Cc).all(1)
            a_true, b_true = B1.fit_diag(Cc[ok].T, base5[ok].T)

            U = cen[keep] / sc
            U_hold = (cen[hold] / sc)[None, :]
            row = {"hold": hold}
            for order in (1, 2):
                Xd, Xh = design(U, order), design(U_hold, order)
                pred = {}
                for idx, name in ((0, "a"), (1, "b")):
                    Y = np.stack([maps5[c][idx] for c in keep])
                    coef, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
                    pred[name] = (Xh @ coef)[0]
                errs_pred, errs_true = [], []
                for src in keep:
                    a_s, b_s = maps5[src]
                    r_pred = render(xs, tr, lab, src, a_s, b_s, pred["a"], pred["b"], grid)
                    r_true = render(xs, tr, lab, src, a_s, b_s, a_true, b_true, grid)
                    good = (np.isfinite(r_pred).all(1) & np.isfinite(r_true).all(1)
                            & np.isfinite(real[hold]).all(1))
                    if good.sum() < 10:
                        continue
                    errs_pred.append(float(np.sqrt(np.mean((r_pred[good] - r_true[good]) ** 2))))
                    errs_true.append(float(np.sqrt(np.mean((r_true[good] - real[hold][good]) ** 2))))
                row[f"order{order}"] = {
                    "map_err": float(np.median(errs_pred)) if errs_pred else None,
                    "truth_gap": float(np.median(errs_true)) if errs_true else None,
                    "gain_rel": float(np.sqrt(np.mean((np.log(np.abs(pred["b"]) + 1e-6)
                                                       - np.log(np.abs(b_true) + 1e-6)) ** 2))),
                }
            rows.append(row)
        res[sub] = rows
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    r = main()
    for sub, rows in r.items():
        print(f"\n=== {sub}")
        print(f"  {'hold':>4s} {'lin map err':>11s} {'quad map err':>12s} {'truth gap':>10s} {'log-gain rel':>12s}")
        for row in rows:
            o1, o2 = row["order1"], row["order2"]
            print(f"  {row['hold']:>4d} {o1['map_err']:>11.4f} {o2['map_err']:>12.4f} "
                  f"{o1['truth_gap']:>10.4f} {o1['gain_rel']:>12.3f}")
        print("  (map err = predicted map vs true map, both rendered; "
              "truth gap = true map vs real target rows)")
    (OUT / "probe_b5.json").write_text(json.dumps(r, indent=1))
    print(f"\nwritten {OUT / 'probe_b5.json'}")
