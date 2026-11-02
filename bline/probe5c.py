"""Probe B5c: which interpolation family should CAI use, chosen without peeking.

B5b showed the linear-in-settings interpolation of (a,b) leaves a large error on
the invariant signal, but it measured that with a plain division, which explodes
on low-gain channels.  Redone here with the ridge inverse actually used by the
model, and with the choice of family made by *nested* leave-one-condition-out:

  outer loop  hold out c*; fit maps on the other five
  inner loop  for each of those five, predict it from the remaining four and
              score every family; the winner is applied to c*

so the family is chosen on the training conditions only.  Families compared:

  linear    (a,b) linear in the three settings                     (4 coefficients)
  quad      quadratic in the settings                              (10)
  nearest   copy the map of the closest seen setpoint
  knn2      inverse-distance blend of the two closest setpoints
  global    one map for all setpoints (no condition index at all)

The score is the error of the *aligned signal* on real rows of the held-out
setpoint, which is what the temporal model consumes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data as D
import probe as B1
import torchdata as T

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "bline"


def design(u, order):
    cols = [np.ones(len(u)), u[:, 0], u[:, 1], u[:, 2]]
    if order >= 2:
        cols += [u[:, 0] ** 2, u[:, 1] ** 2, u[:, 2] ** 2,
                 u[:, 0] * u[:, 1], u[:, 0] * u[:, 2], u[:, 1] * u[:, 2]]
    return np.column_stack(cols)


def maps_for(z, tr, lab, conds, grid=B1.GRID):
    maps, _ = T.fit_maps(z, tr, lab, conds, grid)
    return maps


def predict(family, maps, seen, centres, scale, target_u):
    U = centres[seen] / scale
    ut = target_u / scale
    if family == "global":
        return (np.mean([maps[c][0] for c in seen], 0),
                np.mean([maps[c][1] for c in seen], 0))
    if family in ("nearest", "knn2"):
        dist = np.linalg.norm(U - ut[None, :], axis=1)
        order = np.argsort(dist)
        if family == "nearest":
            c = seen[order[0]]
            return maps[c]
        k = order[:2]
        w = 1.0 / np.maximum(dist[k], 1e-6)
        w = w / w.sum()
        a = sum(wi * maps[seen[j]][0] for wi, j in zip(w, k))
        b = sum(wi * maps[seen[j]][1] for wi, j in zip(w, k))
        return a, b
    order_ = 1 if family == "linear" else 2
    X = design(U, order_)
    xt = design(ut[None, :], order_)
    A = np.stack([maps[c][0] for c in seen])
    B = np.stack([maps[c][1] for c in seen])
    ca, *_ = np.linalg.lstsq(X, A, rcond=None)
    cb, *_ = np.linalg.lstsq(X, B, rcond=None)
    return (xt @ ca)[0], (xt @ cb)[0]


def aligned_err(z_rows, a_p, b_p, a_t, b_t, ridge=0.05):
    sp = (z_rows - a_p) * b_p / (b_p * b_p + ridge)
    st = (z_rows - a_t) * b_t / (b_t * b_t + ridge)
    return float(np.sqrt(np.mean((sp - st) ** 2)))


def main() -> dict:
    families = ("linear", "quad", "nearest", "knn2", "global")
    res = {}
    for sub in ("FD002", "FD004"):
        tr = D.load(sub, "train")
        sc = D.setting_scale([tr])
        cen = D.fit_conditions([tr], k=6)
        lab = D.assign_conditions(tr, cen, sc)
        mu, sd = tr.x.mean(0), np.where(tr.x.std(0) > 1e-9, tr.x.std(0), 1.0)
        z = (tr.x - mu) / sd
        rows = []
        for hold in range(6):
            seen5 = [c for c in range(6) if c != hold]
            maps5 = maps_for(z, tr, lab, seen5)
            # truth: the held-out map fitted against the same five-condition pooled curve
            base5 = T.pooled_curve(z, tr, lab, seen5)
            ok = np.isfinite(base5).all(1)
            Cc, _ = B1.condition_curve(z, tr.rul, lab, tr.unit, tr.engines(), hold, B1.GRID)
            k = ok & np.isfinite(Cc).all(1)
            a_t, b_t = B1.fit_diag(Cc[k].T, base5[k].T)

            # inner selection over the five seen conditions
            inner = {f: [] for f in families}
            for c_in in seen5:
                rest = [c for c in seen5 if c != c_in]
                maps4 = maps_for(z, tr, lab, rest)
                base4 = T.pooled_curve(z, tr, lab, rest)
                ok4 = np.isfinite(base4).all(1)
                C4, _ = B1.condition_curve(z, tr.rul, lab, tr.unit, tr.engines(), c_in, B1.GRID)
                k4 = ok4 & np.isfinite(C4).all(1)
                a_t4, b_t4 = B1.fit_diag(C4[k4].T, base4[k4].T)
                rows_in = z[lab == c_in]
                for f in families:
                    a_p, b_p = predict(f, maps4, rest, cen, sc, cen[c_in])
                    inner[f].append(aligned_err(rows_in, a_p, b_p, a_t4, b_t4))
            pick = min(families, key=lambda f: np.mean(inner[f]))
            rows_hold = z[lab == hold]
            outer = {f: aligned_err(rows_hold, *predict(f, maps5, seen5, cen, sc, cen[hold]),
                                    a_t, b_t) for f in families}
            rows.append({"hold": hold, "picked": pick,
                         "inner": {f: round(float(np.mean(v)), 3) for f, v in inner.items()},
                         "outer": {f: round(float(v), 3) for f, v in outer.items()}})
        res[sub] = rows
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    r = main()
    for sub, rows in r.items():
        print(f"\n=== {sub}")
        print(f"  {'hold':>4s} {'picked':>8s} {'inner best':>10s}   outer error by family")
        fams = ("linear", "quad", "nearest", "knn2", "global")
        print("        " + " ".join(f"{f:>8s}" for f in fams))
        for q in rows:
            print(f"  {q['hold']:>4d} {q['picked']:>8s} {q['inner'][q['picked']]:>10.3f}   "
                  + " ".join(f"{q['outer'][f]:>8.3f}" for f in fams))
        agg = {f: float(np.mean([q["outer"][f] for q in rows])) for f in fams}
        print("  mean outer error:", {k: round(v, 3) for k, v in agg.items()})
        print("  picked:", [q["picked"] for q in rows])
    (OUT / "probe_b5c.json").write_text(json.dumps(r, indent=1))
    print(f"\nwritten {OUT / 'probe_b5c.json'}")
