"""Probe B5b: deployment error of an interpolated condition map.

B5 compared rendered curves in the raw standardised space, which is the wrong
yardstick: an error of 0.2 there can be either negligible or fatal depending on
the channel gain.  What the model actually consumes is the *aligned* signal
s = (x - a(u)) / b(u), so the number that decides CAI is

    err = RMS( (x - a_pred)/b_pred - (x - a_true)/b_true )   on real rows of the
          held-out condition,

i.e. the error of the invariant representation when the map for that condition
was never fitted.  Three ways to fill the gap are compared:

    interp     linear regression of (a,b) on the three settings, five points
    nearest    reuse the map of the closest seen setpoint
    pooled     one global map, no condition index at all

``interp`` is CAI's claim.  If it is not clearly better than ``nearest``, the
"smooth in u" premise is false and CAI has no mechanism the baselines lack.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data as D
import probe as B1
import probe3 as B3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "bline"


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
        a, b = B1.fit_diag(Cc[k].T, base[k].T)
        out[c] = (a, b)
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
        Uall = cen / sc

        rows = []
        for hold in range(6):
            keep = [c for c in range(6) if c != hold]
            base5 = pooled(xs, tr, lab, keep, grid)
            maps5 = maps_against(xs, tr, lab, keep, base5, grid)
            # truth: the map fitted on the held-out condition itself, same base
            Cc, _ = B1.condition_curve(xs, tr.rul, lab, tr.unit, tr.engines(), hold, grid)
            ok = np.isfinite(base5).all(1) & np.isfinite(Cc).all(1)
            a_true, b_true = B1.fit_diag(Cc[ok].T, base5[ok].T)

            # interpolate (a,b) in the settings
            Xd = np.column_stack([np.ones(len(keep)), Uall[keep]])
            Xh = np.column_stack([np.ones(1), Uall[hold][None, :]])
            a_int = (Xh @ np.linalg.lstsq(Xd, np.stack([maps5[c][0] for c in keep]), rcond=None)[0])[0]
            b_int = (Xh @ np.linalg.lstsq(Xd, np.stack([maps5[c][1] for c in keep]), rcond=None)[0])[0]
            # nearest seen setpoint in settings space
            dist = np.linalg.norm(Uall[keep] - Uall[hold][None, :], axis=1)
            a_near, b_near = maps5[keep[int(np.argmin(dist))]]

            m = lab == hold
            z = xs[m]
            ref = (z - a_true) / b_true           # the target invariant representation
            out = {}
            for name, (a_p, b_p) in (("interp", (a_int, b_int)),
                                     ("nearest", (a_near, b_near)),
                                     ("pooled", (np.zeros_like(a_true), np.ones_like(b_true)))):
                s = (z - a_p) / b_p
                out[name] = {"err": float(np.sqrt(np.mean((s - ref) ** 2))),
                             "corr": float(np.corrcoef(s.ravel(), ref.ravel())[0, 1])}
            # reference scale: spread of the invariant signal itself
            out["ref_rms"] = float(np.sqrt(np.mean(ref ** 2)))
            out["hold"] = hold
            out["n_rows"] = int(m.sum())
            rows.append(out)
        res[sub] = rows
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    r = main()
    for sub, rows in r.items():
        print(f"\n=== {sub}   (err on the invariant signal; ref_rms is its own scale)")
        print(f"  {'hold':>4s} {'rows':>6s} {'interp':>8s} {'nearest':>8s} {'pooled':>8s} {'ref_rms':>8s} {'corr_int':>9s}")
        for q in rows:
            print(f"  {q['hold']:>4d} {q['n_rows']:>6d} {q['interp']['err']:>8.3f} "
                  f"{q['nearest']['err']:>8.3f} {q['pooled']['err']:>8.3f} {q['ref_rms']:>8.3f} "
                  f"{q['interp']['corr']:>9.3f}")
    (OUT / "probe_b5b.json").write_text(json.dumps(r, indent=1))
    print(f"\nwritten {OUT / 'probe_b5b.json'}")
