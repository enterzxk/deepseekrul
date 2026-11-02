"""Probe B4: is the condition map good enough to re-render, and does the
condition also change the noise?

(a) Re-rendering.  If x' = a_c' + b_c' * ((x - a_c) / b_c) turns a row recorded
    under condition c into a plausible row under c', then a trajectory can be
    replayed under a *different* operating schedule while its health state, and
    therefore its RUL label, stays exactly valid.  That gives condition-shift
    test sets with trustworthy labels, an evaluation the C-MAPSS literature does
    not have.  Validity is checked against real rows of the target condition.

(b) Noise.  The affine map was fitted to mean curves.  If the residual
    covariance after alignment still depends on the condition, a second
    mechanism is available (condition-indexed observation noise, i.e. weighting
    channels by their reliability in the current condition).
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
MIN_ROWS = 40


def curves_by_condition(xs, rul, lab, grid) -> np.ndarray:
    """(6, anchors, d) mean standardised row at each anchor, NaN where thin."""
    d = xs.shape[1]
    out = np.full((6, len(grid), d), np.nan)
    for c in range(6):
        m = lab == c
        r, x = rul[m], xs[m]
        for i, rr in enumerate(grid):
            k = np.abs(r - rr) < 1.0
            if k.sum() >= MIN_ROWS:
                out[c, i] = x[k].mean(0)
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
        engines = tr.engines()
        maps, base = B3.fit_maps(xs, tr.rul, lab, tr.unit, engines, B1.GRID)

        grid = B1.GRID
        real = curves_by_condition(xs, tr.rul, lab, grid)          # (6, A, d)

        # ---------------- (a) re-rendering --------------------------------- #
        pairs = []
        for src in range(6):
            for dst in range(6):
                if src == dst:
                    continue
                a_s, b_s = maps[src]
                a_d, b_d = maps[dst]
                m = lab == src
                z_new = a_d + b_d * ((xs[m] - a_s) / b_s)
                r_src = tr.rul[m]
                rendered = np.full((len(grid), xs.shape[1]), np.nan)
                for i, rr in enumerate(grid):
                    k = np.abs(r_src - rr) < 1.0
                    if k.sum() >= MIN_ROWS:
                        rendered[i] = z_new[k].mean(0)
                ok = np.isfinite(rendered).all(1) & np.isfinite(real[dst]).all(1)
                if ok.sum() < 10:
                    continue
                err = float(np.sqrt(np.mean((rendered[ok] - real[dst][ok]) ** 2)))
                a, b = B1.fit_diag(rendered[ok].T, real[dst][ok].T)
                pairs.append({"src": src, "dst": dst, "err": err, "anchors": int(ok.sum()),
                              "max_abs_offset": float(np.abs(a).max()),
                              "max_gain_dev": float(np.abs(b - 1).max())})

        # ---------------- (b) residual noise ------------------------------- #
        za = B3.align_rows(tr.x, lab, maps, mu, sd)
        # population mean trajectory read at each row's own RUL, in the same units
        keep = np.isfinite(base).all(1)
        Hgrid = base[keep]                       # (anchors, d)
        rg = grid[keep]
        H = np.stack([np.interp(np.clip(tr.rul, rg.min(), rg.max()), rg, Hgrid[:, j])
                      for j in range(Hgrid.shape[1])], axis=1)
        resid = za - H
        noise = {}
        for c in range(6):
            R = resid[lab == c]
            ev = np.linalg.eigvalsh(np.cov(R.T))[::-1]
            noise[c] = {"trace": float(ev.sum()), "pc1_share": float(ev[0] / ev.sum()),
                        "top5": ev[:5].round(4).tolist(), "n": int(len(R))}
        traces = np.array([noise[c]["trace"] for c in range(6)])
        # per-channel variance ratio across conditions: which channels get noisy
        per_ch = np.stack([resid[lab == c].var(0) for c in range(6)])
        res[sub] = {
            "rerender": {"median_err": float(np.median([p["err"] for p in pairs])),
                         "max_err": float(np.max([p["err"] for p in pairs])),
                         "median_gain_dev": float(np.median([p["max_gain_dev"] for p in pairs])),
                         "median_offset": float(np.median([p["max_abs_offset"] for p in pairs])),
                         "pairs": pairs},
            "noise": {"trace_by_cond": traces.round(4).tolist(),
                      "trace_ratio": float(traces.max() / max(traces.min(), 1e-12)),
                      "pc1_share": [round(noise[c]["pc1_share"], 3) for c in range(6)],
                      "per_channel_var_ratio": (per_ch.max(0) / np.maximum(per_ch.min(0), 1e-12)
                                                ).round(2).tolist(),
                      "detail": noise},
        }
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    r = main()
    for sub, v in r.items():
        rr, nz = v["rerender"], v["noise"]
        print(f"\n=== {sub}")
        print(f"  re-render vs real rows: median {rr['median_err']:.3f}  max {rr['max_err']:.3f} "
              f"(standardised units)")
        print(f"  residual affine between rendered and real: gain dev {rr['median_gain_dev']:.3f}  "
              f"offset {rr['median_offset']:.3f}")
        worst = sorted(rr["pairs"], key=lambda d: -d["err"])[:3]
        print("  worst pairs:", [(d["src"], d["dst"], round(d["err"], 3)) for d in worst])
        print(f"  residual variance by condition: {nz['trace_by_cond']}")
        print(f"  max/min ratio {nz['trace_ratio']:.3f}   PC1 share {nz['pc1_share']}")
        print(f"  per-channel variance ratio across conditions: {nz['per_channel_var_ratio']}")
    (OUT / "probe_b4.json").write_text(json.dumps(r, indent=1))
    print(f"\nwritten {OUT / 'probe_b4.json'}")
