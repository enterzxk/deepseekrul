"""Probe B1: what does the operating condition actually do to the signal?

This is the falsification gate for H1 (``paper/LINE_B_DESIGN.md`` sections 2 and
5).  It is a data-level test, so it runs before any GPU work.

The design doc wrote H1b as "after stratifying by health level, dRUL/dt is the
same across conditions".  On C-MAPSS that test is vacuous: the label is
``last_cycle - cycle``, so dRUL/dt = 1 on every row of every condition by
construction.  The revision log records this.  What is *not* vacuous, and what
the architecture actually depends on, is the structure of the condition effect
on the observed degradation curve.  Candidates, in increasing flexibility:

  identity        C_c(r) = C(r)                      no effect at all
  diag            C_c(r) = a_c + b_c * C(r)          per-channel gain and offset
  lowrank-k       C_c(r) = a_c + B_c C(r), rank k    cross-channel mixing
  warp            C_c(r) = C(phi_c(r))               same trajectory, different speed
  warp+diag       C_c(r) = a_c + b_c * C(phi_c(r))   speed plus gain

Each model is fitted on one half of the engines of a condition and scored on the
other half, so flexibility is paid for.  Two numbers are reported per model:

  explained  1 - mean((model - heldout)^2) / mean((pooled - heldout)^2)
             fraction of the condition-induced deviation that survives the
             held-out test; 0 = useless, 1 = complete
  z          residual in units of the split-half noise; 1 means "no structure
             left beyond estimation noise"

Sensors are standardised per channel on the training rows before anything else,
so Euclidean distances across 21 channels are meaningful and constant channels
contribute nothing instead of degenerating the regressions.

The probe also reports the cycle-to-cycle increment decomposition, which is the
motivation number: how much of the jump in x between consecutive cycles is
condition switching rather than one cycle of degradation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data as D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "bline"
GRID = np.arange(5, 200, 5, dtype=float)
MIN_ENGINES = 15
REPEATS = 6


# --------------------------------------------------------------------------- #
# curves, all in standardised sensor units
# --------------------------------------------------------------------------- #

def standardise(f: D.Frame, ref: D.Frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = ref.x.mean(0)
    sd = ref.x.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (f.x - mu) / sd, mu, sd


def engine_curve(xs: np.ndarray, rul: np.ndarray, lab: np.ndarray, unit: np.ndarray,
                 eng: int, cond: int, grid: np.ndarray) -> np.ndarray:
    m = (unit == eng) & (lab == cond)
    d = xs.shape[1]
    if m.sum() < 3:
        return np.full((len(grid), d), np.nan)
    r = rul[m]
    order = np.argsort(r)
    r = r[order]
    x = xs[m][order]
    out = np.full((len(grid), d), np.nan)
    ok = (grid >= r.min()) & (grid <= r.max())
    out[ok] = np.stack([np.interp(grid[ok], r, x[:, j]) for j in range(d)], axis=1)
    return out


def condition_curve(xs, rul, lab, unit, engines, cond, grid) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack([engine_curve(xs, rul, lab, unit, e, cond, grid) for e in engines])
    support = np.isfinite(stack[:, :, 0]).sum(0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
    return mean, support


def pooled_curve(xs, rul, lab, unit, engines, grid, n_cond=6) -> np.ndarray:
    parts, weights = [], []
    for c in range(n_cond):
        m, s = condition_curve(xs, rul, lab, unit, engines, c, grid)
        parts.append(m)
        weights.append(s.astype(float))
    parts = np.stack(parts)
    weights = np.stack(weights)
    tot = weights.sum(0)
    with np.errstate(invalid="ignore"):
        out = np.nansum(parts * weights[:, :, None], axis=0) / np.maximum(tot, 1.0)[:, None]
    return out


# --------------------------------------------------------------------------- #
# alignment models.  curves are (d, G); d = channels, G = anchors.
# --------------------------------------------------------------------------- #

def fit_diag(target: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel a + b*base over anchors, closed form."""
    a = np.zeros(target.shape[0])
    b = np.ones(target.shape[0])
    for j in range(target.shape[0]):
        y, x = target[j], base[j]
        m = np.isfinite(y) & np.isfinite(x)
        if m.sum() < 3:
            continue
        xm, ym = x[m].mean(), y[m].mean()
        vx = ((x[m] - xm) ** 2).sum()
        if vx > 1e-12:
            b[j] = ((x[m] - xm) * (y[m] - ym)).sum() / vx
        a[j] = ym - b[j] * xm
    return a, b


def apply_diag(base: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, None] + b[:, None] * base


def fit_lowrank(target: np.ndarray, base: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """a + B*base with rank(B) <= rank."""
    d = target.shape[0]
    m = np.isfinite(target).all(0) & np.isfinite(base).all(0)
    if m.sum() < d + 2:
        return np.zeros(d), np.zeros((d, d))
    A = np.column_stack([np.ones(int(m.sum())), base[:, m].T])
    Y = target[:, m].T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    intercept = coef[0].copy()
    B = coef[1:].T.copy()
    if rank < d:
        U, s, Vt = np.linalg.svd(B)
        B = (U[:, :rank] * s[:rank]) @ Vt[:rank]
    return intercept, B


def project_onto_curve(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Nearest point on the polyline through the columns of ``curve``."""
    G = curve.shape[1]
    n = points.shape[1]
    best = np.full(n, np.inf)
    best_p = np.full((curve.shape[0], n), np.nan)
    for t in range(G - 1):
        p0, p1 = curve[:, t], curve[:, t + 1]
        seg = p1 - p0
        denom = float((seg ** 2).sum())
        s = np.zeros(n) if denom < 1e-12 else ((points - p0[:, None]) * seg[:, None]).sum(0) / denom
        s = np.clip(s, 0.0, 1.0)
        proj = p0[:, None] + seg[:, None] * s[None]
        dist = ((points - proj) ** 2).sum(0)
        upd = dist < best
        best[upd] = dist[upd]
        best_p[:, upd] = proj[:, upd]
    return best_p


def fit_warp_diag(target: np.ndarray, base: np.ndarray, iters: int = 12) -> np.ndarray:
    a = np.zeros(target.shape[0])
    b = np.ones(target.shape[0])
    proj = project_onto_curve(target, base)
    for _ in range(iters):
        proj_raw = project_onto_curve((target - a[:, None]) / b[:, None], base)
        a, b = fit_diag(target, proj_raw)
    proj_raw = project_onto_curve((target - a[:, None]) / b[:, None], base)
    return apply_diag(proj_raw, a, b)


# --------------------------------------------------------------------------- #
# main probe
# --------------------------------------------------------------------------- #

def incrment_decomposition(tr: D.Frame, lab: np.ndarray) -> dict:
    same, switch = [], []
    for e in tr.engines():
        idx = tr.index_of(e)
        idx = idx[np.argsort(tr.cycle[idx])]
        if len(idx) < 3:
            continue
        dx = np.diff(tr.x[idx], axis=0)
        keep = lab[idx][1:] == lab[idx][:-1]
        if keep.any():
            same.append(dx[keep])
        if (~keep).any():
            switch.append(dx[~keep])
    same, switch = np.concatenate(same), np.concatenate(switch)
    ratio = np.sqrt((switch ** 2).mean(0)) / np.maximum(np.sqrt((same ** 2).mean(0)), 1e-9)
    return {"same_rms": float(np.sqrt((same ** 2).mean())),
            "switch_rms": float(np.sqrt((switch ** 2).mean())),
            "ratio": float(np.sqrt((switch ** 2).mean()) / np.sqrt((same ** 2).mean())),
            "frac_switched": float(len(switch) / (len(switch) + len(same))),
            "per_sensor_ratio": ratio.round(2).tolist()}


def run(subset: str) -> dict:
    tr = D.load(subset, "train")
    sc = D.setting_scale([tr])
    cen = D.fit_conditions([tr], k=6)
    lab = D.assign_conditions(tr, cen, sc)
    xs, mu, sd = standardise(tr, tr)
    engines = tr.engines()
    inc = incrment_decomposition(tr, lab)

    rng = np.random.default_rng(20261102)
    rec = []
    for c in range(6):
        eng_c = engines.copy()
        for rep in range(REPEATS):
            perm = rng.permutation(len(eng_c))
            ae, be = eng_c[perm[: len(eng_c) // 2]], eng_c[perm[len(eng_c) // 2:]]
            ca, sa = condition_curve(xs, tr.rul, lab, tr.unit, ae, c, GRID)
            cb, sb = condition_curve(xs, tr.rul, lab, tr.unit, be, c, GRID)
            base = pooled_curve(xs, tr.rul, lab, tr.unit, ae, GRID)
            ok = (sa >= MIN_ENGINES) & (sb >= MIN_ENGINES)
            if ok.sum() < 12:
                continue
            ca, cb, base = ca[ok].T, cb[ok].T, base[ok].T
            noise = float(np.mean((ca - cb) ** 2) / 2.0) + 1e-12
            ref = float(np.mean((base - cb) ** 2)) + 1e-12

            preds = {"identity": base,
                     "diag": apply_diag(base, *fit_diag(ca, base))}
            for k in (1, 2, 4, 8):
                a, B = fit_lowrank(ca, base, k)
                preds[f"lowrank{k}"] = a[:, None] + B @ base
            preds["warp"] = project_onto_curve(ca, base)
            preds["warp+diag"] = fit_warp_diag(ca, base)

            for name, p in preds.items():
                r = float(np.mean((p - cb) ** 2))
                rec.append({"cond": c, "rep": rep, "model": name,
                            "explained": 1.0 - r / ref, "z": float(np.sqrt(r / noise)),
                            "resid": r})
            # per-channel detail for the two interesting models
            if rep == 0:
                for name in ("diag", "warp+diag"):
                    rj = ((preds[name] - cb) ** 2).mean(1)
                    r0 = ((base - cb) ** 2).mean(1)
                    rec.append({"cond": c, "rep": -1, "model": name + ":per_channel",
                                "explained": None, "z": None,
                                "per_channel_explained": (1 - rj / np.maximum(r0, 1e-12)).round(3).tolist()})

    models = sorted({r["model"] for r in rec if r["explained"] is not None})
    summary = {}
    for name in models:
        e = np.array([r["explained"] for r in rec if r["model"] == name])
        z = np.array([r["z"] for r in rec if r["model"] == name])
        summary[name] = {"explained_median": float(np.median(e)),
                         "explained_p10": float(np.percentile(e, 10)),
                         "z_median": float(np.median(z)), "n": int(len(e))}
    per_channel = {r["model"]: r["per_channel_explained"]
                   for r in rec if r["model"].endswith(":per_channel")}
    return {"subset": subset, "increment": inc, "summary": summary,
            "per_channel": per_channel, "n_engines": int(len(engines)),
            "grid": GRID.tolist()}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for sub in ("FD002", "FD004"):
        r = run(sub)
        out[sub] = r
        inc = r["increment"]
        print(f"\n=== {sub}  engines={r['n_engines']}")
        print(f"  consecutive-cycle |dx|: same condition {inc['same_rms']:.4f}   "
              f"switched {inc['switch_rms']:.4f}   ratio {inc['ratio']:.1f}x   "
              f"({inc['frac_switched']:.0%} of rows switch)")
        print(f"  {'model':12s} {'explained':>10s} {'p10':>7s} {'z':>10s}")
        for name, s in sorted(r["summary"].items(), key=lambda kv: -kv[1]["explained_median"]):
            print(f"  {name:12s} {s['explained_median']:10.3f} "
                  f"{s['explained_p10']:7.3f} {s['z_median']:10.1f}")
        for k, v in r["per_channel"].items():
            print(f"  {k}: per-channel explained {v}")
    (OUT / "probe_b1.json").write_text(json.dumps(out, indent=1))
    print(f"\nwritten {OUT / 'probe_b1.json'}")
