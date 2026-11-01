"""Run the frozen calibration comparison on N-CMAPSS, reusing the C-MAPSS implementations.

The point of importing `v2_compare` rather than reimplementing is that any difference in
the numbers can then only come from the data, never from a subtly different metric.

`layer.set_grid(100)` must run BEFORE `v2_compare` is imported: v2_compare binds K and
MAX_RUL at module import, and the grid is 0..100 rather than 0..125 because N-CMAPSS counts
remaining life in flight cycles and tops out near 100. The layer's own constants are
dimensionless in the grid scale (see layer.set_grid), so nothing else changes.

Protocols, mirroring the C-MAPSS four-way split:
  pooled    fit on all nine fleets' dev rows, score each fleet's test
  loso      fit on the other eight fleets' dev, score the held-out fleet's test
  perfleet  fit on the fleet's own dev only -- the only regime where 400 dev rows
            fit 66 parameters without help

Bands are fixed by paper/NCMAPSS_PROTOCOL.md: [0,20) [20,40) [40,60) [60,80) [80,100).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts" / "cal"))

import layer  # noqa: E402

layer.set_grid(100)                      # MUST precede the v2_compare import

import v2_compare as V  # noqa: E402
from layer import (K, MAX_RUL, choose_scale, cover, fit_layer, gen_pmf,  # noqa: E402
                   interval, pmf_np)

BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
SPEC = {"cens": ("mlp", "censored"), "trunc": ("mlp", "density"),
        "emos_c": ("linear", "censored"), "emos_u": ("linear", "density")}


def load_preds(path):
    per = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            d = per.setdefault(r["fleet"], {"dev": [], "test": []})
            d[r["split"]].append((int(r["unit"]), float(r["y"]), float(r["yhat"])))
    out = {}
    for f, d in per.items():
        out[f] = {}
        for split in ("dev", "test"):
            rows = d[split]
            out[f][split] = {
                "unit": np.array([r[0] for r in rows], dtype=int),
                "y": np.array([r[1] for r in rows], dtype=np.float64),
                "yhat": np.array([r[2] for r in rows], dtype=np.float64),
            }
    return out


def merge(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ("unit", "yhat", "y")}


def fit_all(dev):
    """Fit every method on one dev sample and return name -> pmf(yhat, y) callables."""
    nets, scales, times = {}, {}, {}
    for name, (kind, mode) in SPEC.items():
        t = time.time()
        nets[name] = fit_layer(dev, kind=kind, mode=mode)
        scales[name], _ = choose_scale(
            lambda d, k=kind, m=mode: fit_layer(d, kind=k, mode=m), dev)
        times[name] = time.time() - t

    s0 = float(np.std(dev["y"] - dev["yhat"]))
    base = nets["emos_c"]
    sd_fn = lambda v: V._params(base, v)[1]          # noqa: E731
    z_exact = V.make_crude(base, dev, keep_capped=True)
    z_drop = V.make_crude(base, dev, keep_capped=False)
    z_km = V.make_crude_km(base, dev)
    iso_e = V.make_isotonic(base, dev, keep_capped=True)
    iso_d = V.make_isotonic(base, dev, keep_capped=False)

    def _mk(keep):
        def f(d):
            b = fit_layer(d, kind="linear", mode="censored")
            return b, V.make_crude(b, d, keep_capped=keep)
        return f

    def _gen(pmf_fn):
        def g(pair, yh, s):
            b, z = pair
            return pmf_fn(z, b, yh, scale=s)
        return g

    def _km(d):
        b = fit_layer(d, kind="linear", mode="censored")
        return b, V.make_crude_km(b, d)

    for tag, fit_fn, gen_fn in (("crude_exact", _mk(True), _gen(V.crude_pmf)),
                                ("crude_drop", _mk(False), _gen(V.crude_pmf)),
                                ("crude_km", _km, _gen(V.crude_km_pmf))):
        t = time.time()
        scales[tag], _ = V.choose_scale_generic(fit_fn, gen_fn, dev)
        times[tag] = time.time() - t

    t = time.time()
    sc = V.split_conformal(dev)
    nsc = V.split_conformal(dev, sd=sd_fn)
    cq = V.cqr(dev)
    times["conformal"] = time.time() - t

    def dist(yh):
        p = {
            "cens": gen_pmf(nets["cens"], yh, scale=scales["cens"]),
            "trunc": gen_pmf(nets["trunc"], yh, mode="density", scale=scales["trunc"]),
            "emos_c": gen_pmf(nets["emos_c"], yh, scale=scales["emos_c"]),
            "emos_u": gen_pmf(nets["emos_u"], yh, mode="density", scale=scales["emos_u"]),
            "homosc": pmf_np(yh, np.full(len(yh), s0), "censored"),
            "crude_exact": V.crude_pmf(z_exact, base, yh, scale=scales["crude_exact"]),
            "crude_drop": V.crude_pmf(z_drop, base, yh, scale=scales["crude_drop"]),
            "crude_km": V.crude_km_pmf(z_km, base, yh, scale=scales["crude_km"]),
            "isot_exact": V.isotonic_pmf(iso_e, base, yh),
            "isot_drop": V.isotonic_pmf(iso_d, base, yh),
        }
        iv = {"scp": V.apply_conformal(sc, yh), "nscp": V.apply_conformal(nsc, yh),
              "cqr": V.apply_cqr(cq, yh)}
        return p, iv

    return dist, scales, times, s0


def eval_fleet(dist, te, n_boot=2000):
    y, yh = te["y"], te["yhat"]
    p_te, iv = dist(yh)
    rec = {k: V.score(v, y) for k, v in p_te.items()}
    for k, v in p_te.items():
        rec[k]["sharp90"] = V.width_at_coverage(v, y, grid=np.linspace(0.5, 3.0, 26))
        pm, sm = V.match_coverage(v, y, 0.90)
        rec[k]["crps_at90"] = None if pm is None else float(V.crps_proper(pm, y).mean())
        rec[k]["widen_to_90"] = None if pm is None else sm
    for k, (lo_, hi_) in iv.items():
        rec[k] = V.score_interval(lo_, hi_, y)
        rec[k]["sharp90"] = float(np.mean(hi_ - lo_))

    bands = []
    for a, b in BANDS:
        m = (y >= a) & (y < b)
        if m.sum() == 0:
            continue
        row = {"band": f"[{a},{b})", "n": int(m.sum())}
        for name in ("cens", "trunc", "emos_c", "homosc", "crude_exact", "scp"):
            e = rec[name]["_per"]["err90"][m]
            lo95, hi95 = V.wilson(int(e.sum()), int(m.sum()))
            mu = (p_te[name] * np.arange(K)[None, :]).sum(1) if name in p_te else None
            row[name] = {"picp": float(e.mean()), "ci": [lo95, hi95],
                         "bias": float((mu[m] - y[m]).mean()) if mu is not None
                         else float(np.mean(np.clip(yh[m], 0, MAX_RUL) - y[m]))}
        bands.append(row)

    pb = {}
    rng = np.random.default_rng(0)
    units = np.unique(te["unit"])
    idxs = [np.where(te["unit"] == u)[0] for u in units]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idxs), len(idxs))
        boots.append(np.concatenate([idxs[i] for i in pick]))
    for other in ("trunc", "emos_c", "homosc", "crude_km", "isot_exact"):
        d = rec[other]["_per"]["crps"] - rec["cens"]["_per"]["crps"]
        bs = np.array([d[b].mean() for b in boots])
        pb[other] = [float(d.mean()), float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))]
    return {"n": int(len(y)), "units": int(len(units)), "bands": bands,
            "methods": {k: {kk: vv for kk, vv in v.items() if kk != "_per"}
                        for k, v in rec.items()},
            "paired_bootstrap_vs_cens": pb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(ROOT / "experiments/ncmapss/preds/preds.csv"))
    ap.add_argument("--out", default=str(ROOT / "experiments/ncmapss"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--protocol", default="pooled",
                    choices=["pooled", "loso", "perfleet", "all"])
    a = ap.parse_args()

    data = load_preds(Path(a.preds))
    fleets = sorted(data)
    print(f"{len(fleets)} fleets: {fleets}")
    print(f"K={K} MAX_RUL={MAX_RUL}")

    protos = ["pooled", "loso", "perfleet"] if a.protocol == "all" else [a.protocol]
    for proto in protos:
        t0 = time.time()
        out = {"protocol": proto, "grid": {"K": K, "max_rul": MAX_RUL},
               "bands": [f"[{x},{y})" for x, y in BANDS], "fleets": {}}
        if proto == "pooled":
            dist, scales, times, s0 = fit_all(merge([data[f]["dev"] for f in fleets]))
            for f in fleets:
                out["fleets"][f] = eval_fleet(dist, data[f]["test"], a.n_boot)
        elif proto == "perfleet":
            for f in fleets:
                dist, sc, tm, s0 = fit_all(data[f]["dev"])
                out["fleets"][f] = eval_fleet(dist, data[f]["test"], a.n_boot)
                print(f"  {f} per-fleet: cens CRPS "
                      f"{out['fleets'][f]['methods']['cens']['crps']:.2f}")
            scales, times, s0 = {}, {}, float("nan")
        else:
            for f in fleets:
                fit_fleets = [g for g in fleets if g != f]
                dist, sc, tm, s0 = fit_all(merge([data[g]["dev"] for g in fit_fleets]))
                out["fleets"][f] = eval_fleet(dist, data[f]["test"], a.n_boot)
                print(f"  {f} loso: cens CRPS "
                      f"{out['fleets'][f]['methods']['cens']['crps']:.2f}")
            scales, times, s0 = {}, {}, float("nan")
        out["meta"] = {"scales": scales, "fit_seconds": times, "s0": s0,
                       "seconds": round(time.time() - t0, 1)}
        p = Path(a.out) / f"cal_{proto}.json"
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"{proto}: {p}  ({out['meta']['seconds']}s)\n")


if __name__ == "__main__":
    main()
