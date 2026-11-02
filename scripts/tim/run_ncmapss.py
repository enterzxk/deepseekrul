"""N-CMAPSS: independent benchmark, same two questions.

The point-prediction pipeline is the one frozen in paper/NCMAPSS_PROTOCOL.md (54-64-32-1
MLP, three seeds, leave-one-unit-out on dev). What is added here is the interval layer.

Two things are reported, and they are not equally informative:

* coverage -- 39 test units. The 95% interval for a coverage of 0.90 at n=39 is
  roughly +/-0.10, so this benchmark can confirm a gross failure but cannot separate two
  methods that differ by a few percent. Reported with a unit-clustered bootstrap.
* width and arm asymmetry -- these are averages over 2,500 cycle-level queries, measured
  precisely, and they test the mechanism (the two arms are not the same length) rather
  than the level.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C

OUT = C.ROOT / "experiments" / "tim"
OUT.mkdir(parents=True, exist_ok=True)
MAXNC = 100.0
LEVELS = (0.90, 0.50)


def load():
    rows = list(csv.DictReader(open(C.ROOT / "experiments" / "ncmapss" / "preds" / "preds.csv")))
    out = {}
    for r in rows:
        key = (r["fleet"], r["split"])
        out.setdefault(key, {"u": [], "c": [], "y": [], "p": []})
        out[key]["u"].append(int(r["unit"]))
        out[key]["c"].append(int(r["cyc"]))
        out[key]["y"].append(float(r["y"]))
        out[key]["p"].append(float(r["yhat"]))
    for k, v in out.items():
        o = np.lexsort((v["c"], v["u"]))
        for f in v:
            v[f] = np.asarray(v[f])[o]
    return out


def boot_units(units, stat, n_boot=2000, seed=3):
    rng = np.random.default_rng(seed)
    ks = np.unique(units)
    idx = {u: np.where(units == u)[0] for u in ks}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(ks, size=len(ks), replace=True)
        vals.append(stat(np.concatenate([idx[u] for u in draw])))
    return np.asarray(vals)


def main():
    data = load()
    fleets = sorted({k[0] for k in data})
    report = {"fleets": {}, "pooled": {}, "level": LEVELS[0]}

    pooled = {"y": [], "p": [], "u": [], "f": []}
    for f in fleets:
        d = data[(f, "dev")]
        t = data[(f, "test")]
        for lv in LEVELS:
            alpha = 1 - lv
            a2 = alpha / 2
            r_dev = d["y"] - d["p"]
            lo_a, hi_a = C.split_conformal(t["p"], r_dev, r_dev, a2, a2)
            lo_s, hi_s = C.split_sym(t["p"], r_dev, lv)
            r = report["fleets"].setdefault(f, {})
            r[f"asym@{lv:.2f}"] = dict(cov=C.coverage(lo_a, hi_a, t["y"]), width=C.width(lo_a, hi_a),
                                       winkler=C.winkler(lo_a, hi_a, t["y"], lv))
            r[f"sym@{lv:.2f}"] = dict(cov=C.coverage(lo_s, hi_s, t["y"]), width=C.width(lo_s, hi_s),
                                      winkler=C.winkler(lo_s, hi_s, t["y"], lv))
        pooled["y"].append(t["y"])
        pooled["p"].append(t["p"])
        pooled["u"].append(np.array([f"{f}-{u}" for u in t["u"]]))
        pooled["f"].append(np.array([f] * len(t["y"])))
        # arm lengths, the mechanism
        r = d["y"] - d["p"]
        report["fleets"][f]["arms"] = dict(
            q05=float(np.quantile(r, 0.05)), q95=float(np.quantile(r, 0.95)),
            lower=float(-np.quantile(r, 0.05)), upper=float(np.quantile(r, 0.95)),
            ratio=float(-np.quantile(r, 0.05) / max(np.quantile(r, 0.95), 1e-9)),
            n_dev=int(len(d["y"])), n_test=int(len(t["y"])))

    y = np.concatenate(pooled["y"])
    p = np.concatenate(pooled["p"])
    u = np.concatenate(pooled["u"])
    fl = np.concatenate(pooled["f"])

    for lv in LEVELS:
        alpha = 1 - lv
        a2 = alpha / 2
        res = {}
        for tag, (slo, shi) in {"asym": (y - p, y - p), "sym": (-np.abs(y - p), np.abs(y - p))}.items():
            # leave-one-fleet-out calibration, so no test cycle is scored from its own fleet
            LO = np.full(len(y), np.nan)
            HI = np.full(len(y), np.nan)
            for f in fleets:
                cal = fl != f
                te = fl == f
                lo, hi = C.split_conformal(p[te], slo[cal], shi[cal], a2, a2)
                LO[te], HI[te] = lo, hi
            cov = C.coverage(LO, HI, y)
            wid = C.width(LO, HI)
            cb = boot_units(u, lambda i: C.coverage(LO[i], HI[i], y[i]))
            wb = boot_units(u, lambda i: C.width(LO[i], HI[i]))
            res[tag] = dict(cov=cov, cov_lo=float(np.percentile(cb, 2.5)), cov_hi=float(np.percentile(cb, 97.5)),
                            width=wid, width_lo=float(np.percentile(wb, 2.5)), width_hi=float(np.percentile(wb, 97.5)),
                            n=int(len(y)), units=int(len(np.unique(u))))
        res["tax"] = float(1 - res["asym"]["width"] / res["sym"]["width"])
        report["pooled"][f"{lv:.2f}"] = res
        print(f"pooled lv={lv:.2f} asym cov {res['asym']['cov']:.3f} "
              f"[{res['asym']['cov_lo']:.3f},{res['asym']['cov_hi']:.3f}] w {res['asym']['width']:.1f} | "
              f"sym cov {res['sym']['cov']:.3f} w {res['sym']['width']:.1f} | tax {res['tax']*100:.0f}%")

    (OUT / "ncmapss.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "ncmapss.json")


if __name__ == "__main__":
    main()
