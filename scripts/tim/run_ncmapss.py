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

    # ------------------------------------------------------------------ pooled protocol
    # The rule is fixed on DEVELOPMENT data only: the score sample is the concatenation
    # of every fleet's dev residuals (units that are not in any test set).  The method is
    # then applied once to the test queries and the labels are read for the first time.
    # No test label enters calibration, so this is the deployment protocol rather than a
    # leave-one-fleet-out construction on the test set.
    r_dev_all = np.concatenate([data[(f, "dev")]["y"] - data[(f, "dev")]["p"] for f in fleets])

    for lv in LEVELS:
        alpha = 1.0 - lv
        a2 = alpha / 2.0
        res = {}
        arms = {
            "asym": (r_dev_all, r_dev_all, a2, a2),
            "sym": (-np.abs(r_dev_all), np.abs(r_dev_all), alpha, alpha),
        }
        per_fleet = {}
        for tag, (slo, shi, alo, ahi) in arms.items():
            LO, HI = C.split_conformal(p, slo, shi, alo, ahi)
            cov = C.coverage(LO, HI, y)
            wid = C.width(LO, HI)
            cb = boot_units(u, lambda i: C.coverage(LO[i], HI[i], y[i]))
            wb = boot_units(u, lambda i: C.width(LO[i], HI[i]))
            cov_f = {f: C.coverage(LO[fl == f], HI[fl == f], y[fl == f]) for f in fleets}
            wid_f = {f: C.width(LO[fl == f], HI[fl == f]) for f in fleets}
            res[tag] = dict(
                cov=cov, cov_lo=float(np.percentile(cb, 2.5)), cov_hi=float(np.percentile(cb, 97.5)),
                width=wid, width_lo=float(np.percentile(wb, 2.5)), width_hi=float(np.percentile(wb, 97.5)),
                # macro = every fleet weighted equally, micro = every query weighted equally
                cov_macro=float(np.mean(list(cov_f.values()))),
                width_macro=float(np.mean(list(wid_f.values()))),
                per_fleet={f: dict(cov=cov_f[f], width=wid_f[f]) for f in fleets},
                n=int(len(y)), units=int(len(np.unique(u))), n_cal=int(len(r_dev_all)))
            per_fleet[tag] = res[tag]
        res["tax"] = float(1 - res["asym"]["width"] / res["sym"]["width"])
        res["tax_macro"] = float(1 - res["asym"]["width_macro"] / res["sym"]["width_macro"])
        report["pooled"][f"{lv:.2f}"] = res
        print(f"pooled lv={lv:.2f} asym cov {res['asym']['cov']:.3f} "
              f"[{res['asym']['cov_lo']:.3f},{res['asym']['cov_hi']:.3f}] w {res['asym']['width']:.1f} | "
              f"sym cov {res['sym']['cov']:.3f} w {res['sym']['width']:.1f} | tax {res['tax']*100:.0f}% "
              f"(macro {res['tax_macro']*100:.0f}%) | cal n={len(r_dev_all)}")

    (OUT / "ncmapss.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "ncmapss.json")


if __name__ == "__main__":
    main()
