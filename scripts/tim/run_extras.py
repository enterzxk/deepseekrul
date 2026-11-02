"""Supporting experiments: bootstrap over engines, decision consequences, calibration-size
curve, and runtime.

Bootstrap resamples *engines*, not windows -- the engine is the independent unit of the
benchmark, so a window-level bootstrap would understate the interval by roughly the design
effect.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
from run_matrix import best_gamma, fit_scale

OUT = C.ROOT / "experiments" / "tim"
OUT.mkdir(parents=True, exist_ok=True)
LV = 0.90
ALPHA = 1 - LV
L0 = 20.0          # intervention threshold, cycles
N_BOOT = 2000


def build(subset, seed):
    dev, tst, layer = C.load_fleet(subset)
    mu = np.stack([tst[(seed, f)][2] for f in range(5)], axis=1)
    y = tst[(seed, 0)][1]
    r, p, fold = [], [], []
    for f in range(5):
        yc = dev[(seed, f)][1]
        pc = np.mean([dev[(s, f)][2] for s in C.SEEDS], axis=0)
        r.append(yc - pc)
        p.append(pc)
        fold.append(np.full(len(yc), f))
    return dict(mu=mu, y=y, pbar=mu.mean(1), r=np.concatenate(r), p=np.concatenate(p),
                fold=np.concatenate(fold), layer=layer)


def intervals(d, lv=LV, cal_frac=1.0, rng=None):
    alpha = 1 - lv
    a2 = alpha / 2
    r, p, fold = d["r"], d["p"], d["fold"]
    if cal_frac < 1.0:
        keep = np.zeros(len(r), bool)
        for f in range(5):
            idx = np.where(fold == f)[0]
            k = max(10, int(round(cal_frac * len(idx))))
            keep[rng.choice(idx, size=k, replace=False)] = True
        r, p, fold = r[keep], p[keep], fold[keep]
    sig = fit_scale(p, r)
    z = r / sig(p)
    sz = sig(d["pbar"])
    g = best_gamma(z, alpha)
    out = {}
    out["EAC-sym"] = C.split_sym(d["pbar"], z, lv, scale_test=sz)
    out["EAC"] = C.split_conformal(d["pbar"], z, z, g, alpha - g, scale_test=sz)
    lo, hi = C.interval_from_pmf(d["layer"]["p_cens"], lv)
    out["layer"] = (lo.astype(float), hi.astype(float))
    return out


def main():
    report = {"level": LV, "threshold": L0, "subsets": {}}
    t0 = time.perf_counter()
    for subset in C.SUBSETS:
        d0 = build(subset, C.SEEDS[0])
        y = d0["y"]
        units = d0["layer"]["unit"]
        n = len(y)

        # ---------------- bootstrap over engines
        rng = np.random.default_rng(4242)
        per_seed = []
        for seed in C.SEEDS:
            d = build(subset, seed)
            iv = intervals(d)
            per_seed.append({k: (v[0], v[1]) for k, v in iv.items()})
        boots = {k: {"width": [], "cov": [], "gap": []} for k in per_seed[0]}
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            for k in boots:
                ws, cs = [], []
                for iv in per_seed:
                    lo, hi = iv[k]
                    ws.append(np.mean((hi - lo)[idx]))
                    cs.append(np.mean(((lo <= y) & (y <= hi))[idx]))
                boots[k]["width"].append(float(np.mean(ws)))
                boots[k]["cov"].append(float(np.mean(cs)))
        # paired width difference EAC vs EAC-sym (same resample, so the difference is paired)
        diff = np.array(boots["EAC-sym"]["width"]) - np.array(boots["EAC"]["width"])
        rel = diff / np.array(boots["EAC-sym"]["width"])
        dly = np.array(boots["EAC"]["width"]) - np.array(boots["layer"]["width"])
        report["subsets"][subset] = dict(
            n=n,
            width={k: C.summary(v["width"]) for k, v in boots.items()},
            cov={k: C.summary(v["cov"]) for k, v in boots.items()},
            width_saving_vs_sym=dict(mean=float(diff.mean()),
                                     lo=float(np.percentile(diff, 2.5)),
                                     hi=float(np.percentile(diff, 97.5)),
                                     rel=float(rel.mean())),
            width_over_layer=dict(mean=float(dly.mean()),
                                  lo=float(np.percentile(dly, 2.5)),
                                  hi=float(np.percentile(dly, 97.5))),
        )

        # ---------------- decision consequences
        dec = {}
        for k in per_seed[0]:
            viol, unnec, safe_frac = [], [], []
            for iv in per_seed:
                lo, hi = iv[k]
                safe = lo > L0
                viol.append(float(np.mean(safe & (y < L0))))
                unnec.append(float(np.mean((~safe) & (y >= L0))))
                safe_frac.append(float(np.mean(safe)))
            dec[k] = dict(late=float(np.mean(viol)), late_sd=float(np.std(viol)),
                          unnecessary=float(np.mean(unnec)), safe_frac=float(np.mean(safe_frac)))
        report["subsets"][subset]["decision"] = dec

        # ---------------- calibration-size curve
        curve = {}
        rng2 = np.random.default_rng(7)
        for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
            vals = {"EAC": [], "EAC-sym": []}
            for seed in C.SEEDS:
                d = build(subset, seed)
                iv = intervals(d, cal_frac=frac, rng=rng2)
                for k in vals:
                    lo, hi = iv[k]
                    vals[k].append((C.coverage(lo, hi, d["y"]), C.width(lo, hi)))
            curve[f"{frac:.1f}"] = {k: dict(cov=float(np.mean([v[0] for v in vv])),
                                            width=float(np.mean([v[1] for v in vv])))
                                    for k, vv in vals.items()}
        report["subsets"][subset]["calibration_curve"] = curve
        print(f"{subset}: saving {diff.mean():.1f} cycles ({rel.mean()*100:.0f}%), "
              f"EAC over layer {dly.mean():+.1f}; late {dec['EAC']['late']:.3f} "
              f"vs sym {dec['EAC-sym']['late']:.3f}")

    # ---------------- runtime
    d = build("FD004", C.SEEDS[0])
    t1 = time.perf_counter()
    for _ in range(20):
        r = d["r"]
        a2 = ALPHA / 2
        C.split_conformal(d["pbar"], r, r, a2, a2)
    t2 = time.perf_counter()
    report["runtime_ms_per_call"] = (t2 - t1) / 20 * 1000
    report["runtime_note"] = "quantile evaluation on 100 calibration scores, 228 test points"

    (OUT / "extras.json").write_text(json.dumps(report, indent=1))
    print(f"runtime {report['runtime_ms_per_call']:.2f} ms/call, total {time.perf_counter()-t0:.1f}s")
    print("wrote", OUT / "extras.json")


if __name__ == "__main__":
    main()
