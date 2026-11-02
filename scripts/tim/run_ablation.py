"""Leave-one-out ablation of the matched-object construction (EAC).

EAC is four decisions stacked: (i) calibrate on the object that is deployed, (ii) divide
the score by a fitted variance function, (iii) keep the sign, (iv) choose the tail
allocation that minimises width. This script removes one decision at a time from the
full construction and leaves the other three in place -- including recomputing the free
tail for the variant, because the tail choice is part of each variant's own definition.
Every variant is evaluated on the same engines at the same nominal levels.

  EAC          matched object, score z = r / sigma(yhat), signed, free tail
  EAC no-object  fold-matched calibration (CV+), otherwise identical
  EAC no-norm    raw signed cycles as the score, otherwise identical
  EAC no-sign    |z| (a symmetric band), tail choice is vacuous
  EAC no-tail    z, equal tails

Width is not comparable across variants when coverage drifts, and it does drift (the
calibration and deployment RUL mixtures differ, which is the paper's Section V point).
The joint criterion is the Winkler interval score at the same nominal level, which is
what the increments below are measured on; coverage and width are reported alongside so
the reader can see what each variant trades. Increments carry an engine-level block
bootstrap 95 % CI (1500 resamples of the 93/235/97/228 test engines, averaged over the
three seeds).

Writes experiments/tim/ablation.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C  # noqa: E402
from run_matrix import LEVELS, best_gamma, fit_scale, scores_for_seed  # noqa: E402

OUT = C.ROOT / "experiments" / "tim"
VARIANTS = ("EAC", "EAC no-object", "EAC no-norm", "EAC no-sign", "EAC no-tail")
LABELS = {
    "EAC": "matched object, z = r/sigma, signed, free tail",
    "EAC no-object": "calibrate the fold model (CV+) instead of the deployed ensemble",
    "EAC no-norm": "raw signed cycles instead of z = r/sigma",
    "EAC no-sign": "symmetric band, |z| (tail choice vacuous)",
    "EAC no-tail": "equal tails instead of the width-minimising allocation",
}


def variants_for_seed(dev, tst, seed, lv):
    """All variants on one seed as (lo, hi) pairs, plus the test truth vector."""
    alpha = 1.0 - lv
    a2 = alpha / 2.0
    mu = np.stack([tst[(seed, f)][2] for f in range(5)], axis=1)
    yt = tst[(seed, 0)][1]
    pbar = mu.mean(1)

    r_ens, p_ens, _ = scores_for_seed(dev, seed, ensemble=True)
    r_sing, p_sing, fold_sing = scores_for_seed(dev, seed, ensemble=False)
    sig = fit_scale(p_ens, r_ens)
    sig_f = fit_scale(p_sing, r_sing)
    z, s_tst = r_ens / sig(p_ens), sig(pbar)
    z_f = r_sing / sig_f(p_sing)

    g = best_gamma(z, alpha)
    g_f = best_gamma(z_f, alpha)
    g_raw = best_gamma(r_ens, alpha)

    out = {}
    out["EAC"] = C.split_conformal(pbar, z, z, g, alpha - g, scale_test=s_tst)
    out["EAC no-object"] = C.cvplus(mu, z_f, z_f, fold_sing, g_f, alpha - g_f,
                                    scale_test=s_tst)
    out["EAC no-norm"] = C.split_conformal(pbar, r_ens, r_ens, g_raw, alpha - g_raw)
    out["EAC no-sign"] = C.split_sym(pbar, z, lv, scale_test=s_tst)
    out["EAC no-tail"] = C.split_conformal(pbar, z, z, a2, a2, scale_test=s_tst)
    out["_gamma"] = g
    return out, yt


def run_subset(subset):
    dev, tst, layer = C.load_fleet(subset)
    y = layer["y"].astype(float)
    rows = {v: {} for v in VARIANTS}
    gammas = []

    for lv in LEVELS:
        acc = {v: dict(cov=[], width=[], winkler=[]) for v in VARIANTS}
        iv = {v: [] for v in VARIANTS}
        for seed in C.SEEDS:
            d, yt = variants_for_seed(dev, tst, seed, lv)
            if lv == 0.90:
                gammas.append(d["_gamma"])
            for v in VARIANTS:
                lo, hi = d[v]
                acc[v]["cov"].append(C.coverage(lo, hi, yt))
                acc[v]["width"].append(C.width(lo, hi))
                acc[v]["winkler"].append(C.winkler(lo, hi, yt, lv))
                iv[v].append((lo, hi, yt))

        for v in VARIANTS:
            a = acc[v]
            rows[v][f"{lv:.2f}"] = dict(
                cov=float(np.mean(a["cov"])), width=float(np.mean(a["width"])),
                winkler=float(np.mean(a["winkler"])),
                cov_sd=float(np.std(a["cov"], ddof=1)), label=LABELS[v])

        units = np.arange(len(y))
        for v in VARIANTS[1:]:
            def stat(idx, v=v):
                """Winkler of this variant minus Winkler of the full construction."""
                def w(variant):
                    return np.mean([C.winkler(np.clip(l[idx], 0, C.MAX_RUL),
                                              np.clip(h[idx], 0, C.MAX_RUL), yy[idx], lv)
                                    for l, h, yy in iv[variant]])
                return float(w(v) - w("EAC"))

            bs = C.bootstrap_units(stat, units, n_boot=1500, seed=7)
            rows.setdefault("increment", {}).setdefault(f"{lv:.2f}", []).append(dict(
                step=f"EAC -> {v}", d_winkler=stat(units),
                lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                p_pos=float(np.mean(bs > 0))))

    rows["gamma@0.90"] = dict(mean=float(np.mean(gammas)))
    return rows


def main():
    report = {"variants": list(VARIANTS), "labels": LABELS, "subsets": {}}
    for subset in C.SUBSETS:
        report["subsets"][subset] = run_subset(subset)
        r = report["subsets"][subset]
        print(f"== {subset}  (gamma* = {r['gamma@0.90']['mean']:.3f})")
        for v in VARIANTS:
            a = r[v]["0.90"]
            print(f"   {v:14s} cov {a['cov']:.3f}  width {a['width']:6.2f}  "
                  f"winkler {a['winkler']:6.2f}")
        for inc in r["increment"]["0.90"]:
            print(f"     {inc['step']:26s} dWinkler {inc['d_winkler']:+7.2f} "
                  f"[{inc['lo']:+.2f}, {inc['hi']:+.2f}]")
    dst = OUT / "ablation.json"
    dst.write_text(json.dumps(report, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
