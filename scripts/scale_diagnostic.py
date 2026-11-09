"""H005: when is a conditional (epistemic) scale useful?

Pre-registered in docs/wiki/hypotheses.md (H005, 2026-11-09, before this script
was first run).  Gates G-A..G-D and tau=0.5 are frozen there; do not tune them.

The phenomenon (MC-Dropout scale effect splits by subset) is already observed.
This script tests the MECHANISM: does a calibration-side diagnostic -- the rank
correlation between an engine's mean dropout scale and its mean absolute centre
error, computed on CALIBRATION engines only -- predict whether using that scale
helps or hurts?

No training, no interval/calibration changes, 0 GPU.  Reads only the frozen
artefacts (experiments/punet_mcd80, experiments/punet_matrix).

Usage:  python scripts/scale_diagnostic.py [--json out.json]
Exit 0 always (this is a measurement, not a build); the printed PASS/FAIL is the
verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 7, 2024, 13]
MULTI = {"FD002", "FD004"}          # multi operating-condition subsets
TAU = 0.5                            # frozen in the pre-registration


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, a=0.1):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def nominal_scaled(y, mu, w, cal, tst, delta=0.1):
    q = float(np.quantile(np.abs(y[cal] - mu[cal]) / np.maximum(w[cal], 1e-9),
                          min(1.0, (1 - delta) * (1 + 1 / cal.sum())), method="higher"))
    L, H = mu[tst] - q * w[tst], mu[tst] + q * w[tst]
    return float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H)


def spearman(a, b) -> float:
    ra = pd.Series(np.asarray(a, float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, float)).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cell_diagnostics(mcd_root: Path, subset: str, fold: int, seed: int) -> dict | None:
    p = mcd_root / subset / f"f{fold}_s{seed}" / "preds_mcd.csv"
    if not p.exists():
        return None
    m = np.genfromtxt(p, delimiter=",", names=True)
    y, mu, sd, units = m["y_true"], m["mu"], m["sd"], m["unit"]
    cal, tst = engine_split(units)

    # ---- calibration-side diagnostics (labels used only on calibration engines)
    uc = np.unique(units[cal])
    s_bar = np.array([sd[cal & (units == u)].mean() for u in uc])
    a_bar = np.array([np.abs(y[cal & (units == u)] - mu[cal & (units == u)]).mean() for u in uc])
    rho_cal = spearman(s_bar, a_bar)
    delta_eng = float(np.std(s_bar) / max(np.mean(s_bar), 1e-12))

    ut = np.unique(units[tst])
    s_test = np.array([sd[tst & (units == u)].mean() for u in ut])
    shift_sd = float(abs(s_test.mean() - s_bar.mean()) / max(s_bar.mean(), 1e-12))

    # ---- outcomes at the nominal level
    cov_mcd, w_mcd = nominal_scaled(y, mu, sd, cal, tst)
    cov_cmc, w_cmc = nominal_scaled(y, mu, np.ones_like(sd), cal, tst)
    return dict(subset=subset, fold=fold, seed=seed,
                rho_cal=rho_cal, delta_eng=delta_eng, shift_sd=shift_sd,
                gap=cov_mcd - 0.90, r_scale=(w_mcd - w_cmc) / w_cmc * 100,
                w_mcd=w_mcd, w_cmc=w_cmc)


def boot_ci_pairs(x, y, n=1000, seed=0):
    """Cluster bootstrap over fold units, resampling (x, y) PAIRS together.

    The earlier version resampled only the predictor and held the outcome fixed,
    which breaks the pairing and makes the interval invalid (caught in the L2
    review, D022).  This version keeps each fold's (rho_cal, r_scale) pair intact.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        s = spearman(x[idx], y[idx])
        if not np.isnan(s):
            out.append(s)
    if not out:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(out, [2.5, 97.5]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcd", default="experiments/punet_mcd80")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    rows = [d for s in SUBSETS for f in range(5) for sd in SEEDS
            if (d := cell_diagnostics(Path(args.mcd), s, f, sd)) is not None]
    df = pd.DataFrame(rows)
    print(f"cells with diagnostics: {len(df)}")

    fold = df.groupby(["subset", "fold"]).agg(
        rho_cal=("rho_cal", "mean"), delta_eng=("delta_eng", "mean"),
        shift_sd=("shift_sd", "mean"), gap=("gap", "mean"),
        r_scale=("r_scale", "mean")).reset_index()
    print(f"fold units: {len(fold)}")

    # ---------------- G-A ----------------
    rho_a = spearman(fold.rho_cal, fold.r_scale)
    lo, hi = boot_ci_pairs(fold.rho_cal, fold.r_scale)
    ga = bool(rho_a <= -0.35 and hi < 0)
    print(f"\n[G-A] Spearman(rho_cal, r_scale) = {rho_a:+.3f}  bootstrap95=[{lo:+.3f}, {hi:+.3f}]"
          f"  gate: <=-0.35 and CI_hi<0  -> {'PASS' if ga else 'FAIL'}")

    # ---------------- G-B ----------------
    multi = fold[fold.subset.isin(MULTI)].rho_cal.mean()
    single = fold[~fold.subset.isin(MULTI)].rho_cal.mean()
    per_fold = []
    for f in range(5):
        ff = fold[fold.fold == f]
        per_fold.append(ff[ff.subset.isin(MULTI)].rho_cal.mean() - ff[~ff.subset.isin(MULTI)].rho_cal.mean())
    same_sign = int(sum(1 for x in per_fold if x > 0))
    gb = bool(multi - single >= 0.20 and same_sign >= 4)
    print(f"[G-B] rho_cal multi={multi:+.3f} single={single:+.3f} diff={multi-single:+.3f} "
          f"| per-fold diffs={['%+.2f' % x for x in per_fold]} ({same_sign}/5 positive)"
          f"  gate: diff>=0.20 and >=4/5 -> {'PASS' if gb else 'FAIL'}")

    # ---------------- G-C ----------------
    rho_c = spearman(fold.shift_sd, fold.gap)
    lo_c, hi_c = boot_ci_pairs(fold.shift_sd, fold.gap)
    gc = bool(rho_c <= -0.30 and hi_c < 0)
    print(f"[G-C] Spearman(shift_sd, gap) = {rho_c:+.3f}  bootstrap95=[{lo_c:+.3f}, {hi_c:+.3f}]"
          f"  gate: <=-0.30 and CI_hi<0 -> {'PASS' if gc else 'FAIL'}")

    # ---------------- G-D ----------------
    df["use_scale"] = df.rho_cal >= TAU
    df["w_rule"] = np.where(df.use_scale, df.w_mcd, df.w_cmc)
    df["r_rule"] = (df.w_rule - df.w_cmc) / df.w_cmc * 100
    mean_rule = float(df.r_rule.mean())
    fd003_rule = float(df[df.subset == "FD003"].r_rule.mean())
    always = float(df.r_scale.mean())
    # G-D wording (pre-registered): "rule mean <= -2% AND FD003 mean >= -2%"
    # means FD003 must not be made WORSE than +2% relative to never-use.
    # The first implementation wrote `fd003_rule >= -2.0`, which instead demands
    # FD003 be at least 2% BETTER -- a stricter, semantically wrong test (D022).
    # Both versions fail here; the corrected form is reported.
    gd = bool(mean_rule <= -2.0 and fd003_rule <= 2.0)
    print(f"[G-D] rule(tau={TAU}) mean r={mean_rule:+.2f}%  FD003 mean={fd003_rule:+.2f}%  "
          f"| always-use mean={always:+.2f}%  never-use=0.00%  "
          f"cells using scale: {int(df.use_scale.sum())}/{len(df)}"
          f"  gate: mean<=-2% and FD003<=+2% -> {'PASS' if gd else 'FAIL'}")

    print("\n--- per-subset (cell level) ---")
    print(df.groupby("subset")[["rho_cal", "delta_eng", "shift_sd", "gap", "r_scale", "r_rule"]]
          .mean().round(3).to_string())
    print("\n--- fold level table ---")
    print(fold.round(3).to_string(index=False))

    # ---------------- POST-HOC: estimator variance of rho_cal ----------------
    # Not part of the pre-registered gates.  G-A failed; this asks whether the
    # diagnostic is even estimable at the available calibration size, by
    # bootstrapping engines *within* each cell.
    print("\n--- POST-HOC: within-cell bootstrap SD of rho_cal (500 engine resamples) ---")
    rng = np.random.default_rng(0)
    est_rows = []
    for s in SUBSETS:
        sds, ncal, rhos, hw, undef = [], [], [], [], []
        for f in range(5):
            for sd in SEEDS:
                p = Path(args.mcd) / s / f"f{f}_s{sd}" / "preds_mcd.csv"
                if not p.exists():
                    continue
                m = np.genfromtxt(p, delimiter=",", names=True)
                y, mu, sdv, units = m["y_true"], m["mu"], m["sd"], m["unit"]
                cal, _ = engine_split(units)
                uc = np.unique(units[cal])
                sb = np.array([sdv[cal & (units == u)].mean() for u in uc])
                ab = np.array([np.abs(y[cal & (units == u)] - mu[cal & (units == u)]).mean() for u in uc])
                rhos.append(spearman(sb, ab))
                ncal.append(uc.size)
                bs = [spearman(sb[i], ab[i]) for i in rng.integers(0, uc.size, (500, uc.size))]
                undef.append(float(np.mean([np.isnan(b) for b in bs])))
                bs = [b for b in bs if not np.isnan(b)]
                sds.append(float(np.std(bs)))
                # per-cell percentile interval (NOT 1.96 * mean SD; D022 fix)
                hw.append(float((np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2) if bs else float("nan"))
        est_rows.append(dict(subset=s, n_cal=float(np.mean(ncal)), rho_cal=float(np.nanmean(rhos)),
                             boot_sd=float(np.mean(sds)), half95_percell=float(np.nanmean(hw)),
                             undef_frac=float(np.mean(undef))))
    est = pd.DataFrame(est_rows)
    print(est.round(3).to_string(index=False))

    verdict = dict(GA=ga, GB=gb, GC=gc, GD=gd,
                   spearman_A=rho_a, ci_A=[lo, hi],
                   rho_multi=multi, rho_single=single, diff=multi - single,
                   spearman_C=rho_c, ci_C=[lo_c, hi_c],
                   rule_mean=mean_rule, rule_fd003=fd003_rule, always_use=always,
                   tau=TAU)
    print("\nVERDICT:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in verdict.items()}, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"cells": df.to_dict("records"), "folds": fold.to_dict("records"),
             "estimator_variance": est.to_dict("records"), "verdict": verdict},
            indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
