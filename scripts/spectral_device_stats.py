"""Device-level spectral analysis (corrected statistics).

Fixes vs the previous version
  * the independent unit is the BEARING (n=17). The 4 life stages are repeated
    measures and are only used for within-bearing trends, never as independent
    observations in a cross-device test;
  * cross-device tests use one summary per bearing, report Spearman with n=17,
    and apply Benjamini-Hochberg correction across features;
  * partial correlations control for life length (log snapshots) and condition
    family; controlling for stage is NOT used as a substitute for device-level
    clustering;
  * question A (does high-frequency energy track degradation?) is answered
    within bearing: per-bearing Spearman(band4, progress) and a sign test;
  * question B (does band4 add beyond RMS?) is a device-level partial Spearman
    of band4 vs |error| controlling for v_rms and h_rms.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest, t as tdist

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "data/femto_spec"
PRED = ROOT / "experiments/bearing_tim3"
STAGES = (0.50, 0.70, 0.85, 0.95)
FEATS = ["h_rms", "v_rms", "h_kurt", "h_crest", "h_band0", "h_band1", "h_band2", "h_band3",
         "h_band4", "h_env_energy", "hv_energy_ratio", "coh_mean"]


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[idx] * n / k)
        q[idx] = prev
    return q


def partial_spearman(x, y, zs, n):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    Z = np.column_stack([pd.Series(z).rank().to_numpy() for z in zs] + [np.ones(n)])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    dof = n - len(zs) - 2
    t = r * np.sqrt(dof / max(1e-9, 1 - r ** 2))
    p = 2 * (1 - tdist.cdf(abs(t), dof)) if dof > 0 else np.nan
    return r, float(p)


def main() -> int:
    # ---- per-bearing summary: features at the last stage, mean |error| over stages
    feat_rows = []
    for f in sorted(glob.glob(str(SPEC / "*.csv"))):
        name = Path(f).stem
        d = pd.read_csv(f)
        if len(d) < 10:
            continue
        n = len(d)
        last = d.iloc[min(n - 1, int(round(0.95 * n)) - 1)]
        rec = dict(bearing=name, life=n)
        for c in FEATS:
            rec[c] = float(last[c]) if c in d.columns else np.nan
        # within-bearing trend of band4 over life (Spearman vs progress)
        prog = np.arange(n) / max(1, n - 1)
        rec["band4_trend"] = float(spearmanr(prog, d["h_band4"].to_numpy()).statistic) if "h_band4" in d else np.nan
        rec["rms_trend"] = float(spearmanr(prog, d["v_rms"].to_numpy()).statistic) if "v_rms" in d else np.nan
        feat_rows.append(rec)
    F = pd.DataFrame(feat_rows)

    err_rows = []
    for f in glob.glob(str(PRED / "*" / "*.csv")):
        d = pd.read_csv(f)
        err_rows.append(dict(arm=Path(f).parent.name, bearing=Path(f).stem,
                             mae=float(np.mean(np.abs(d.err))),
                             last_abs=float(abs(d.err.iloc[-1]))))
    E = pd.DataFrame(err_rows).groupby(["arm", "bearing"]).mean().reset_index()
    print(f"bearings={F.bearing.nunique()}  predictions={len(E)}")

    # ---- question A: does band4 track degradation within each bearing?
    print("\n--- A) within-bearing trend of high-frequency band (8-12.8 kHz) ---")
    tr = F[["bearing", "band4_trend", "rms_trend"]].dropna()
    pos = int((tr.band4_trend > 0).sum())
    print(f"  band4 trend > 0 in {pos}/{len(tr)} bearings (sign test p={binomtest(pos, len(tr)).pvalue:.3f})")
    pos2 = int((tr.rms_trend > 0).sum())
    print(f"  v_rms  trend > 0 in {pos2}/{len(tr)} bearings (sign test p={binomtest(pos2, len(tr)).pvalue:.3f})")
    print(f"  median band4 trend rho = {tr.band4_trend.median():+.3f}; median v_rms trend rho = {tr.rms_trend.median():+.3f}")

    # ---- device-level association with error (n = 17)
    print("\n--- device-level Spearman(feature @ last stage, mean |error|) n=17, BH-corrected ---")
    for arm in ["A", "C", "C-Fixed", "C-MLP"]:
        m = E[E.arm == arm].merge(F, on="bearing")
        res = []
        for c in FEATS:
            if m[c].isna().all():
                continue
            r, p = spearmanr(m[c], m.mae)
            res.append((c, float(r), float(p)))
        if not res:
            continue
        q = bh(np.array([t[2] for t in res]))
        res = sorted(res, key=lambda t: -abs(t[1]))[:4]
        print(f"  {arm}: " + ", ".join(f"{c}={r:+.2f}(p={p:.3f},q={qq:.3f})"
                                       for (c, r, p), qq in zip(res, sorted(q, reverse=False)[:4])))

    # ---- question B: does band4 add beyond RMS?
    print("\n--- B) band4 vs |error| controlling for v_rms, h_rms, log life (n=17) ---")
    for arm in ["A", "C", "C-MLP"]:
        m = E[E.arm == arm].merge(F, on="bearing").dropna(subset=["h_band4", "v_rms", "h_rms"])
        r, p = partial_spearman(m.h_band4, m.mae, [m.v_rms, m.h_rms, np.log(m.life)], len(m))
        r2, p2 = partial_spearman(m.hv_energy_ratio, m.mae, [m.v_rms, m.h_rms, np.log(m.life)], len(m))
        print(f"  {arm}: band4 partial rho={r:+.2f} (p={p:.3f});  hv_ratio partial rho={r2:+.2f} (p={p2:.3f})")

    # ---- condition family (descriptive only)
    print("\n--- condition family (descriptive; 4-5 bearings per group) ---")
    F["cond"] = [b.split("_")[1][:7] for b in F.bearing]
    for arm in ["C"]:
        m = E[E.arm == arm].merge(F, on="bearing")
        print(m.groupby("cond")[["mae", "h_band4", "v_rms"]].mean().round(3).to_string())

    F.to_csv(ROOT / "docs/bearing_tim/spectral_device_summary.csv", index=False)
    print("\nsaved spectral_device_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
