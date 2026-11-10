"""Do causal spectral features explain bearing RUL prediction errors?

Join
  * per-snapshot spectral features (data/femto_spec/<bearing>.csv), computed from
    each snapshot only (no future information),
  * per-stage prediction errors of the four arms (experiments/bearing_tim2).

Analyses
  1. device-level: across the 17 bearings, Spearman between a feature measured at
     the last stage and the arm's absolute error at that stage;
  2. stage-controlled: partial Spearman controlling for the life stage;
  3. condition breakdown (Bearing1_x / 2_x / 3_x);
  4. C vs C-MLP: which features track the paired difference.

Caveat printed in the output: 17 bearings is the device-level n; the 4 stages are
repeated measures on the same device and are NOT independent samples.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "data/femto_spec"
PRED = ROOT / "experiments/bearing_tim2"
STAGES = (0.50, 0.70, 0.85, 0.95)
FEATS = ["h_rms", "h_kurt", "h_crest", "h_band0", "h_band1", "h_band2", "h_band3", "h_band4",
         "h_env_energy", "h_env_peak_f", "v_rms", "v_kurt", "hv_energy_ratio", "coh_mean"]


def stage_features() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(SPEC / "*.csv"))):
        name = Path(f).stem
        d = pd.read_csv(f)
        if len(d) < 10:
            continue
        n = len(d)
        for si, frac in enumerate(STAGES):
            idx = min(n - 1, int(round(frac * n)) - 1)
            r = d.iloc[idx]
            rec = dict(bearing=name, stage=si, frac=frac, n=n)
            for c in FEATS:
                rec[c] = float(r[c]) if c in d.columns else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def stage_errors() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(PRED / "*" / "*.csv")):
        d = pd.read_csv(f)
        arm = Path(f).parent.name
        bearing = Path(f).stem
        for si, r in enumerate(d.itertuples()):
            rows.append(dict(arm=arm, bearing=bearing, stage=si, err=float(r.err),
                             abs_err=abs(float(r.err)), true_rul=float(r.true_rul)))
    return pd.DataFrame(rows)


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-residual)."""
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    bx = np.polyfit(rz, rx, 1); by = np.polyfit(rz, ry, 1)
    ex = rx - np.polyval(bx, rz); ey = ry - np.polyval(by, rz)
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 3) / max(1e-9, 1 - r ** 2))
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(t), n - 3)) if n > 3 else float("nan")
    return float(r), float(p)


def main() -> int:
    sf = stage_features()
    se = stage_errors()
    print(f"bearings={sf.bearing.nunique()} feature_rows={len(sf)} pred_rows={len(se)}")
    m = se.merge(sf, on=["bearing", "stage"], how="inner")
    print(f"joined rows={len(m)}\n")

    print("--- device-level Spearman: feature @ last stage vs |error| @ last stage (n=17) ---")
    last = m[m.stage == 3]
    out = []
    for arm in ["A", "C", "C-Fixed", "C-MLP"]:
        sub = last[last.arm == arm]
        for c in FEATS:
            if sub[c].isna().all():
                continue
            r, p = spearmanr(sub[c], sub.abs_err)
            out.append(dict(arm=arm, feat=c, rho=float(r), p=float(p), n=len(sub)))
    o = pd.DataFrame(out)
    for arm in ["A", "C", "C-MLP"]:
        s = o[(o.arm == arm)].sort_values("rho", key=lambda x: -x.abs()).head(5)
        print(f"  {arm}: " + ", ".join(f"{r.feat}={r.rho:+.2f}(p={r.p:.2f})" for r in s.itertuples()))

    print("\n--- stage-controlled partial Spearman (all stages, controlling for stage) ---")
    for arm in ["A", "C", "C-MLP"]:
        sub = m[m.arm == arm].dropna(subset=["h_kurt", "abs_err"])
        res = []
        for c in FEATS:
            s2 = sub.dropna(subset=[c])
            if len(s2) < 12:
                continue
            r, p = partial_spearman(s2[c], s2.abs_err, s2.stage)
            res.append((c, r, p, len(s2)))
        res.sort(key=lambda t: -abs(t[1]))
        print(f"  {arm}: " + ", ".join(f"{c}={r:+.2f}(p={p:.2f})" for c, r, p, _ in res[:5]))

    print("\n--- C vs C-MLP paired difference vs features (last stage, n=17) ---")
    piv = last.pivot_table(index="bearing", columns="arm", values="abs_err")
    if {"C", "C-MLP"} <= set(piv.columns):
        diff = (piv["C"] - piv["C-MLP"]).dropna()
        fe = last[last.arm == "C"].set_index("bearing")
        for c in FEATS:
            if c not in fe.columns or fe[c].isna().all():
                continue
            common = diff.index.intersection(fe.index)
            r, p = spearmanr(fe.loc[common, c], diff.loc[common])
            if abs(r) > 0.3:
                print(f"  {c}: rho={r:+.2f} p={p:.2f} n={len(common)}")

    m.to_csv(ROOT / "docs/bearing_tim/spectral_error_join.csv", index=False)
    sf.to_csv(ROOT / "docs/bearing_tim/spectral_stages.csv", index=False)
    print("\nsaved spectral_error_join.csv / spectral_stages.csv")
    print("NOTE: 17 bearings are the device-level n; stages are repeated measures, not independent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
