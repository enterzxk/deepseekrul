"""E0-E3 analysis: does high-frequency information add value beyond RMS?

Device-level paired statistics (n=17 bearings), sign-flip permutation p, and a
percentile bootstrap CI of the same statistic. Also reports the first causal
detection time of the high-frequency band (threshold from the early-life baseline
of that bearing only) and per-condition descriptive tables.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments/bearing_tim4"
SPEC = ROOT / "data/femto_spec"
ARMS = ["none", "rms", "hf", "bands"]
LABEL = {"none": "E0", "rms": "E1", "hf": "E2", "bands": "E3"}


def per_bearing() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(E / "*" / "*.csv")):
        d = pd.read_csv(f)
        rows.append(dict(arm=Path(f).parent.name, bearing=Path(f).stem,
                         rmse=float(np.sqrt(np.mean(d.err ** 2))),
                         mae=float(np.mean(np.abs(d.err))),
                         bias=float(np.mean(d.err))))
    return pd.DataFrame(rows)


def paired(piv: pd.DataFrame, base: str, cand: str, reps: int = 20000) -> dict:
    b, c = piv[base].to_numpy(float), piv[cand].to_numpy(float)
    d = (c - b) / b
    stat = float(d.mean())
    rng = np.random.default_rng(0)
    n = len(d)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(reps)])
    ci = (float(np.percentile(boot, 2.5)) * 100, float(np.percentile(boot, 97.5)) * 100)
    signs = rng.choice([-1.0, 1.0], size=(reps, n))
    null = (signs * d).mean(axis=1)
    return dict(stat=stat * 100, ci=ci, p=float((np.abs(null) >= abs(stat)).mean()),
                wins=int((d < 0).sum()), n=n)


def detection_time() -> pd.DataFrame:
    """First life fraction at which the high-frequency band exceeds its early baseline.

    Threshold = mean + 2*std of h_band4 over the first 20% of that bearing's life
    (causal: only past data). Reported fraction uses the full life for ANALYSIS
    only, never as an input.
    """
    rows = []
    for f in sorted(glob.glob(str(SPEC / "*.csv"))):
        d = pd.read_csv(f)
        n = len(d)
        if n < 30 or "h_band4" not in d:
            continue
        x = d["h_band4"].to_numpy()
        base = x[:max(5, int(0.2 * n))]
        thr = base.mean() + 2 * base.std()
        above = np.where(x > thr)[0]
        first = int(above[0]) / (n - 1) if len(above) else np.nan
        rows.append(dict(bearing=Path(f).stem, first_frac=first,
                         base_mean=float(base.mean()), base_std=float(base.std()),
                         final=float(x[-1]), n=n))
    return pd.DataFrame(rows)


def main() -> int:
    pb = per_bearing()
    piv = pb.pivot(index="bearing", columns="arm", values="rmse")
    print("--- per-bearing RMSE (E0 ds=8 / E1 +RMS / E2 +HF-abs / E3 +bands) ---")
    print(piv.rename(columns=LABEL).round(1).to_string())
    print("\n--- macro mean ---")
    print(piv.rename(columns=LABEL).mean().round(1).to_string())

    print("\n--- device-level paired (n=17, sign-flip p, bootstrap CI) ---")
    for base, cand in [("none", "rms"), ("none", "hf"), ("none", "bands"), ("rms", "hf"), ("rms", "bands")]:
        if base in piv and cand in piv:
            r = paired(piv, base, cand)
            print(f"  {LABEL[cand]} vs {LABEL[base]}: D={r['stat']:+.2f}% "
                  f"95%CI=[{r['ci'][0]:+.2f}%, {r['ci'][1]:+.2f}%] p={r['p']:.3f} wins={r['wins']}/{r['n']}")

    print("\n--- by condition family (descriptive; 4-5 bearings/group) ---")
    t = piv.copy()
    t["cond"] = [b.split("_")[1][:7] for b in t.index]
    print(t.groupby("cond").mean().rename(columns=LABEL).round(1).to_string())

    dt = detection_time()
    print("\n--- first causal detection of the 8-12.8 kHz band (fraction of life) ---")
    print(f"  detected in {dt.first_frac.notna().sum()}/{len(dt)} bearings; "
          f"median={dt.first_frac.median():.2f} IQR=({dt.first_frac.quantile(.25):.2f}, {dt.first_frac.quantile(.75):.2f})")
    print(dt.round(2).to_string(index=False))

    pb.to_csv(ROOT / "docs/bearing_tim/e_per_bearing.csv", index=False)
    dt.to_csv(ROOT / "docs/bearing_tim/hf_detection_time.csv", index=False)
    print("\nsaved e_per_bearing.csv / hf_detection_time.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
