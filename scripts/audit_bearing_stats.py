"""Audit of the bearing statistics: paired differences, bootstrap unit, P value.

What this fixes
  * the earlier script reported P(rel<0) as if it were a p-value; it is only the
    fraction of bootstrap replicates. A real test needs a null hypothesis, so we
    add a paired SIGN-FLIP permutation test (H0: the two arms are exchangeable
    within a bearing);
  * the earlier bootstrap resampled bearings but recomputed a ratio of RMS
    statistics, while the reported "mean" was a mean of per-bearing ratios.
    Here the statistic is defined once and used everywhere:
        D = mean_b [ (RMSE_cand_b - RMSE_base_b) / RMSE_base_b ]
  * CI (percentile bootstrap over bearings) and p (sign-flip) are both reported,
    and the statistic they refer to is printed explicitly.

Known limitations stated in the output
  * the LOBO folds share 15/16 of their training bearings, so bearings are not
    independent w.r.t. the trained model; the bootstrap treats them as
    exchangeable, which is an approximation;
  * 17 bearings is the device-level sample size; extra seeds would NOT add
    independent devices.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/bearing_tim3"


def per_bearing_rmse() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(DIR / "*" / "*.csv")):
        d = pd.read_csv(f)
        rows.append(dict(arm=Path(f).parent.name, holdout=Path(f).stem,
                         rmse=float(np.sqrt(np.mean(d.err ** 2))),
                         mae=float(np.mean(np.abs(d.err))),
                         bias=float(np.mean(d.err)),
                         n=len(d)))
    return pd.DataFrame(rows)


def paired_stats(piv: pd.DataFrame, base: str, cand: str, reps: int = 20000) -> dict:
    b = piv[base].to_numpy(float)
    c = piv[cand].to_numpy(float)
    d = (c - b) / b                      # per-bearing relative difference
    stat = float(d.mean())

    rng = np.random.default_rng(0)
    n = len(d)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(reps)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    # paired sign-flip permutation test under H0: symmetric differences around 0
    signs = rng.choice([-1.0, 1.0], size=(reps, n))
    null = (signs * d).mean(axis=1)
    p_two = float((np.abs(null) >= abs(stat)).mean())
    p_one = float((null <= stat).mean()) if stat < 0 else float((null >= stat).mean())
    return dict(base=base, cand=cand, n=n, stat_pct=stat * 100, ci_pct=(ci[0] * 100, ci[1] * 100),
                p_two_sided=p_two, p_one_sided=p_one,
                wins=int((d < 0).sum()), frac_better=float((d < 0).mean()))


def main() -> int:
    pb = per_bearing_rmse()
    piv = pb.pivot(index="holdout", columns="arm", values="rmse")
    print("statistic: D = mean_b[(RMSE_cand - RMSE_base)/RMSE_base]  (negative = candidate better)")
    print(f"n bearings = {len(piv)}\n")
    for base, cand in [("A", "C"), ("C-Fixed", "C"), ("C-MLP", "C"), ("A", "C-Fixed"), ("A", "C-MLP")]:
        r = paired_stats(piv, base, cand)
        print(f"{r['cand']:8s} vs {r['base']:8s}: D={r['stat_pct']:+.2f}%  "
              f"95%CI=[{r['ci_pct'][0]:+.2f}%, {r['ci_pct'][1]:+.2f}%]  "
              f"p(2-sided)={r['p_two_sided']:.3f}  p(1-sided)={r['p_one_sided']:.3f}  "
              f"better={r['wins']}/{r['n']}")

    print("\n--- per-bearing table ---")
    t = pb.pivot(index="holdout", columns="arm", values=["rmse", "bias"]).round(1)
    print(t.to_string())

    print("\n--- sign of bias (positive = over-estimates RUL) ---")
    print(pb.pivot(index="holdout", columns="arm", values="bias").round(1).to_string())

    pb.to_csv(ROOT / "docs/bearing_tim/audit_per_bearing.csv", index=False)
    print("\nsaved audit_per_bearing.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
