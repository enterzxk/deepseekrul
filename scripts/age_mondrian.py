"""Does conditioning the conformal quantile on OBSERVED AGE improve device-level tail coverage?

Cheap, zero-training test on the frozen predictions. Score = |y - mu_T| (absolute
residual of the frozen centre). Calibration is leave-one-fold-out over ENGINES:
the quantile is fitted on the other four folds' windows and applied to the held-out
fold's engines. Two variants:

  global   one q for all windows
  age      one q per age bucket (Mondrian on the window's own cutoff quantile)

The question is whether the age-conditional q reduces the lower tail of the
per-engine coverage distribution without inflating width much.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.10
BUCKETS = 4


def load() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(ROOT / "experiments/punet_matrix/*/f*_s*/preds_cqr.csv")):
        p = Path(f).parts
        d = pd.read_csv(f)
        d["subset"], d["fold"], d["seed"] = p[-3], int(p[-2].split("_")[0][1:]), int(p[-2].split("_")[1][1:])
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def qhat(s: np.ndarray, alpha: float) -> float:
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[min(k, n) - 1])


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (subset, seed), g in df.groupby(["subset", "seed"]):
        g = g.copy()
        g["score"] = (g.y_true - g.mu_T).abs()
        g["age"] = g.groupby("fold")["cutoff"].transform(lambda c: pd.qcut(c, BUCKETS, labels=False, duplicates="drop"))
        for f in sorted(g.fold.unique()):
            tr = g[g.fold != f]
            te = g[g.fold == f].copy()
            q_g = qhat(tr.score.to_numpy(), ALPHA)
            te["lo_g"], te["hi_g"] = te.mu_T - q_g, te.mu_T + q_g
            # age-conditional: fit a q per bucket on training folds
            qs = {}
            for b, gb in tr.groupby("age"):
                qs[int(b)] = qhat(gb.score.to_numpy(), ALPHA)
            qs = pd.Series(qs)
            te["q_age"] = te.age.map(qs).fillna(q_g)
            te["lo_a"], te["hi_a"] = te.mu_T - te.q_age, te.mu_T + te.q_age
            for arm, lo, hi in [("global", te.lo_g, te.hi_g), ("age", te.lo_a, te.hi_a)]:
                cov = ((te.y_true >= lo) & (te.y_true <= hi)).astype(float)
                hw = (hi - lo)
                per = pd.DataFrame({"unit": te.unit, "cov": cov, "hw": hw}).groupby("unit").mean()
                out.append(dict(subset=subset, seed=seed, fold=f, arm=arm,
                                picp=float(cov.mean()), width=float(hw.mean()),
                                worst=float(per["cov"].min()), p05=float(per["cov"].quantile(0.05)),
                                n_eng=int(len(per))))
    return pd.DataFrame(out)


def main() -> int:
    df = load()
    res = evaluate(df)
    print(f"cells={res.groupby(['subset','seed','fold']).ngroups}  windows={len(df)}")
    print("\n--- mean over cells ---")
    print(res.groupby("arm")[["picp", "width", "worst", "p05"]].mean().round(4).to_string())
    print("\n--- by subset ---")
    print(res.groupby(["subset", "arm"])[["picp", "width", "worst", "p05"]].mean().round(4).to_string())
    print("\n--- paired per cell (age - global) ---")
    piv = res.pivot_table(index=["subset", "seed", "fold"], columns="arm",
                          values=["picp", "width", "worst", "p05"])
    for m in ["picp", "width", "worst", "p05"]:
        d = (piv[(m, "age")] - piv[(m, "global")]).dropna()
        print(f"  {m:6s} mean={d.mean():+.4f} median={d.median():+.4f} "
              f"better={(d>0).sum() if m!='width' else (d<0).sum()}/{len(d)}")
    res.to_csv(ROOT / "docs/tables/age_mondrian.csv", index=False)
    print("\nsaved age_mondrian.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
