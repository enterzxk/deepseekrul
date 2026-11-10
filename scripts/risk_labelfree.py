"""Phase 1b: label-free predictability of device-level interval failure.

In deployment the true RUL of a running engine is unknown, so any feature built
from realised coverage (cov_k*, err_k*, n) is NOT available at prediction time.
This script therefore evaluates only features computable from the frozen model's
own outputs and the observed sensor history:

    half-width mean / CV, sigma_T mean / CV, predicted-centre slope / std,
    predicted RUL level.

It also reports the realised-coverage variant for reference, so the gap between
"what we could do with labels" and "what is actually deployable" is explicit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "docs/tables/risk_engine_table.csv"

NO_LABEL = ["hw_mean", "hw_cv", "sigma_mean", "sigma_cv", "mu_slope", "mu_std", "rul_pred_mean"]
PREFIX10 = ["hw_k10", "hwcv_k10", "sig_k10", "mu_k10", "muslope_k10", "rul_k10"]
PREFIX20 = ["hw_k20", "hwcv_k20", "sig_k20", "mu_k20", "muslope_k20", "rul_k20"]
PREFIX40 = ["hw_k40", "hwcv_k40", "sig_k40", "mu_k40", "muslope_k40", "rul_k40"]
WITH_COV = ["cov_k10", "cov_k20"]
LEAKY = ["n"]


def oof_auc(tab: pd.DataFrame, feats: list[str]) -> dict:
    aucs, preds = [], []
    for f in sorted(tab["fold"].unique()):
        tr, te = tab[tab.fold != f], tab[tab.fold == f]
        if tr["y"].nunique() < 2:
            continue
        sc = StandardScaler().fit(tr[feats].to_numpy())
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(tr[feats].to_numpy()), tr["y"])
        p = clf.predict_proba(sc.transform(te[feats].to_numpy()))[:, 1]
        preds.append(pd.DataFrame({"subset": te.subset, "y": te.y.to_numpy(), "p": p}))
        if te["y"].nunique() > 1:
            aucs.append(roc_auc_score(te["y"], p))
    pooled = pd.concat(preds, ignore_index=True)
    out = dict(auc=float(roc_auc_score(pooled.y, pooled.p)), folds=[round(a, 3) for a in aucs])
    for s in sorted(pooled.subset.unique()):
        q = pooled[pooled.subset == s]
        out[f"auc_{s}"] = float(roc_auc_score(q.y, q.p)) if q.y.nunique() > 1 else float("nan")
    return out


def main() -> int:
    tab = pd.read_csv(TAB)
    print(f"engine-cells={len(tab)}  prevalence={tab.y.mean():.3f}")

    sets = {
        "prefix-10 label-free": PREFIX10,
        "prefix-20 label-free": PREFIX20,
        "prefix-40 label-free": PREFIX40,
        "label_free (deployable, full trajectory)": NO_LABEL,
        "label_free + early coverage": NO_LABEL + WITH_COV,
        "early coverage only": WITH_COV,
        "width+sigma only (heteroscedastic scale)": ["hw_mean", "hw_cv", "sigma_mean", "sigma_cv"],
        "label_free + engine length (LEAKY)": NO_LABEL + LEAKY,
    }
    res = {}
    print("\n--- out-of-fold AUROC (engine-level split) ---")
    for name, feats in sets.items():
        feats = [c for c in feats if c in tab.columns]
        r = oof_auc(tab, feats)
        res[name] = r
        per = " ".join(f"{k[4:]}={v:.3f}" for k, v in r.items() if k.startswith("auc_FD"))
        print(f"  {name:42s} pooled={r['auc']:.3f}  {per}")

    # permutation baseline for the label-free set
    rng = np.random.default_rng(0)
    base = oof_auc(tab, NO_LABEL)["auc"]
    null = []
    for _ in range(50):
        t = tab.copy()
        t["y"] = rng.permutation(t["y"].to_numpy())
        null.append(oof_auc(t, NO_LABEL)["auc"])
    null = np.array(null)
    print(f"\nlabel-free AUROC={base:.3f}; permutation null mean={null.mean():.3f} "
          f"p95={np.percentile(null,95):.3f}; empirical p={float((null>=base).mean()):.3f}")

    (ROOT / "docs/tables/risk_predictability_labelfree.json").write_text(
        json.dumps(dict(sets=res, null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                        empirical_p=float((null >= base).mean())), indent=1), encoding="utf-8")
    print("saved risk_predictability_labelfree.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
