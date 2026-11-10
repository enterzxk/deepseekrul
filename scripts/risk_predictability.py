"""Phase 1: can device-level interval failure be predicted in advance?

Unit of analysis is the ENGINE, never the window. For every (subset, seed, fold,
unit) we compute

  * the engine's overall coverage over all its windows (the TARGET),
  * features from the FIRST k windows only (observable at prediction time).

Leakage guards
  * the target engine's future RUL / future coverage is never a feature;
  * the split is by ENGINE (leave-one-fold-out over the 5 engine folds), so the
    same physical engine never appears in both train and test;
  * no test label is used to choose a threshold: AUROC is threshold-free and
    the decision threshold, when reported, is fitted on training folds only.

Outputs
  * prevalence of low-coverage engines,
  * AUROC of single features and of a fitted model, per subset and pooled,
  * comparison of feature sets: early-coverage only vs width/sigma vs all,
  * cross-seed stability of the engine ranking.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LOW = 0.80          # "poorly covered engine" threshold on the engine's own coverage
KS = (10, 20, 40)   # early-window cutoffs


def load_cells() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(ROOT / "experiments/punet_matrix/*/f*_s*/preds_cqr.csv")):
        parts = Path(f).parts
        subset = parts[-3]
        tag = parts[-2]                      # f0_s13
        fold = int(tag.split("_")[0][1:])
        seed = int(tag.split("_")[1][1:])
        d = pd.read_csv(f)
        d["subset"], d["fold"], d["seed"] = subset, fold, seed
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def engine_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (subset, seed, fold, unit), g in df.groupby(["subset", "seed", "fold", "unit"]):
        g = g.sort_values("cutoff")
        cov = ((g.y_true >= g.y_lo) & (g.y_true <= g.y_hi)).to_numpy()
        hw = (g.y_hi - g.y_lo).to_numpy()
        rec = dict(subset=subset, seed=seed, fold=fold, unit=unit,
                   n=len(g), cov=cov.mean(), hw_mean=hw.mean(),
                   hw_cv=hw.std() / (hw.mean() + 1e-9),
                   mu_slope=float(np.polyfit(np.arange(len(g)), g.mu_T.to_numpy(), 1)[0]) if len(g) > 2 else 0.0,
                   mu_std=float(g.mu_T.std()),
                   sigma_mean=float(g.sigma_T.mean()),
                   sigma_cv=float(g.sigma_T.std() / (g.sigma_T.mean() + 1e-9)),
                   rul_pred_mean=float((g.mu_T - g.cutoff).mean()),
                   y_true_mean=float(g.y_true.mean()))
        for k in KS:
            kk = min(k, len(g))
            rec[f"cov_k{k}"] = cov[:kk].mean()
            rec[f"hw_k{k}"] = hw[:kk].mean()
            rec[f"hwcv_k{k}"] = hw[:kk].std() / (hw[:kk].mean() + 1e-9)
            rec[f"sig_k{k}"] = float(g.sigma_T.to_numpy()[:kk].mean())
            rec[f"mu_k{k}"] = float(g.mu_T.to_numpy()[:kk].std())
            rec[f"muslope_k{k}"] = float(np.polyfit(np.arange(kk), g.mu_T.to_numpy()[:kk], 1)[0]) if kk > 2 else 0.0
            rec[f"rul_k{k}"] = float((g.mu_T.to_numpy()[:kk] - g.cutoff.to_numpy()[:kk]).mean())
            rec[f"err_k{k}"] = float(np.abs(g.y_true.to_numpy()[:kk] - g.mu_T.to_numpy()[:kk]).mean())
        out.append(rec)
    return pd.DataFrame(out)


def auroc_oof(tab: pd.DataFrame, feats: list[str], label: str = "y") -> dict:
    """Leave-one-fold-out over engines; the same engine never spans train/test."""
    aucs, n_test, preds = [], 0, []
    for f in sorted(tab["fold"].unique()):
        tr = tab[tab.fold != f]
        te = tab[tab.fold == f]
        if tr[label].nunique() < 2 or te[label].nunique() < 1:
            continue
        sc = StandardScaler().fit(tr[feats].to_numpy())
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(tr[feats].to_numpy()), tr[label])
        p = clf.predict_proba(sc.transform(te[feats].to_numpy()))[:, 1]
        preds.append(pd.DataFrame({"fold": f, "unit": te.unit, "subset": te.subset,
                                   "y": te[label].to_numpy(), "p": p}))
        if te[label].nunique() > 1:
            aucs.append(roc_auc_score(te[label], p))
        n_test += len(te)
    pooled = pd.concat(preds, ignore_index=True)
    auc = roc_auc_score(pooled.y, pooled.p) if pooled.y.nunique() > 1 else float("nan")
    return dict(auc_pooled=float(auc), auc_per_fold=[round(a, 3) for a in aucs], n_test=int(n_test))


def main() -> int:
    df = load_cells()
    tab = engine_table(df)
    tab["y"] = (tab["cov"] < LOW).astype(int)
    print(f"engine-cells={len(tab)} unique_engines={tab.groupby(['subset','unit']).ngroups}")
    print(f"prevalence of cov<{LOW}: {tab.y.mean():.3f} (by subset)")
    print(tab.groupby("subset")["y"].mean().round(3).to_string())

    # single-feature AUROC (pooled, out-of-fold via the same engine split)
    single = {}
    for c in ["cov_k10", "cov_k20", "cov_k40", "hw_mean", "hw_cv", "sigma_mean", "sigma_cv",
              "mu_slope", "mu_std", "err_k20", "n"]:
        if c not in tab:
            continue
        r = auroc_oof(tab, [c])
        single[c] = r["auc_pooled"]
    print("\n--- single-feature AUROC (leave-one-fold-out) ---")
    for k, v in sorted(single.items(), key=lambda kv: -abs(kv[1] - 0.5)):
        print(f"  {k:12s} {v:.3f}")

    sets = {
        "early_cov_only": ["cov_k10", "cov_k20"],
        "width_sigma": ["hw_mean", "hw_cv", "sigma_mean", "sigma_cv"],
        "early_all": [c for c in tab.columns if c.endswith(("_k10", "_k20"))],
        "all": ["cov_k20", "hw_mean", "hw_cv", "sigma_mean", "sigma_cv", "mu_slope", "mu_std", "n"],
    }
    print("\n--- multi-feature OOF AUROC ---")
    res = {}
    for name, feats in sets.items():
        feats = [c for c in feats if c in tab.columns]
        r = auroc_oof(tab, feats)
        res[name] = r
        print(f"  {name:16s} auc={r['auc_pooled']:.3f} per_fold={r['auc_per_fold']} n={r['n_test']}")

    # cross-seed stability: rank correlation of engine risk score across seeds
    print("\n--- cross-seed stability (Spearman of engine coverage across seeds) ---")
    piv = tab.pivot_table(index=["subset", "unit"], columns="seed", values="cov")
    print(piv.corr(method="spearman").round(3).to_string())

    out = ROOT / "docs/tables/risk_predictability.json"
    out.write_text(json.dumps(dict(single=single, sets=res, prevalence=float(tab.y.mean()),
                                   n_engine_cells=int(len(tab))), indent=1), encoding="utf-8")
    tab.to_csv(ROOT / "docs/tables/risk_engine_table.csv", index=False)
    print(f"\nsaved {out} and risk_engine_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
