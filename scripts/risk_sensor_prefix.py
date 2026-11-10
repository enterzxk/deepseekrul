"""Phase 1c: do RAW SENSOR prefix features predict device-level failure?

The previous run showed that model-side prefix features (half-width, sigma,
predicted-centre statistics over the first k windows) are at chance. This script
adds the raw-sensor features the review asked for, all computed from the prefix
only and without any label:

    * number of operating-condition switches in the prefix,
    * mean absolute sensor change per cycle (trend magnitude),
    * cross-sensor trend dispersion (do sensors agree on the direction?),
    * sensor volatility (mean rolling std),
    * prefix length.

Evaluation is leave-one-fold-out over engines, identical to the earlier scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ida as R  # noqa: E402

TAB = ROOT / "docs/tables/risk_engine_table.csv"
COLS = [c for c in R.COLS] if hasattr(R, "COLS") else None


def prefix_sensor_features(subset: str, k: int) -> pd.DataFrame:
    df = R.load_cmapss_train(ROOT / "data" / "CMAPSS", subset)
    op = ["op_1", "op_2", "op_3"]
    rows = []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        # a window of 50 ends at cutoff c, so the first k windows reach cycle 50+k-1
        end = min(len(g), 49 + k)
        pre = g.iloc[:end]
        vals = pre[R.COLS].to_numpy(np.float64) if COLS is None else pre[COLS].to_numpy(np.float64)
        d = np.diff(vals, axis=0)
        oc = pre[op].to_numpy(np.float64)
        doc = np.abs(np.diff(oc, axis=0)).sum(axis=1)
        switch = float((doc > 1e-6).sum())
        trend = float(np.abs(d).mean())
        # cross-sensor agreement: fraction of sensor pairs whose trend signs agree
        signs = np.sign(vals[-1] - vals[0])
        agree = float(np.abs(signs.mean()))
        vol = float(np.mean([np.std(vals[max(0, i - 10):i + 1], axis=0).mean()
                             for i in range(10, len(vals), 10)])) if len(vals) > 20 else float(np.std(vals, axis=0).mean())
        rows.append(dict(subset=subset, unit=int(unit), sw=switch, trend=trend,
                         agree=agree, vol=vol, plen=len(pre)))
    return pd.DataFrame(rows)


def oof_auc(tab: pd.DataFrame, feats: list[str]) -> dict:
    aucs, preds = [], []
    for f in sorted(tab["fold"].unique()):
        tr, te = tab[tab.fold != f], tab[tab.fold == f]
        if tr["y"].nunique() < 2 or te["y"].nunique() < 2:
            continue
        sc = StandardScaler().fit(tr[feats].to_numpy())
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(tr[feats].to_numpy()), tr["y"])
        p = clf.predict_proba(sc.transform(te[feats].to_numpy()))[:, 1]
        aucs.append(roc_auc_score(te["y"], p))
        preds.append(pd.DataFrame({"subset": te.subset, "y": te.y.to_numpy(), "p": p}))
    pooled = pd.concat(preds, ignore_index=True)
    out = dict(auc=float(roc_auc_score(pooled.y, pooled.p)), folds=[round(a, 3) for a in aucs])
    for s in sorted(pooled.subset.unique()):
        q = pooled[pooled.subset == s]
        out[f"auc_{s}"] = float(roc_auc_score(q.y, q.p)) if q.y.nunique() > 1 else float("nan")
    return out


def main() -> int:
    tab = pd.read_csv(TAB)
    parts = [prefix_sensor_features(s, 20) for s in ["FD001", "FD002", "FD003", "FD004"]]
    sens = pd.concat(parts, ignore_index=True)
    m = tab.merge(sens, on=["subset", "unit"], how="left")
    print(f"merged rows={len(m)} missing_sensor={m.sw.isna().sum()}")

    model_feats = ["hw_k20", "hwcv_k20", "sig_k20", "mu_k20", "muslope_k20", "rul_k20"]
    sens_feats = ["sw", "trend", "agree", "vol"]
    for name, feats in {
        "sensor prefix only": sens_feats,
        "model prefix only": model_feats,
        "model + sensor prefix": model_feats + sens_feats,
    }.items():
        r = oof_auc(m, feats)
        per = " ".join(f"{k[4:]}={v:.3f}" for k, v in r.items() if k.startswith("auc_FD"))
        print(f"  {name:24s} pooled={r['auc']:.3f}  {per}")

    print("\n--- single sensor-feature AUROC ---")
    for c in sens_feats:
        try:
            print(f"  {c:8s} {oof_auc(m, [c])['auc']:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {c}: {e}")
    m.to_csv(ROOT / "docs/tables/risk_sensor_prefix.csv", index=False)
    print("saved risk_sensor_prefix.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
