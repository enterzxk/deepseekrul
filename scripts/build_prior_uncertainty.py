"""Build a usable prior uncertainty sigma_T via engine bootstrap.

Why this exists
---------------
src/lifetime_prior/nested_prior.py estimates sigma_T as the standard deviation
across 5 random seeds of HistGradientBoostingRegressor. HGB is deterministic
given the data, so every seed returns bit-identical predictions and sigma_T is
exactly 0 (verified: 7145/7145 rows in FD001, max 2.8e-14). That makes any
"prior-uncertainty-scaled" method untestable.

This script keeps mu_T unchanged and replaces sigma_T with the standard
deviation of predictions from B models fitted on engine-level bootstrap
resamples of the training engines. Training rows get out-of-fold bootstrap
predictions through the existing inner folds, so their sigma is not in-sample.

Zero intrusion: nested_prior.py is imported, never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dataset import (  # noqa: E402
    SELECTED_SENSOR_COLUMNS,
    OperatingConditionScaler,
    load_cmapss_train,
)
from lifetime_prior.nested_prior import build_rows, hgb_fit_predict  # noqa: E402

OUTER_SEED = 20261104


def outer_folds(subset: str, n_outer: int = 5):
    frame = load_cmapss_train(ROOT / "data" / "CMAPSS", subset)
    units = np.array(sorted(frame["unit"].unique().tolist()), dtype=np.int64)
    kf = KFold(n_splits=n_outer, shuffle=True, random_state=OUTER_SEED)
    return list(kf.split(units)), frame, units


def boot_sigma(rows_tr, rows_pred, units_tr, b: int, seed: int, group_units=None):
    """Std of B bootstrap predictions; bootstrap resamples engines, not rows."""
    rng = np.random.default_rng(seed)
    preds = np.zeros((b, len(rows_pred)), dtype=np.float64)
    for j in range(b):
        pick = rng.choice(units_tr, size=len(units_tr), replace=True)
        picked = set(int(u) for u in pick)
        sub = [r for r in rows_tr if r["unit"] in picked]
        preds[j] = np.asarray(hgb_fit_predict(sub, rows_pred, seed + j), dtype=np.float64)
    return preds.std(axis=0), preds.mean(axis=0)


def run(subset: str, out_dir: Path, b: int, inner_folds: int) -> pd.DataFrame:
    folds, frame, units = outer_folds(subset)
    cols = list(SELECTED_SENSOR_COLUMNS)
    cutoffs = list(range(50, 201, 10))
    n_cond = 6 if subset in ("FD002", "FD004") else 1
    records = []

    for fold, (ii, iva) in enumerate(folds):
        tr_units = [int(units[j]) for j in ii]
        va_units = [int(units[j]) for j in iva]
        assert not (set(tr_units) & set(va_units)), "engine leak in outer split"
        tr_df = frame[frame["unit"].isin(tr_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)
        va_df = frame[frame["unit"].isin(va_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)
        scaler = OperatingConditionScaler(cols, n_clusters=n_cond, random_state=OUTER_SEED + fold)
        scaler.fit(tr_df)
        rows_tr = build_rows(tr_df, scaler, cutoffs)
        rows_va = build_rows(va_df, scaler, cutoffs)

        # ---- outer-val sigma: bootstrap over outer-train engines ----
        sig_va, mu_va = boot_sigma(rows_tr, rows_va, tr_units, b, seed=1000 + fold)
        for row, s, m in zip(rows_va, sig_va, mu_va):
            records.append(dict(subset=subset, outer_fold=fold, role="val", unit=int(row["unit"]),
                                c=int(row["c"]), mu_T=float(m), sigma_T=float(s),
                                true_T=int(row["true"])))

        # ---- outer-train sigma: OOF through inner folds ----
        inner = KFold(n_splits=inner_folds, shuffle=True, random_state=OUTER_SEED + 100 + fold)
        tr_sorted = sorted(tr_units)
        sig_tr = {}
        mu_tr = {}
        for k, (ki, kv) in enumerate(inner.split(tr_sorted)):
            fit_units = [tr_sorted[j] for j in ki]
            pred_units = set(tr_sorted[j] for j in kv)
            r_fit = [r for r in rows_tr if r["unit"] in set(fit_units)]
            r_pred = [r for r in rows_tr if r["unit"] in pred_units]
            if not r_pred:
                continue
            s, m = boot_sigma(r_fit, r_pred, fit_units, b, seed=5000 + fold * 10 + k)
            for row, sv, mv in zip(r_pred, s, m):
                sig_tr[(row["unit"], row["c"])] = float(sv)
                mu_tr[(row["unit"], row["c"])] = float(mv)
        for row in rows_tr:
            key = (row["unit"], row["c"])
            if key not in sig_tr:
                continue
            records.append(dict(subset=subset, outer_fold=fold, role="train", unit=int(row["unit"]),
                                c=int(row["c"]), mu_T=mu_tr[key], sigma_T=sig_tr[key],
                                true_T=int(row["true"])))
        print(f"[{subset} f{fold}] val rows={len(rows_va)} sigma_med={np.median(sig_va):.3f}", flush=True)

    df = pd.DataFrame.from_records(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{subset}_prior_cutoffs.csv", index=False)
    (out_dir / f"{subset}_prior_meta.json").write_text(
        json.dumps({"source": "engine-bootstrap", "bootstrap": b, "inner_folds": inner_folds,
                    "folds": {str(f): {"median_T_train": float(np.median(df[(df.outer_fold == f) & (df.role == "train")].mu_T))}
                              for f in range(len(folds))}}, indent=1)
    )
    print(f"[{subset}] wrote {len(df)} rows, sigma mean={df.sigma_T.mean():.3f}", flush=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["FD001"])
    ap.add_argument("--out", default="experiments/ida/prior_boot")
    ap.add_argument("--bootstrap", type=int, default=20)
    ap.add_argument("--inner-folds", type=int, default=4)
    args = ap.parse_args()
    for s in args.subsets:
        run(s.upper(), ROOT / args.out, args.bootstrap, args.inner_folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
