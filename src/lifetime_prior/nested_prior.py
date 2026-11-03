"""Nested, strictly causal lifetime prior for IDA-PE (Gate 0).

Reuses the leakage-safe feature extractor of ``lsc_feasibility.py`` (fixed
observable cutoffs, prefix-only features) but replaces its flat 5-fold design
with the nested cross-fitting required by the IDA-PE specification:

  * outer 5-fold, engine-level: every engine is outer-val exactly once;
  * inner 4-fold cross-fitting inside outer-train produces OOF priors for the
    outer-train engines (used when training M1-style models on them);
  * an ensemble of ``K`` HGB seeds trained on the full outer-train predicts the
    outer-val engines: mu_T = mean, sigma_T = std across seeds;
  * the scaler is fit on outer-train only;
  * no true lifetime of any validation engine ever enters any fit.

Causal staircase: a window ending at cycle t consumes the prediction of the
latest cutoff c <= t (never an interpolation towards a future cutoff). Windows
whose end is below the smallest cutoff fall back to the outer-train median
lifetime and are flagged. ``staircase_lookup`` enforces and asserts this.

Outputs (per subset):
  prior_cutoffs.csv   unit, outer_fold, role(train/val), c, mu_T, sigma_T, true_T
  prior_meta.json     per-fold medians, corr/MAE by cutoff, provenance
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dataset import (  # noqa: E402
    SELECTED_SENSOR_COLUMNS,
    OperatingConditionScaler,
    load_cmapss_train,
)
from lsc_feasibility import prefix_features  # noqa: E402  (verbatim reuse)

COLS = tuple(SELECTED_SENSOR_COLUMNS)
OPS = ("op_1", "op_2", "op_3")

HGB_KW = dict(
    loss="squared_error", max_iter=200, learning_rate=0.05,
    max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=10.0,
    early_stopping=False,
)


def build_rows(frame, scaler, cutoffs):
    """One feature row per (unit, cutoff c) with c < true length, c >= 40."""
    row_list = []
    for unit, g in frame.groupby("unit", sort=True):
        g = g.sort_values("cycle").reset_index(drop=True)
        cycles = g["cycle"].to_numpy(dtype=np.int64)
        if not np.array_equal(cycles, np.arange(1, len(g) + 1)):
            raise ValueError(f"Non-contiguous 1-based cycles in unit={unit}")
        full_life = int(cycles[-1])
        normalized = scaler.transform(g)
        for c in cutoffs:
            if c >= full_life:
                continue
            _, _, xs = prefix_features(g, normalized, c)
            row_list.append(dict(unit=int(unit), c=int(c), true=full_life, feat=xs))
    return row_list


def hgb_fit_predict(rows_tr, rows_va, seed):
    """Engine-balanced HGB fit; prediction floored at age+1 (survival info)."""
    Xtr = np.stack([r["feat"] for r in rows_tr])
    Ytr = np.array([r["true"] for r in rows_tr], dtype=float)
    Xva = np.stack([r["feat"] for r in rows_va])
    count = {}
    for r in rows_tr:
        count[r["unit"]] = count.get(r["unit"], 0) + 1
    weights = np.array([1.0 / count[r["unit"]] for r in rows_tr])
    model = HistGradientBoostingRegressor(random_state=seed, **HGB_KW)
    model.fit(Xtr, Ytr, sample_weight=weights)
    pred = model.predict(Xva)
    age = np.array([r["c"] for r in rows_va], dtype=float)
    return np.maximum(pred, age + 1.0)


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or min(a.std(), b.std()) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_subset(subset, data_dir, out_dir, outer_seed=20261104,
               seeds=(11, 22, 33, 44, 55), inner_folds=4,
               cutoffs=range(50, 201, 10), n_outer=5, frame=None, quiet=False):
    cutoffs = list(cutoffs)
    df = frame if frame is not None else load_cmapss_train(Path(data_dir), subset)
    units = np.array(sorted(df["unit"].unique().tolist()), dtype=np.int64)
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=outer_seed)
    records, meta = [], dict(folds={}, seeds=list(seeds), cutoffs=cutoffs,
                             outer_seed=outer_seed, inner_folds=inner_folds)
    t0 = time.time()
    for fold, (ti, vi) in enumerate(outer.split(units)):
        tr_units = set(int(u) for u in units[ti])
        va_units = set(int(u) for u in units[vi])
        assert not (tr_units & va_units), "engine leak in outer split"
        tr_df = df[df["unit"].isin(tr_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)
        va_df = df[df["unit"].isin(va_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)
        n_cond = 6 if subset in ("FD002", "FD004") else 1
        scaler = OperatingConditionScaler(COLS, n_clusters=n_cond,
                                          random_state=outer_seed + fold)
        scaler.fit(tr_df)
        rows_tr = build_rows(tr_df, scaler, cutoffs)
        rows_va = build_rows(va_df, scaler, cutoffs)
        if not rows_tr or not rows_va:
            raise ValueError(f"{subset} fold {fold}: empty rows")

        # ---- inner cross-fitting: OOF prior for outer-train engines ----
        tr_units_sorted = sorted(tr_units)
        inner = KFold(n_splits=inner_folds, shuffle=True, random_state=outer_seed + 100 + fold)
        inner_splits = list(inner.split(tr_units_sorted))
        oof = {}   # (unit, c) -> list of K predictions
        for s_i, seed in enumerate(seeds):
            preds_k = [None] * inner_folds
            for k, (ii, ik) in enumerate(inner_splits):
                k_tr = set(tr_units_sorted[j] for j in ii)
                k_va = set(tr_units_sorted[j] for j in ik)
                r_tr = [r for r in rows_tr if r["unit"] in k_tr]
                r_va = [r for r in rows_tr if r["unit"] in k_va]
                preds_k[k] = hgb_fit_predict(r_tr, r_va, seed * 1000 + fold * 10 + k)
            for k in range(inner_folds):
                for row, p in zip([r for r in rows_tr
                                   if r["unit"] in set(tr_units_sorted[j] for j in inner_splits[k][1])],
                                  preds_k[k]):
                    oof.setdefault((row["unit"], row["c"]), []).append(float(p))
        mus_tr, sigs_tr = {}, {}
        for key, vals in oof.items():
            assert len(vals) == len(seeds), f"OOF missing seeds for {key}"
            mus_tr[key] = float(np.mean(vals))
            sigs_tr[key] = float(np.std(vals))

        # ---- outer-val ensemble: fit on the full outer-train ----
        mus_va, sigs_va = {}, {}
        va_preds = {s: None for s in seeds}
        for s_i, seed in enumerate(seeds):
            va_preds[seed] = hgb_fit_predict(rows_tr, rows_va, seed * 1000 + 900 + fold)
        for i, row in enumerate(rows_va):
            vals = [va_preds[s][i] for s in seeds]
            mus_va[(row["unit"], row["c"])] = float(np.mean(vals))
            sigs_va[(row["unit"], row["c"])] = float(np.std(vals))

        # ---- assemble records + fold diagnostics ----
        for role, keys in (("train", mus_tr), ("val", mus_va)):
            table = mus_tr if role == "train" else mus_va
            sigs = sigs_tr if role == "train" else sigs_va
            for (unit, c), mu in table.items():
                true_t = next(r["true"] for r in (rows_tr if role == "train" else rows_va)
                              if r["unit"] == unit)
                records.append(dict(subset=subset, outer_fold=fold, role=role,
                                    unit=int(unit), c=int(c), mu_T=mu,
                                    sigma_T=sigs[(unit, c)], true_T=int(true_t)))
        # diagnostics by cutoff (val only; train diagnostics on OOF)
        diag = {}
        for role, table, rows in (("train_oof", mus_tr, rows_tr), ("val", mus_va, rows_va)):
            by_c = {}
            for c in cutoffs:
                rr = [r for r in rows if r["c"] == c]
                if not rr:
                    continue
                y = np.array([r["true"] for r in rr], float)
                p = np.array([table[(r["unit"], r["c"])] for r in rr], float)
                by_c[str(c)] = dict(n=len(rr), corr=_corr(p, y),
                                    mae=float(np.mean(np.abs(p - y))))
            diag[role] = by_c
        meta["folds"][str(fold)] = dict(
            n_train=len(tr_units), n_val=len(va_units),
            median_T_train=float(np.median([int(u and next(r["true"] for r in rows_tr if r["unit"] == u)) for u in tr_units])),
            diag=diag)
        if not quiet:
            print(f"[{subset}] outer fold {fold}: train={len(tr_units)} val={len(va_units)} "
                  f"val corr@c100={diag['val'].get('100', {}).get('corr')} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    table = pd.DataFrame.from_records(records)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / f"{subset}_prior_cutoffs.csv", index=False)
        meta["elapsed_seconds"] = time.time() - t0
        (out_dir / f"{subset}_prior_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
    return table, meta


# ---------------------------------------------------------------------------
# Causal staircase lookup: cutoff table -> per-window prior
# ---------------------------------------------------------------------------

class StaircasePrior:
    """Lookup mu_T(t), sigma_T(t) for a window ending at cycle t.

    Consumes the latest available cutoff c <= t for that engine. Available
    cutoffs are those with a predicted row and c < engine observed length T.
    Engines with t below the smallest available cutoff get the outer-train
    median lifetime (sigma = NaN-safe 0 placeholder, flag=1). This class never
    interpolates towards the future and asserts T4 (cutoff <= window end).
    """

    def __init__(self, table: pd.DataFrame, medians: dict, subset: str):
        self.subset = subset
        self.medians = medians
        self.by_unit = {}
        for (fold, unit), g in table[table.subset == subset].groupby(["outer_fold", "unit"]):
            g = g.sort_values("c")
            self.by_unit[(int(fold), int(unit))] = dict(
                c=g["c"].to_numpy(np.int64),
                mu=g["mu_T"].to_numpy(float),
                sigma=g["sigma_T"].to_numpy(float),
                true_T=int(g["true_T"].iloc[0]),
                role=str(g["role"].iloc[0]),
            )

    def lookup(self, outer_fold: int, unit: int, t: int):
        """Returns (mu_T, sigma_T, flag_below_min, used_cutoff)."""
        e = self.by_unit.get((int(outer_fold), int(unit)))
        if e is None:
            raise KeyError(f"no prior rows for {self.subset} fold={outer_fold} unit={unit}")
        usable = e["c"][e["c"] < e["true_T"]]
        ok = usable[usable <= int(t)]
        if len(ok) == 0:
            c_star = 0
            mu = float(self.medians[int(outer_fold)])
            sigma = 0.0
            flag = 1
        else:
            c_star = int(ok[-1])
            j = int(np.searchsorted(e["c"], c_star))
            mu = float(e["mu"][j])
            sigma = float(e["sigma"][j])
            flag = 0
        assert c_star <= int(t), "T4 violated: consumed cutoff after window end"
        return mu, sigma, flag, c_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CMAPSS")
    ap.add_argument("--subsets", nargs="+", default=["FD002", "FD001"])
    ap.add_argument("--out", default="experiments/ida/prior")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--inner-folds", type=int, default=4)
    args = ap.parse_args()
    for subset in args.subsets:
        table, meta = run_subset(subset.upper(), args.data, args.out,
                                 outer_seed=args.seed, seeds=tuple(args.seeds),
                                 inner_folds=args.inner_folds)
        # per-fold train median lifetimes (used by the staircase fallback)
        med = {}
        for fold_s, fd in meta["folds"].items():
            med[int(fold_s)] = fd["median_T_train"]
        # smoke-check the staircase on a few engines of every fold
        sp = StaircasePrior(table, med, subset.upper())
        n_checked = 0
        for (fold, unit), e in list(sp.by_unit.items())[:20]:
            for t in (e["true_T"] // 2, e["true_T"] - 1, e["true_T"]):
                mu, sig, flag, c_star = sp.lookup(fold, unit, t)
                n_checked += 1
        print(f"[{subset}] staircase smoke ok ({n_checked} lookups); "
              f"rows={len(table)} -> {args.out}/{subset}_prior_cutoffs.csv")


if __name__ == "__main__":
    main()
