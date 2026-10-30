"""ORDER_03 aggregation: per-seed, pooled, macro and engine-clustered paired analysis.

Reads only artifacts already produced under experiments/order03. Never trains,
never touches official test data. All pairing keys are explicit:
(dataset, unit_id, cutoff, model_seed).

Usage (on lab-server):
  /home/yndx/miniconda3/envs/a40_env/bin/python results/dsh/analysis/aggregate_order03.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = "/home/yndx/zxk/bearing-rul-pi"
OUT = os.path.join(ROOT, "results", "dsh", "order03")
REPS = 20000
RNG_SEED = 20261022


def nasa_score(err: np.ndarray) -> float:
    s = np.where(err > 0, np.exp(err / 13.0) - 1.0, np.exp(-err / 10.0) - 1.0)
    return float(s.mean())


def basic(err: np.ndarray) -> dict:
    err = np.asarray(err, dtype=np.float64)
    return dict(
        n=int(err.size),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        bias=float(err.mean()),
        nasa=nasa_score(err),
        late_fraction=float((err > 0).mean()),
        severe_late_gt10=float((err > 10).mean()),
        severe_late_count=int((err > 10).sum()),
    )


def cell_metrics(tag: str, subset: str, cell: str, seed: int) -> dict:
    d = os.path.join(ROOT, "experiments", "order03", tag, subset, cell)
    out = dict(tag=tag, subset=subset, cell=cell, seed=seed)
    mpath = os.path.join(d, "metrics.json")
    if os.path.exists(mpath):
        m = json.load(open(mpath))
        out.update(
            variant=m.get("variant"), params=m.get("params"),
            best_epoch=m.get("best_epoch"), epochs_ran=m.get("epochs_ran"),
            elapsed=m.get("elapsed_seconds"),
            best_val_rmse=m.get("best_val_rmse"),
            weights=m.get("weights"),
        )
    for kind, fname, key in [
        ("matched", "matched_cutoff_predictions.csv", ("unit_id",)),
        ("protocol_c", "protocol_c_predictions.csv", ("unit_id", "abs_cut")),
        ("protocol_b", "pseudo_test_predictions.csv", ("unit_id", "cutoff")),
    ]:
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df["seed"] = seed
        df["tag"] = tag
        df["subset"] = subset
        b = basic(df["signed_error"].to_numpy())
        out[kind] = b
        if kind in ("protocol_c", "protocol_b"):
            ckey = "abs_cut" if kind == "protocol_c" else "cutoff"
            per = df.groupby(ckey)["signed_error"].apply(lambda s: float(np.sqrt(np.mean(s ** 2))))
            out[kind]["macro_rmse"] = float(per.mean())
            out[kind]["per_cutoff_rmse"] = {str(k): float(v) for k, v in per.items()}
    return out


def load_frames(tags, subset):
    frames = {}
    for tag in tags:
        fs = sorted(glob.glob(os.path.join(ROOT, "experiments", "order03", tag, subset, "*", "matched_cutoff_predictions.csv")))
        for p in fs:
            cell = os.path.basename(os.path.dirname(p))
            seed = int(cell.rsplit("_s", 1)[1])
            df = pd.read_csv(p)
            df["tag"] = tag
            df["seed"] = seed
            df["dataset"] = subset
            frames.setdefault("matched", []).append(df)
        fs = sorted(glob.glob(os.path.join(ROOT, "experiments", "order03", tag, subset, "*", "protocol_c_predictions.csv")))
        for p in fs:
            cell = os.path.basename(os.path.dirname(p))
            seed = int(cell.rsplit("_s", 1)[1])
            df = pd.read_csv(p)
            df["tag"] = tag
            df["seed"] = seed
            df["dataset"] = subset
            frames.setdefault("protocol_c", []).append(df)
        fs = sorted(glob.glob(os.path.join(ROOT, "experiments", "order03", tag, subset, "*", "pseudo_test_predictions.csv")))
        for p in fs:
            cell = os.path.basename(os.path.dirname(p))
            seed = int(cell.rsplit("_s", 1)[1])
            df = pd.read_csv(p)
            df["tag"] = tag
            df["seed"] = seed
            df["dataset"] = subset
            frames.setdefault("protocol_b", []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}


def pair_key(kind: str) -> list[str]:
    if kind == "matched":
        return ["dataset", "unit_id", "seed"]
    if kind == "protocol_c":
        return ["dataset", "unit_id", "abs_cut", "seed"]
    return ["dataset", "unit_id", "cutoff", "seed"]


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, kind: str) -> dict:
    keys = pair_key(kind)
    cols = keys + ["signed_error"]
    m = a[cols].merge(b[cols], on=keys, suffixes=("_a", "_b"), how="inner")
    units = sorted(m["unit_id"].unique())
    # per-engine sufficient statistics keep all seeds/cutoffs of an engine together
    sse_a, cnt_a, sse_b, cnt_b = {}, {}, {}, {}
    for u, g in m.groupby("unit_id"):
        sse_a[u] = float((g.signed_error_a ** 2).sum())
        cnt_a[u] = int(len(g))
        sse_b[u] = float((g.signed_error_b ** 2).sum())
        cnt_b[u] = int(len(g))
    sa = np.array([sse_a[u] for u in units]); ca = np.array([cnt_a[u] for u in units], float)
    sb = np.array([sse_b[u] for u in units]); cb = np.array([cnt_b[u] for u in units], float)

    def rmse(s, c):
        return math.sqrt(float(s.sum()) / float(c.sum()))

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(units), size=(REPS, len(units)))
    diffs = np.empty(REPS)
    for i in range(REPS):
        j = idx[i]
        diffs[i] = rmse(sa[j], ca[j]) - rmse(sb[j], cb[j])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(
        kind=kind, n_engines=len(units), n_pairs=int(len(m)),
        rmse_a=rmse(sa, ca), rmse_b=rmse(sb, cb),
        rmse_diff=rmse(sa, ca) - rmse(sb, cb),
        ci_low=float(lo), ci_high=float(hi),
        excludes_zero=bool(lo > 0 or hi < 0),
        prob_a_better=float((diffs < 0).mean()),
        clustering="engines resampled; all seeds and cutoffs of an engine stay together",
    )


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    tags = ["M0_raw_tf", "M2_pop", "M3_cond", "M4_racdr"]
    subset = "FD002"
    cells = []
    for tag in tags:
        for cell in sorted(os.listdir(os.path.join(ROOT, "experiments", "order03", tag, subset))):
            seed = int(cell.rsplit("_s", 1)[1])
            cells.append(cell_metrics(tag, subset, cell, seed))
    # optional one-seed extras currently produced by o3d
    for tag in ("M4b_popgate",):
        base = os.path.join(ROOT, "experiments", "order03", tag, subset)
        if os.path.isdir(base):
            for cell in sorted(os.listdir(base)):
                cells.append(cell_metrics(tag, subset, cell, int(cell.rsplit("_s", 1)[1])))

    pd.DataFrame([
        dict(tag=c["tag"], cell=c["cell"], seed=c["seed"], variant=c.get("variant"),
             params=c.get("params"), best_epoch=c.get("best_epoch"), epochs=c.get("epochs_ran"),
             val=c.get("best_val_rmse"),
             **{f"matched_{k}": c["matched"][k] for k in ("rmse", "mae", "bias", "nasa", "late_fraction", "severe_late_gt10", "severe_late_count")},
             **{f"protoc_{k}": c["protocol_c"][k] for k in ("rmse", "macro_rmse", "late_fraction", "severe_late_gt10")},
             **{f"protob_{k}": c["protocol_b"][k] for k in ("rmse", "macro_rmse", "late_fraction", "severe_late_gt10")},
             )
        for c in cells
    ]).to_csv(os.path.join(OUT, "cells.csv"), index=False)

    group = []
    for tag in sorted({c["tag"] for c in cells}):
        rows = [c for c in cells if c["tag"] == tag and c["subset"] == subset]
        if len(rows) < 2:
            continue
        row = dict(tag=tag, n_seeds=len(rows))
        for kind in ("matched", "protocol_c", "protocol_b"):
            for k in ("rmse", "macro_rmse", "late_fraction", "severe_late_gt10", "nasa"):
                if kind not in rows[0] or k not in rows[0][kind]:
                    continue
                xs = [r[kind][k] for r in rows]
                row[f"{kind}_{k}_mean"] = float(np.mean(xs))
                row[f"{kind}_{k}_sd"] = float(np.std(xs, ddof=1))
                row[f"{kind}_{k}_vals"] = [round(float(x), 4) for x in xs]
        group.append(row)
    pd.DataFrame(group).to_csv(os.path.join(OUT, "group_summary.csv"), index=False)

    frames = load_frames(tags + (["M4b_popgate"] if os.path.isdir(os.path.join(ROOT, "experiments", "order03", "M4b_popgate", subset)) else []), subset)
    val_sets = {}
    for kind, df in frames.items():
        for tag, g in df.groupby("tag"):
            val_sets[(kind, tag)] = {int(x) for x in g.unit_id.unique()}
    ref = val_sets[("matched", "M0_raw_tf")]
    unit_check = {f"{k[0]}/{k[1]}": bool(v == ref) for k, v in val_sets.items()}

    paired = []
    for kind in ("matched", "protocol_c", "protocol_b"):
        df = frames.get(kind)
        if df is None:
            continue
        for a_tag in ("M4_racdr", "M3_cond", "M2_pop"):
            for b_tag in ("M0_raw_tf", "M3_cond"):
                if a_tag == b_tag:
                    continue
                if a_tag not in set(df.tag) or b_tag not in set(df.tag):
                    continue
                a = df[df.tag == a_tag]; b = df[df.tag == b_tag]
                # A_s2027 vs RACDR_s2027 are the same model seed; pairing is on seed int
                r = paired_bootstrap(a, b, kind)
                r.update(a_tag=a_tag, b_tag=b_tag)
                paired.append(r)

    out = dict(subset=subset, cells=cells, group=group, paired=paired, unit_set_check=unit_check,
               n_units=int(len(ref)), generated_by="results/dsh/analysis/aggregate_order03.py")
    with open(os.path.join(OUT, "aggregate.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("== cells ==")
    for c in cells:
        if "matched" not in c:
            print(f"{c['tag']:12s} {c['cell']:14s} NO PREDICTIONS"); continue
        m = c["matched"]
        print(f"{c['tag']:12s} {c['cell']:14s} matched={m['rmse']:.4f} late={m['late_fraction']:.3f} "
              f"sev10={m['severe_late_gt10']:.3f} nasa={m['nasa']:.1f} "
              f"protoB={c['protocol_b']['rmse']:.4f} protoB_macro={c['protocol_b']['macro_rmse']:.4f} "
              f"protoC={c['protocol_c']['rmse']:.4f} protoC_macro={c['protocol_c']['macro_rmse']:.4f}")
    print("\n== group (mean sd) ==")
    for r in group:
        print(f"{r['tag']:12s} n={r['n_seeds']} matched={r['matched_rmse_mean']:.4f}+-{r['matched_rmse_sd']:.4f} "
              f"late={r['matched_late_fraction_mean']:.3f}+-{r['matched_late_fraction_sd']:.3f} "
              f"sev10={r['matched_severe_late_gt10_mean']:.3f} "
              f"protoB={r['protocol_b_rmse_mean']:.4f} protoC={r['protocol_c_rmse_mean']:.4f} "
              f"protoCmacro={r['protocol_c_macro_rmse_mean']:.4f}")
    print("\n== unit set check ==")
    print(unit_check, "n_units", len(ref))
    print("\n== paired (A - B) ==")
    for r in paired:
        print(f"{r['kind']:11s} {r['a_tag']:10s} - {r['b_tag']:10s} diff={r['rmse_diff']:+.4f} "
              f"CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] excl0={r['excludes_zero']} "
              f"p(A better)={r['prob_a_better']:.3f} n_eng={r['n_engines']} pairs={r['n_pairs']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
