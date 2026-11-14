"""Task C: four conditional-error tables from the saved validation predictions.

Everything here is read-only: the predictions come from `paper_b/results/raw/*/`,
the operating-condition clusters and the sensor values come from the CMAPSS text
files.  No training, no new checkpoints.

Conventions
-----------
* Primary reading: label capped at 125, prediction NOT clipped (the historical
  convention; `metrics.py` computes from the stored predictions).
* `cut` in val_pred.csv is joined to `cycle` in the CMAPSS train file.  The join
  is counted and reported: an unmatched row is a schema problem, not a modelling
  one, and it is never silently dropped.
* Operating conditions: KMeans(k=6, random_state=7) on the three op-setting
  columns, the same construction `data.py::OpScaler` uses, so a regime label
  here means the same thing it meant to the model.
* Every per-point statistic is first reduced to one number per (run, engine),
  then aggregated.  Points inside one engine are not independent, so pooling
  them would shrink every interval for free.

Outputs `TASK_C_CONDITIONAL_ERRORS.csv` and a printed summary.
"""
import argparse
import collections
import csv
import glob
import math
import os

import numpy as np
import pandas as pd

CAP = 125.0
OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = ["s%d" % i for i in range(1, 22)]
RAW_COLS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t975(df):
    return T975.get(df, 1.96 if df > 30 else float("nan"))


def ci95(values):
    """Mean and 95% t interval over independent units (runs or engines)."""
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(np.mean(v))
    if n == 1:
        return m, float("nan"), float("nan"), 1
    sd = float(np.std(v, ddof=1))
    half = t975(n - 1) * sd / math.sqrt(n)
    return m, m - half, m + half, n


def load_raw(cmapss_dir, subset):
    path = os.path.join(cmapss_dir, "train_%s.txt" % subset)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLS)
    df["cycle"] = df["cycle"].astype(int)
    return df


def regime_frames(cmapss_dir, subsets):
    from sklearn.cluster import KMeans
    out = {}
    for s in subsets:
        df = load_raw(cmapss_dir, s)
        km = KMeans(n_clusters=6, random_state=7, n_init=20).fit(
            df[OP_COLS].to_numpy(np.float64))
        df["regime"] = km.predict(df[OP_COLS].to_numpy(np.float64))
        # cluster-wise z-score, same shape as OpScaler.transform
        z = np.zeros((len(df), len(SENSOR_COLS)))
        for c in range(6):
            m = df["regime"].to_numpy() == c
            if m.sum() < 2:
                continue
            sub = df.loc[m, SENSOR_COLS].to_numpy(np.float64)
            mu, sd = sub.mean(0), sub.std(0)
            sd = np.where(sd > 0, sd, 1.0)
            z[m] = (sub - mu) / sd
        df["pc1"] = z @ np.linalg.svd(z, full_matrices=False)[2][0]
        # lifetime per unit: the last cycle the engine appears in the train file
        life = df.groupby("unit")["cycle"].max()
        df["lifetime"] = df["unit"].map(life)
        df["rel_pos"] = df["cycle"] / df["lifetime"]
        out[s] = df[["unit", "cycle", "regime", "pc1", "lifetime", "rel_pos"]]
    return out


def load_preds(raw_dir, tags):
    frames = []
    for tag in tags:
        for p in sorted(glob.glob(os.path.join(raw_dir, tag, "*", "*", "val_pred.csv"))):
            parts = p.replace("\\", "/").split("/")
            subset, seed = parts[-3], parts[-2]
            df = pd.read_csv(p)
            df["tag"] = tag
            df["subset"] = subset
            df["seed"] = seed
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.expanduser("~/rulb/paper_b/results/raw"))
    ap.add_argument("--cmapss", default=os.path.expanduser("~/rulb/data/CMAPSS"))
    ap.add_argument("--tag", action="append", default=None,
                    help="run tag to include; repeatable (default: parent + A8 arms)")
    ap.add_argument("--out", default="TASK_C_CONDITIONAL_ERRORS.csv")
    args = ap.parse_args()

    tags = args.tag or (["F_full_basic_hi1"] +
                        ["A8_%s_L70" % a for a in
                         ("base", "ms", "sa", "hi", "mssa", "mshi", "sahi", "full")])
    preds = load_preds(args.raw, tags)
    subsets = sorted(preds["subset"].unique())
    reg = regime_frames(args.cmapss, subsets)

    joined = []
    for s in subsets:
        p = preds[preds["subset"] == s].copy()
        m = p.merge(reg[s], left_on=["unit", "cut"], right_on=["unit", "cycle"],
                    how="left")
        joined.append(m)
    df = pd.concat(joined, ignore_index=True)
    n_all = len(df)
    n_miss = int(df["regime"].isna().sum())
    df = df[df["regime"].notna()].copy()
    df["regime"] = df["regime"].astype(int)
    df["err"] = df["y_pred"] - df["y_true"]
    df["ae"] = df["err"].abs()
    df["capped"] = (df["y_true"] >= CAP - 1e-9)

    rows = []

    def add(analysis, subset, stratum, metric, value, n, note=""):
        rows.append(dict(analysis=analysis, subset=subset, stratum=stratum,
                         metric=metric, value=value, n=n, note=note))

    add("schema", "all", "join", "points", n_all, "")
    add("schema", "all", "join", "points_unmatched_cycle", n_miss, "")
    add("schema", "all", "pool", "runs", int(preds.groupby(["tag", "subset", "seed"]).ngroups),
        n_all, "A8 96 cells + F_full_basic_hi1 12 cells")

    # ---- A1: same running age, different engines -------------------------- #
    # Two axes, deliberately named differently and never substituted for each
    # other:
    #   A1a "same running age"     = absolute cycle, 50-cycle bins
    #   A1b "same life fraction"   = cut / that engine's lifetime
    # A machine at cycle 100 may be halfway through its life or 80% through it,
    # so the two questions have different answers.
    def engine_level(g):
        """One mean error per engine, then the cross-engine spread."""
        per_eng = g.groupby("unit")["err"].mean()
        return per_eng

    for axis, col, width, label in (("A1a_same_age", "cut", 50, "cycle"),
                                    ("A1b_same_life_fraction", "rel_pos", 0.05, "life")):
        if axis.startswith("A1a"):
            df["axis_bin"] = (df[col] // width) * width
        else:
            df["axis_bin"] = (df[col] / width).round() * width
        for (s, b), g in df.groupby(["subset", "axis_bin"]):
            per_run_med, per_run_iqr, per_run_n = [], [], []
            for (tag, seed), gg in g.groupby(["tag", "seed"]):
                pe = engine_level(gg)
                if len(pe) < 3:
                    continue
                per_run_med.append(float(np.median(pe)))
                per_run_iqr.append(float(np.percentile(pe, 75) - np.percentile(pe, 25)))
                per_run_n.append(len(pe))
            if not per_run_med:
                continue
            all_eng = engine_level(g)
            med = float(np.median(all_eng))
            iqr = float(np.percentile(all_eng, 75) - np.percentile(all_eng, 25))
            m, lo, hi, k = ci95(per_run_med)
            if axis.startswith("A1a"):
                stratum = "cycle %d-%d" % (b, b + width - 1)
            else:
                stratum = "life %.2f" % b
            n_eng = int(g["unit"].nunique())
            thin = "engines<10: raw value only" if n_eng < 10 else ""
            add(axis, s, stratum, "median_error_across_engines", med, n_eng,
                "IQR %.3f; CI over %d runs %.3f..%.3f%s"
                % (iqr, k, lo, hi, ("; " + thin) if thin else ""))
            add(axis, s, stratum, "engines", n_eng, len(g), "")

    # ---- A2: regime and post-switch -------------------------------------- #
    # a window is "post-switch" if the regime differs from the previous cycle
    sw = {}
    for s in subsets:
        r = reg[s].sort_values(["unit", "cycle"])
        prev = r.groupby("unit")["regime"].shift(1)
        r = r.assign(post=(r["regime"] != prev) & prev.notna())
        sw[s] = r.set_index(["unit", "cycle"])["post"]
    df["post"] = [bool(sw[s].get((u, c), False)) for s, u, c in
                  zip(df["subset"], df["unit"], df["cut"])]

    for s in subsets:
        d = df[df["subset"] == s]
        if d["regime"].nunique() < 2:
            add("A2_regime", s, "single regime", "regimes", d["regime"].nunique(), 0, "")
            continue
        # paired within engine: post-switch mean ae minus steady mean ae
        diffs = []
        for (tag, seed, unit), gg in d.groupby(["tag", "seed", "unit"]):
            a = gg.loc[gg["post"], "ae"]
            b = gg.loc[~gg["post"], "ae"]
            if len(a) >= 1 and len(b) >= 1:
                diffs.append(float(a.mean() - b.mean()))
        m, lo, hi, n = ci95(diffs)
        add("A2_regime", s, "post-switch vs steady", "paired_abs_error_diff", m, n,
            "CI %.3f..%.3f over engines" % (lo, hi))
        add("A2_regime", s, "post-switch vs steady", "n_post_windows",
            int(d["post"].sum()), len(d), "")

    # ---- A3: capped vs degradation --------------------------------------- #
    for s in subsets:
        d = df[df["subset"] == s]
        cap = d[d["capped"]]
        deg = d[~d["capped"]]
        add("A3_capped", s, "y_true==125", "mean_abs_error", float(cap["ae"].mean()),
            len(cap), "")
        add("A3_capped", s, "y_true<125", "mean_abs_error", float(deg["ae"].mean()),
            len(deg), "")
        add("A3_capped", s, "y_true==125", "mean_signed_error", float(cap["err"].mean()),
            len(cap), "positive = over-prediction")

    # ---- A4: sensor information inside (cycle bin, regime) cells --------- #
    # 50-cycle bins, so a cell has enough engines for the comparison to mean
    # something; a 25-cycle bin left cells with 5 engines and one regime.
    df["bin50"] = (df["cut"] // 50) * 50
    for s in subsets:
        d = df[df["subset"] == s]
        for (b, r), g in d.groupby(["bin50", "regime"]):
            if len(g) < 50 or g["unit"].nunique() < 5:
                continue
            rho = g[["pc1", "ae"]].corr(method="spearman").iloc[0, 1]
            add("A4_sensor_within_cell", s, "cycle %d-%d regime %d" % (b, b + 49, r),
                "spearman_abs_error_vs_pc1", float(rho), len(g),
                "engines %d" % g["unit"].nunique())

    # ---- A4b: the same test inside each arm ------------------------------ #
    # The pooled A4 mixes eight arms, so "the sign is inconsistent across cells"
    # could be an artefact of mixing arms whose sensor-error relation points in
    # opposite directions.  If every arm is internally mixed, the pooled
    # statement stands; if each arm is internally consistent and the arms differ,
    # the statement has to become "the direction depends on the module
    # configuration", which is a different claim.
    per_arm = collections.defaultdict(list)
    for (tag, s, b, r), g in df.groupby(["tag", "subset", "bin50", "regime"]):
        if len(g) < 50 or g["unit"].nunique() < 5:
            continue
        rho = g[["pc1", "ae"]].corr(method="spearman").iloc[0, 1]
        if rho != rho:
            continue
        per_arm[tag].append(float(rho))
        add("A4b_per_arm_cell", s, "%s cycle %d-%d regime %d" % (tag, b, b + 49, r),
            "spearman_abs_error_vs_pc1", float(rho), len(g),
            "engines %d" % g["unit"].nunique())
    for tag, v in sorted(per_arm.items()):
        n_pos = sum(1 for x in v if x > 0)
        n_neg = sum(1 for x in v if x < 0)
        add("A4c_per_arm_summary", "all", tag, "cells", len(v), len(v),
            "pos %d neg %d median %+.3f" % (n_pos, n_neg, float(np.median(v))))
        add("A4c_per_arm_summary", "all", tag, "sign_consistency",
            ("mixed" if n_pos and n_neg else "consistent"), len(v),
            "pos %d neg %d" % (n_pos, n_neg))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["analysis", "subset", "stratum", "metric",
                                          "value", "n", "note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("points:", n_all, "unmatched cycle join:", n_miss)
    print("rows written:", len(rows))
    for r in rows:
        if r["analysis"] in ("A1a_same_age", "A1b_same_life_fraction",
                             "A2_regime", "A3_capped"):
            print("%-14s %-6s %-24s %-26s %8.3f n=%-5s %s"
                  % (r["analysis"], r["subset"], r["stratum"], r["metric"],
                     float(r["value"]), r["n"], r["note"]))


if __name__ == "__main__":
    main()
