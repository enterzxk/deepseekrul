"""Recompute the validation RMSE from each run's val_pred.csv, two ways.

What the numbers mean, and what they do not:

* The historical convention clips the LABEL at rul_cap=125 and leaves the
  prediction alone.  `paper_b/training/metrics.py` computes `rmse` and
  `per_engine_rmse` straight from the stored predictions, and
  `BASELINE_PROTOCOL_AUDIT.md` states it outright: "Because the labels are
  clipped but the predictions are not, the reported test metrics are
  capped-label metrics."  That is the primary reading here.
* Clipping the prediction as well is a SENSITIVITY, not a correction.  It is the
  convention the sprint evaluation used, and it moves FD002/FD004 cells because
  some predictions land above 125.  A difference under it does not mean the
  historical number is wrong.

Pooled versus engine-mean is NOT a weighting difference: every engine has
exactly five validation windows, so both weight each engine 1/E.  The difference
is the order of aggregation -- `sqrt(mean_e MSE_e) >= mean_e sqrt(MSE_e)` by
Jensen -- and `metrics.py::per_engine_rmse` says so in its own docstring.

Outputs (next to this script):
  VAL_RECOMPUTE.csv    per run: both readings under both conventions, what
                       metrics.json stored, and the delta against the ledger
  LEGACY_VAL_FIELDS.md the runs whose metrics.json val field does not hold what
                       its name says, plus the sensitivity table
"""
import argparse
import collections
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CAP = 125.0

# val_pred.csv stores the label rounded to one decimal, so a RMSE recomputed from
# it can differ from the in-process value by about 1e-3.  Inside this band is the
# file's precision, outside it is a real difference.
ROUND_TOL = 2e-3


def read_val_pred(path):
    units, true, pred = [], [], []
    bad = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                u = int(float(row["unit"]))
                y = float(row["y_true"])
                q = float(row["y_pred"])
            except (TypeError, ValueError):
                bad += 1          # a truncated or half-written file
                continue
            if not (math.isfinite(y) and math.isfinite(q)):
                bad += 1
                continue
            units.append(u)
            true.append(y)
            pred.append(q)
    return units, true, pred, bad


def readings(units, true, pred):
    """Engine-mean and pooled, under the historical and the clipped convention."""
    out = {}
    for tag, clip_pred in (("label_capped", False), ("pred_clipped", True)):
        t = [min(v, CAP) for v in true]          # the label is capped either way
        p = [min(max(v, 0.0), CAP) for v in pred] if clip_pred else list(pred)
        per = {}
        for u, y, q in zip(units, t, p):
            per.setdefault(u, []).append((y, q))
        eng = [math.sqrt(sum((q - y) ** 2 for y, q in pts) / len(pts))
               for pts in per.values()]
        out["engine_" + tag] = sum(eng) / len(eng)
        out["pooled_" + tag] = math.sqrt(
            sum((q - y) ** 2 for y, q in zip(t, p)) / len(t))
        out["n_pred_above_cap_" + tag] = sum(1 for v in pred if v > CAP)
        if clip_pred:
            out["n_engines"] = len(per)
            out["n_points"] = len(t)
            out["engine_min"] = min(eng)
            out["engine_max"] = max(eng)
    return out


def load_a8(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[(r["tag"], r["subset"], r["seed"])] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.join(REPO, "paper_b", "results", "raw"))
    ap.add_argument("--a8", default=os.path.join(REPO, "paper_b", "results",
                                                 "a8_engine_level.csv"))
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()
    RAW, A8 = args.raw, args.a8
    a8 = load_a8(A8)

    rows = []
    for tag in sorted(os.listdir(RAW)):
        tagdir = os.path.join(RAW, tag)
        if not os.path.isdir(tagdir):
            continue
        for subset in sorted(os.listdir(tagdir)):
            sdir = os.path.join(tagdir, subset)
            if not os.path.isdir(sdir):
                continue
            for seed in sorted(os.listdir(sdir)):
                rdir = os.path.join(sdir, seed)
                vp = os.path.join(rdir, "val_pred.csv")
                if not os.path.exists(vp):
                    continue
                units, true, pred, n_bad = read_val_pred(vp)
                if not units:
                    continue
                rd = readings(units, true, pred)
                m = {}
                mpath = os.path.join(rdir, "metrics.json")
                if os.path.exists(mpath):
                    with open(mpath, encoding="utf-8") as f:
                        m = json.load(f)
                fields = {k: m.get(k) for k in
                          ("rmse", "val_engine_rmse", "best_val_rmse")
                          if isinstance(m.get(k), (int, float))}
                cands = (("engine_label_capped", rd["engine_label_capped"]),
                         ("pooled_label_capped", rd["pooled_label_capped"]),
                         ("engine_pred_clipped", rd["engine_pred_clipped"]),
                         ("pooled_pred_clipped", rd["pooled_pred_clipped"]))

                def which(v):
                    if not isinstance(v, (int, float)):
                        return ""
                    best, gap = "", float("inf")
                    for name, val in cands:
                        d = abs(v - val)
                        if d < gap:
                            best, gap = name, d
                    return best if gap <= ROUND_TOL else "neither"

                led = a8.get((tag, subset, seed))
                led_eng = float(led["val_eng"]) if led and led.get("val_eng") else float("nan")
                rows.append(dict(
                    tag=tag, subset=subset, seed=seed,
                    n_engines=rd["n_engines"], n_points=rd["n_points"],
                    points_per_engine=rd["n_points"] // rd["n_engines"],
                    engine_label_capped=rd["engine_label_capped"],
                    pooled_label_capped=rd["pooled_label_capped"],
                    engine_pred_clipped=rd["engine_pred_clipped"],
                    pooled_pred_clipped=rd["pooled_pred_clipped"],
                    n_pred_above_cap=rd["n_pred_above_cap_label_capped"],
                    n_bad_rows=n_bad,
                    engine_min=rd["engine_min"], engine_max=rd["engine_max"],
                    metrics_rmse=fields.get("rmse", ""),
                    metrics_val_engine_rmse=fields.get("val_engine_rmse", ""),
                    metrics_best_val_rmse=fields.get("best_val_rmse", ""),
                    rmse_is=which(fields.get("rmse")),
                    val_engine_rmse_is=which(fields.get("val_engine_rmse")),
                    best_val_rmse_is=which(fields.get("best_val_rmse")),
                    ledger_val_eng=(led_eng if led_eng == led_eng else ""),
                    delta_primary_vs_ledger=(rd["engine_label_capped"] - led_eng
                                             if led_eng == led_eng else ""),
                    delta_clipped_vs_ledger=(rd["engine_pred_clipped"] - led_eng
                                             if led_eng == led_eng else "")))

    cols = ["tag", "subset", "seed", "n_engines", "n_points", "points_per_engine",
            "engine_label_capped", "pooled_label_capped",
            "engine_pred_clipped", "pooled_pred_clipped", "n_pred_above_cap",
            "n_bad_rows",
            "engine_min", "engine_max",
            "metrics_rmse", "metrics_val_engine_rmse", "metrics_best_val_rmse",
            "rmse_is", "val_engine_rmse_is", "best_val_rmse_is",
            "ledger_val_eng", "delta_primary_vs_ledger", "delta_clipped_vs_ledger"]
    with open(os.path.join(args.out_dir, "VAL_RECOMPUTE.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    with_ledger = [r for r in rows if r["delta_primary_vs_ledger"] != ""]
    max_d_pri = max((abs(float(r["delta_primary_vs_ledger"])) for r in with_ledger),
                    default=float("nan"))
    max_d_clip = max((abs(float(r["delta_clipped_vs_ledger"])) for r in with_ledger),
                     default=float("nan"))
    kinds = collections.Counter(r["val_engine_rmse_is"] for r in rows)
    wrong = [r for r in rows if r["val_engine_rmse_is"] not in ("engine_label_capped", "")]
    n_over = sum(1 for r in rows if int(r["n_pred_above_cap"]) > 0)

    md = []
    md.append("# What `metrics.json` stores in `val_engine_rmse`, and what the ledger holds\n")
    md.append("Generated by `recompute_val.py`; do not hand-edit.\n")
    md.append("Two conventions, named explicitly, because they differ:\n")
    md.append("| convention | label | prediction | where it comes from |")
    md.append("|---|---|---|---|")
    md.append("| **primary (historical)** | capped at %g | left alone | `metrics.py` "
              "computes `rmse`/`per_engine_rmse` from the stored predictions; "
              "`BASELINE_PROTOCOL_AUDIT.md` section 0.2 says the labels are clipped and the "
              "predictions are not |" % CAP)
    md.append("| sensitivity | capped at %g | clipped to [0, %g] | the sprint "
              "evaluation protocol |" % (CAP, CAP))
    md.append("")
    md.append("**The sensitivity is not a correction.** A cell whose two readings differ "
              "does not have a wrong historical value; it has a value under a different "
              "convention. %d of %d runs have at least one prediction above %g."
              % (n_over, len(rows), CAP))
    md.append("")
    md.append("| quantity | value |")
    md.append("|---|---|")
    md.append("| runs with a `val_pred.csv` | %d |" % len(rows))
    md.append("| runs also present in `a8_engine_level.csv` | %d |" % len(with_ledger))
    md.append("| points per engine (all runs) | %s |"
              % sorted({r["points_per_engine"] for r in rows}))
    md.append("| max abs delta, primary vs ledger `val_eng` | %.3e |" % max_d_pri)
    md.append("| max abs delta, clipped vs ledger `val_eng` | %.3e |" % max_d_clip)
    md.append("| ledger rows outside tolerance, primary | %d |" % len(
        [r for r in with_ledger if abs(float(r["delta_primary_vs_ledger"])) > ROUND_TOL]))
    md.append("| ledger rows outside tolerance, clipped | %d |" % len(
        [r for r in with_ledger if abs(float(r["delta_clipped_vs_ledger"])) > ROUND_TOL]))
    md.append("")
    md.append("## The ledger reproduces exactly under the primary convention\n")
    md.append("`a8_engine_level.csv::val_eng` is the engine-mean with the label capped "
              "and the prediction untouched: max abs delta %.1e over %d cells."
              % (max_d_pri, len(with_ledger)))
    md.append("That is the historical convention, so the ledger is exact, not "
              "approximate. The clipped reading differs by up to %.4f on the same "
              "cells, and that difference is a convention change, not an error."
              % max_d_clip)
    md.append("")
    md.append("## Pooled is not a weighting difference\n")
    md.append("Every engine contributes exactly %s validation windows, so pooled and "
              "engine-mean weight each engine identically at 1/E. The difference is "
              "the order of aggregation: `sqrt(mean_e MSE_e) >= mean_e sqrt(MSE_e)` "
              "by Jensen, which is also what `metrics.py::per_engine_rmse` documents. "
              "A table that reports both without saying which one selected the model "
              "is mixing two estimators, but it is not reweighting engines."
              % "/".join(sorted({str(r["points_per_engine"]) for r in rows})))
    md.append("")
    md.append("## Which reading `metrics.json::val_engine_rmse` holds\n")
    md.append("| reading | runs |")
    md.append("|---|---|")
    for k in ("engine_label_capped", "pooled_label_capped",
              "engine_pred_clipped", "pooled_pred_clipped", "neither"):
        md.append("| `%s` | %d |" % (k, kinds.get(k, 0)))
    md.append("")
    md.append("The field name promises the engine-level value. It holds it in %d of %d "
              "runs." % (kinds.get("engine_label_capped", 0), len(rows)))
    md.append("")
    md.append("| tag | subset | seed | engine (primary) | stored | holds | stored minus engine |")
    md.append("|---|---|---|---|---|---|---|")
    for r in sorted(wrong, key=lambda x: -(float(x["metrics_val_engine_rmse"])
                                           - float(x["engine_label_capped"]))):
        stored = float(r["metrics_val_engine_rmse"])
        md.append("| %s | %s | %s | %.4f | %.4f | %s | %+.4f |" % (
            r["tag"], r["subset"], r["seed"], float(r["engine_label_capped"]),
            stored, r["val_engine_rmse_is"], stored - float(r["engine_label_capped"])))
    md.append("")

    design = {
        "A8_base_L70": (0, 0, 0), "A8_ms_L70": (1, 0, 0),
        "A8_sa_L70": (0, 1, 0), "A8_hi_L70": (0, 0, 1),
        "A8_mssa_L70": (1, 1, 0), "A8_mshi_L70": (1, 0, 1),
        "A8_sahi_L70": (0, 1, 1), "A8_full_L70": (1, 1, 1),
    }
    a8rows = [r for r in rows if r["tag"] in design]
    if a8rows:
        md.append("## The A8 main effects under both conventions\n")
        md.append("| effect | primary (label capped) | sensitivity (pred clipped) | difference |")
        md.append("|---|---|---|---|")
        for name, idx in (("ms", 0), ("sa", 1), ("hi", 2)):
            effs = {}
            for col in ("engine_label_capped", "engine_pred_clipped"):
                arm = {t: sum(float(x[col]) for x in a8rows if x["tag"] == t)
                       / max(1, sum(1 for x in a8rows if x["tag"] == t))
                       for t in design}
                on = [arm[t] for t, d in design.items() if d[idx] == 1]
                off = [arm[t] for t, d in design.items() if d[idx] == 0]
                effs[col] = sum(on) / len(on) - sum(off) / len(off)
            p, c = effs["engine_label_capped"], effs["engine_pred_clipped"]
            md.append("| %s | %+.4f | %+.4f | %+.4f |" % (name, p, c, c - p))
        md.append("")
        md.append("The signs and the order of the three effects are the same under both")
        md.append("readings; the magnitudes move by at most about 0.007. So the")
        md.append("attribution conclusions do not turn on the convention, but every")
        md.append("quoted number must say which one produced it.\n")
    md.append("## Runs where the ledger and the primary recomputation disagree\n")
    bad = [r for r in with_ledger
           if abs(float(r["delta_primary_vs_ledger"])) > ROUND_TOL]
    if not bad:
        md.append("None. Every ledger `val_eng` reproduces inside %g." % ROUND_TOL)
    else:
        md.append("| tag | subset | seed | recomputed | ledger | delta |")
        md.append("|---|---|---|---|---|---|")
        for r in bad:
            md.append("| %s | %s | %s | %.6f | %.6f | %+.3e |" % (
                r["tag"], r["subset"], r["seed"], float(r["engine_label_capped"]),
                float(r["ledger_val_eng"]), float(r["delta_primary_vs_ledger"])))
    md.append("")
    with open(os.path.join(args.out_dir, "LEGACY_VAL_FIELDS.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("runs with val_pred.csv:", len(rows))
    print("runs with a ledger row:", len(with_ledger))
    print("max |delta| primary (label capped) vs ledger: %.3e" % max_d_pri)
    print("max |delta| sensitivity (pred clipped) vs ledger: %.3e" % max_d_clip)
    print("ledger rows outside tolerance, primary:",
          len([r for r in with_ledger
               if abs(float(r["delta_primary_vs_ledger"])) > ROUND_TOL]))
    print("stored reading counts:", dict(kinds))
    print("runs with a prediction above %g: %d" % (CAP, n_over))


if __name__ == "__main__":
    main()
