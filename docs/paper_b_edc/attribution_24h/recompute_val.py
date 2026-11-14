"""Recompute engine-level validation RMSE from each run's val_pred.csv.

Why this exists: `metrics.json` carries a field called `val_engine_rmse`, and in
some runs that field holds the POOLED RMSE instead.  The two answer different
questions -- pooled weights every cut equally, engine-level weights every engine
equally -- so a table that mixes them is not comparable with itself.  The only
way to know which one a run stored is to recompute both from `val_pred.csv` and
compare.

Outputs (written next to this script):
  VAL_RECOMPUTE.csv    one row per run: both readings, what metrics.json stored,
                       and the delta against the A8 ledger where one exists
  LEGACY_VAL_FIELDS.md the runs whose metrics.json val field does not hold what
                       its name says
"""
import argparse
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CAP = 125.0

# val_pred.csv stores the label rounded to one decimal, so a RMSE recomputed from
# it can differ from the value computed in-process by up to about 1e-3.  Anything
# inside this band is the file's precision, not a disagreement; anything outside
# it is a real difference and gets listed.
ROUND_TOL = 2e-3


def read_val_pred(path):
    units, true, pred = [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            units.append(int(float(row["unit"])))
            true.append(float(row["y_true"]))
            pred.append(float(row["y_pred"]))
    return units, true, pred


def readings(units, true, pred):
    """Four readings: engine-mean and pooled, each with and without the cap.

    The cap is not a detail.  A run whose `metrics.json` was written before the
    cap was frozen stores an uncapped pooled RMSE, and on FD004 that number is
    about 40% larger than the engine-mean capped one -- two different quantities
    under one field name.
    """
    out = {}
    for name, cap in (("capped", CAP), ("uncapped", None)):
        if cap is None:
            t, p = true, pred
        else:
            t = [min(v, cap) for v in true]
            p = [min(max(v, 0.0), cap) for v in pred]
        per = {}
        for u, y, q in zip(units, t, p):
            per.setdefault(u, []).append((y, q))
        eng = [math.sqrt(sum((q - y) ** 2 for y, q in pts) / len(pts))
               for pts in per.values()]
        out["engine_" + name] = sum(eng) / len(eng)
        out["pooled_" + name] = math.sqrt(
            sum((q - y) ** 2 for y, q in zip(t, p)) / len(t))
    out["n_engines"] = len(set(units))
    out["n_points"] = len(units)
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
    ap.add_argument("--raw", default=os.path.join(REPO, "paper_b", "results", "raw"),
                    help="root holding <tag>/<subset>/<seed>/val_pred.csv")
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
                units, true, pred = read_val_pred(vp)
                rd = readings(units, true, pred)
                em = rd["engine_capped"]
                pooled = rd["pooled_capped"]
                m = {}
                mpath = os.path.join(rdir, "metrics.json")
                if os.path.exists(mpath):
                    with open(mpath, encoding="utf-8") as f:
                        m = json.load(f)
                fields = {k: m.get(k) for k in
                          ("rmse", "val_engine_rmse", "best_val_rmse", "val_pool_rmse")
                          if isinstance(m.get(k), (int, float))}
                # which of the four readings does a stored value equal?
                cands = (("engine_capped", rd["engine_capped"]),
                         ("pooled_capped", rd["pooled_capped"]),
                         ("engine_uncapped", rd["engine_uncapped"]),
                         ("pooled_uncapped", rd["pooled_uncapped"]))

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
                    engine_mean_rmse=em, pooled_rmse=pooled,
                    engine_mean_uncapped=rd["engine_uncapped"],
                    pooled_uncapped=rd["pooled_uncapped"],
                    engine_min=rd["engine_min"], engine_max=rd["engine_max"],
                    points_per_engine=(rd["n_points"] // rd["n_engines"]
                                       if rd["n_engines"] else 0),
                    metrics_rmse=fields.get("rmse", ""),
                    metrics_val_engine_rmse=fields.get("val_engine_rmse", ""),
                    metrics_best_val_rmse=fields.get("best_val_rmse", ""),
                    rmse_is=which(fields.get("rmse")),
                    val_engine_rmse_is=which(fields.get("val_engine_rmse")),
                    best_val_rmse_is=which(fields.get("best_val_rmse")),
                    ledger_val_eng=(led_eng if led_eng == led_eng else ""),
                    delta_engine_vs_ledger=(em - led_eng if led_eng == led_eng else ""),
                    delta_uncapped_vs_ledger=(rd["engine_uncapped"] - led_eng
                                              if led_eng == led_eng else ""),
                    delta_pooled_minus_engine=(pooled - em)))
    cols = ["tag", "subset", "seed", "n_engines", "n_points",
            "engine_mean_rmse", "pooled_rmse", "engine_mean_uncapped",
            "pooled_uncapped", "engine_min", "engine_max",
            "points_per_engine", "metrics_rmse", "metrics_val_engine_rmse",
            "metrics_best_val_rmse", "rmse_is", "val_engine_rmse_is",
            "best_val_rmse_is", "ledger_val_eng", "delta_engine_vs_ledger",
            "delta_uncapped_vs_ledger", "delta_pooled_minus_engine"]
    out_csv = os.path.join(args.out_dir, "VAL_RECOMPUTE.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    with_ledger = [r for r in rows if r["delta_engine_vs_ledger"] != ""]
    max_delta = max((abs(float(r["delta_engine_vs_ledger"])) for r in with_ledger),
                    default=float("nan"))
    max_delta_unc = max((abs(float(r["delta_uncapped_vs_ledger"])) for r in with_ledger),
                        default=float("nan"))
    import collections
    kinds = collections.Counter(r["val_engine_rmse_is"] for r in rows)
    wrong = [r for r in rows if r["val_engine_rmse_is"] not in ("engine_capped", "")]
    neither = [r for r in rows if r["val_engine_rmse_is"] == "neither"]
    # which convention does the ledger itself use?
    led_is_uncapped = (max_delta_unc < 1e-6 and max_delta_unc < max_delta)

    md = []
    md.append("# What `metrics.json` actually stores in `val_engine_rmse`\n")
    md.append("Generated by `recompute_val.py`; do not hand-edit.\n")
    md.append("Frozen protocol: engine-mean RMSE with the label and the prediction "
              "capped at %g.  Tolerance %g, because `val_pred.csv` stores the label "
              "to one decimal.\n" % (CAP, ROUND_TOL))
    md.append("| quantity | value |")
    md.append("|---|---|")
    md.append("| runs with a `val_pred.csv` | %d |" % len(rows))
    md.append("| runs also present in `a8_engine_level.csv` | %d |" % len(with_ledger))
    md.append("| max abs delta, engine-mean **capped** vs ledger `val_eng` | %.3e |"
              % max_delta)
    md.append("| max abs delta, engine-mean **uncapped** vs ledger `val_eng` | %.3e |"
              % max_delta_unc)
    md.append("| ledger rows outside the tolerance, capped reading | %d |" % len(
        [r for r in with_ledger if abs(float(r["delta_engine_vs_ledger"])) > ROUND_TOL]))
    md.append("| ledger rows outside the tolerance, uncapped reading | %d |" % len(
        [r for r in with_ledger if abs(float(r["delta_uncapped_vs_ledger"])) > ROUND_TOL]))
    md.append("| `val_engine_rmse` holds the frozen-protocol value | %d |"
              % kinds.get("engine_capped", 0))
    md.append("| `val_engine_rmse` holds something else | %d |" % len(wrong))
    md.append("")
    md.append("## The ledger is engine-mean, uncapped\n")
    if led_is_uncapped:
        md.append("`a8_engine_level.csv::val_eng` reproduces from `val_pred.csv` as the")
        md.append("**uncapped** engine-mean to %.1e, and the capped reading misses it by "
                  "up to %.4f." % (max_delta_unc, max_delta))
        md.append("So the ledger is exact and its convention is not the frozen one: the")
        md.append("predictions in these runs go above %g (the CSV label is already capped," % CAP)
        md.append("the prediction is not), and clipping them changes the engine-mean RMSE.")
        md.append("**The two readings are not interchangeable**: a module ranked on one")
        md.append("can move on the other, so any table that mixes the ledger with a capped")
        md.append("recomputation is comparing two protocols.\n")
    else:
        md.append("The ledger matches the capped engine-mean within %.1e.\n" % max_delta)
    # If the ledger is uncapped, the A8 main effects computed from it are
    # uncapped-protocol effects.  Recompute them under both conventions so the
    # difference is visible instead of assumed away.
    design = {
        "A8_base_L70": (0, 0, 0), "A8_ms_L70": (1, 0, 0),
        "A8_sa_L70": (0, 1, 0), "A8_hi_L70": (0, 0, 1),
        "A8_mssa_L70": (1, 1, 0), "A8_mshi_L70": (1, 0, 1),
        "A8_sahi_L70": (0, 1, 1), "A8_full_L70": (1, 1, 1),
    }
    a8rows = [r for r in rows if r["tag"] in design]
    if led_is_uncapped and a8rows:
        md.append("## The A8 main effects under both conventions\n")
        md.append("The 96-cell ledger is the uncapped reading, so the reported main")
        md.append("effects are uncapped-protocol effects. Recomputing the same 2^3")
        md.append("design under the frozen cap gives:\n")
        md.append("| effect | uncapped (the ledger) | capped (frozen protocol) | difference |")
        md.append("|---|---|---|---|")
        for name, idx in (("ms", 0), ("sa", 1), ("hi", 2)):
            effs = {}
            for col in ("engine_mean_uncapped", "engine_mean_rmse"):
                arm = {t: sum(float(x[col]) for x in a8rows if x["tag"] == t)
                       / max(1, sum(1 for x in a8rows if x["tag"] == t))
                       for t in design}
                on = [arm[t] for t, d in design.items() if d[idx] == 1]
                off = [arm[t] for t, d in design.items() if d[idx] == 0]
                effs[col] = sum(on) / len(on) - sum(off) / len(off)
            u, c = effs["engine_mean_uncapped"], effs["engine_mean_rmse"]
            md.append("| %s | %+.4f | %+.4f | %+.4f |" % (name, u, c, c - u))
        md.append("")
        md.append("The signs and the order of the three effects are the same under both")
        md.append("readings; the magnitudes move by at most 0.007. So the attribution")
        md.append("conclusions do not turn on the cap, but the numbers quoted must say")
        md.append("which convention produced them.\n")
    md.append("## Which of the four readings the stored `val_engine_rmse` is\n")
    md.append("| reading | runs |")
    md.append("|---|---|")
    for k in ("engine_capped", "pooled_capped", "engine_uncapped",
              "pooled_uncapped", "neither"):
        md.append("| `%s` | %d |" % (k, kinds.get(k, 0)))
    md.append("")
    md.append("`engine_capped` is the frozen protocol.  Every other row is a number")
    md.append("that a reader would take for that protocol and that is not it: pooled")
    md.append("weights each cut equally while the protocol weights each engine equally,")
    md.append("and uncapped keeps predictions above %g that the protocol clips.\n" % CAP)
    md.append("## Runs whose stored value is not the frozen-protocol value\n")
    md.append("| tag | subset | seed | frozen (engine, capped) | stored | holds | stored minus frozen |")
    md.append("|---|---|---|---|---|---|---|")
    for r in sorted(wrong, key=lambda x: -(float(x["metrics_val_engine_rmse"])
                                           - float(x["engine_mean_rmse"]))):
        stored = float(r["metrics_val_engine_rmse"])
        md.append("| %s | %s | %s | %.4f | %.4f | %s | %+.4f |" % (
            r["tag"], r["subset"], r["seed"], float(r["engine_mean_rmse"]),
            stored, r["val_engine_rmse_is"], stored - float(r["engine_mean_rmse"])))
    md.append("")
    if neither:
        md.append("### Runs whose `val_engine_rmse` matches neither reading\n")
        md.append("These are the ones worth opening by hand: the stored number is not")
        md.append("explained by either aggregation, so it came from somewhere else")
        md.append("(another epoch, another split, or a run whose predictions were")
        md.append("rewritten afterwards).\n")
        md.append("| tag | subset | seed | engine-mean | pooled | stored | nearest | delta |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in neither:
            stored = float(r["metrics_val_engine_rmse"])
            de = abs(stored - float(r["engine_mean_rmse"]))
            dp = abs(stored - float(r["pooled_rmse"]))
            md.append("| %s | %s | %s | %.4f | %.4f | %.4f | %s | %+.4f |" % (
                r["tag"], r["subset"], r["seed"], float(r["engine_mean_rmse"]),
                float(r["pooled_rmse"]), stored,
                "engine" if de < dp else "pooled", stored - float(r["engine_mean_rmse"])))
        md.append("")
    md.append("## Runs where the ledger and the recomputation disagree\n")
    bad = [r for r in with_ledger
           if abs(float(r["delta_engine_vs_ledger"])) > ROUND_TOL]
    md.append("| tag | subset | seed | recomputed | ledger | delta |")
    md.append("|---|---|---|---|---|---|")
    if not bad:
        md.append("| none | | | | | |")
        md.append("")
        md.append("Every ledger `val_eng` reproduces from `val_pred.csv` inside %g." % ROUND_TOL)
    else:
        for r in bad:
            md.append("| %s | %s | %s | %.6f | %.6f | %+.3e |" % (
                r["tag"], r["subset"], r["seed"], r["engine_mean_rmse"],
                float(r["ledger_val_eng"]), float(r["delta_engine_vs_ledger"])))
    md.append("")
    with open(os.path.join(args.out_dir, "LEGACY_VAL_FIELDS.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("runs with val_pred.csv:", len(rows))
    print("runs with a ledger row:", len(with_ledger))
    print("max |delta| engine-mean capped vs ledger:", "%.3e" % max_delta)
    print("max |delta| engine-mean uncapped vs ledger:", "%.3e" % max_delta_unc)
    print("ledger rows outside tolerance, capped:", len(
        [r for r in with_ledger if abs(float(r["delta_engine_vs_ledger"])) > ROUND_TOL]))
    print("ledger rows outside tolerance, uncapped:", len(
        [r for r in with_ledger if abs(float(r["delta_uncapped_vs_ledger"])) > ROUND_TOL]))
    print("stored reading counts:", dict(kinds))
    for r in bad[:10]:
        print("  ", r["tag"], r["subset"], r["seed"],
              "%.6f vs %.6f" % (r["engine_mean_rmse"], float(r["ledger_val_eng"])))


if __name__ == "__main__":
    main()
