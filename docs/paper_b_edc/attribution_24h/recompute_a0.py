"""Recompute the four A0 (CycleOnly) cells from the A40 HC prediction files.

A0 lives outside `paper_b/results/raw`: its predictions are
`experiments/hc/HC-C-A0-<subset>-s101-W1/preds.csv` with columns
`engine_id, cut, cycle, y_true, y_pred, split`.  They are read here as a source,
not retrained: the A40 machine is a file and checkpoint source only.

Same two conventions as `recompute_val.py`:
  primary      label capped at rul_cap, prediction untouched  (historical)
  sensitivity  label capped, prediction clipped to [0, rul_cap]

Also reported: `mean_e MSE_e`, because a share-of-error figure has to divide by a
mean of squared errors.  `(mean_e RMSE_e)**2` is NOT that quantity -- the two
differ by Jensen in the same way pooled and engine-mean do.

Usage:
  python recompute_a0.py --hc-root <dir with HC-C-A0-*-W1/preds.csv>
"""
import argparse
import csv
import math
import os

CAP = 125.0


def read_hc(path):
    units, true, pred = [], [], []
    bad = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split = (row.get("split") or "val").strip().lower()
            if split and split != "val":
                continue
            try:
                u = int(float(row["engine_id"]))
                y = float(row["y_true"])
                q = float(row["y_pred"])
            except (KeyError, TypeError, ValueError):
                bad += 1
                continue
            if not (math.isfinite(y) and math.isfinite(q)):
                bad += 1
                continue
            units.append(u)
            true.append(y)
            pred.append(q)
    return units, true, pred, bad


def readings(units, true, pred):
    out = {}
    for tag, clip_pred in (("label_capped", False), ("pred_clipped", True)):
        t = [min(v, CAP) for v in true]
        p = [min(max(v, 0.0), CAP) for v in pred] if clip_pred else list(pred)
        per = {}
        for u, y, q in zip(units, t, p):
            per.setdefault(u, []).append((y, q))
        mse = [sum((q - y) ** 2 for y, q in pts) / len(pts) for pts in per.values()]
        out["engine_" + tag] = sum(math.sqrt(v) for v in mse) / len(mse)
        out["mean_e_mse_" + tag] = sum(mse) / len(mse)
        out["pooled_" + tag] = math.sqrt(sum(mse_e * len(pts) for mse_e, pts in
                                            zip(mse, per.values())) / len(t))
        out["n_engines"] = len(per)
        out["n_points"] = len(t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hc-root", required=True)
    ap.add_argument("--arm", default="A0")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(args.hc_root)):
        if not name.startswith("HC-C-%s-" % args.arm):
            continue
        p = os.path.join(args.hc_root, name, "preds.csv")
        if not os.path.exists(p):
            continue
        # HC-C-A0-FD001-s101-W1
        parts = name.split("-")
        subset = parts[3]
        seed = parts[4]
        units, true, pred, bad = read_hc(p)
        if not units:
            continue
        rd = readings(units, true, pred)
        rows.append(dict(
            arm=args.arm, subset=subset, seed=seed, source=os.path.join(name, "preds.csv"),
            n_engines=rd["n_engines"], n_points=rd["n_points"],
            points_per_engine=rd["n_points"] // rd["n_engines"],
            engine_label_capped=rd["engine_label_capped"],
            pooled_label_capped=rd["pooled_label_capped"],
            mean_e_mse_label_capped=rd["mean_e_mse_label_capped"],
            engine_pred_clipped=rd["engine_pred_clipped"],
            pooled_pred_clipped=rd["pooled_pred_clipped"],
            n_bad_rows=bad))

    cols = ["arm", "subset", "seed", "source", "n_engines", "n_points",
            "points_per_engine", "engine_label_capped", "pooled_label_capped",
            "mean_e_mse_label_capped", "engine_pred_clipped",
            "pooled_pred_clipped", "n_bad_rows"]
    out = os.path.join(args.out_dir, "A0_RECOMPUTE.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    print("cells:", len(rows))
    for r in rows:
        print("  %-6s %-5s n_eng=%-3s pts=%-3s engine=%.6f pooled=%.6f mean_e_MSE=%.6f"
              % (r["subset"], r["seed"], r["n_engines"], r["n_points"],
                 r["engine_label_capped"], r["pooled_label_capped"],
                 r["mean_e_mse_label_capped"]))


if __name__ == "__main__":
    main()
