"""Independent E0 recompute from val_pred.csv.

Computes the two engine-level readings plus pooled, and records the four
metrics.json fields so the field-to-definition mapping can be checked.

Cells are resolved by split (see scripts/_cells.py): `runs/` holds every split
side by side and the run ids share a prefix, so a bare `E0-*` glob would now
mix splits.

Usage:
  python3 scripts/audit_e0.py <runs_root> [--split-seed 17]
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cells import resolve  # noqa: E402

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = (101, 202, 303)


def read_pred(path):
    acc, n = {}, 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            u = int(r.get("unit") or r.get("engine_id"))
            t, p = float(r["y_true"]), float(r["y_pred"])
            acc.setdefault(u, []).append((t, p))
            n += 1
    return acc, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--split-seed', type=int, default=17)
    a = ap.parse_args()
    root = Path(a.root)
    out = []
    for sub in SUBSETS:
        for seed in SEEDS:
            try:
                c = Path(resolve(root, 'E0', sub, seed, a.split_seed))
            except RuntimeError as exc:
                out.append({"cell": f'E0-{sub}-s{seed}', "missing": True, "why": str(exc)})
                continue
            pred, met = c / "val_pred.csv", c / "metrics.json"
            if not pred.exists() or not met.exists():
                out.append({"cell": str(c), "missing": True})
                continue
            acc, n = read_pred(pred)
            per_mse = {u: sum((p - t) ** 2 for t, p in pts) / len(pts) for u, pts in acc.items()}
            mean_roots = sum(math.sqrt(m) for m in per_mse.values()) / len(per_mse)
            engine_equal = math.sqrt(sum(per_mse.values()) / len(per_mse))
            pooled = math.sqrt(sum((p - t) ** 2 for pts in acc.values() for t, p in pts) / n)
            m = json.loads(met.read_text(encoding="utf-8"))
            out.append({
                "run": c.parents[1].name, "subset": sub, "seed": f's{seed}',
                "n_units": len(per_mse), "n_rows": n,
                "cuts_ok": all(len(v) == 5 for v in acc.values()),
                "mean_roots": mean_roots, "engine_equal": engine_equal, "pooled": pooled,
                "rec_mean_roots": m.get("val_engine_mean_roots_rmse"),
                "rec_engine_equal": m.get("val_engine_equal_rmse"),
                "rec_rmse": m.get("rmse"), "rec_pooled": m.get("val_pooled_rmse"),
                "rec_engine_rmse": m.get("val_engine_rmse"),
            })
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
