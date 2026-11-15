"""Independent per-cell reader for the S1-A phase-2 audit.

Reads config.json and validation_predictions.csv only; never reads metrics.json
for scores. Emits per-engine RMSE so the pairing/bootstrap can be done elsewhere.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def engine_rmse(path):
    acc = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            u = int(r["unit"])
            t, p = float(r["y_true"]), float(r["y_pred"])
            acc.setdefault(u, []).append((t, p))
    return {u: math.sqrt(sum((p - t) ** 2 for t, p in pts) / len(pts))
            for u, pts in acc.items()}


def read_cell(d, arm):
    cfg_path = d / "config.json"
    pred_path = d / "validation_predictions.csv"
    if not cfg_path.exists() or not pred_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    er = engine_rmse(pred_path)
    rows, cuts = 0, {}
    with open(pred_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows += 1
            u = int(r["unit"])
            cuts[u] = cuts.get(u, 0) + 1
    method, subset, seed = d.parent.parent.name, d.parent.name, int(d.name[1:])
    problems = []
    if arm != "v0" and cfg.get("cycle_arm") != arm:
        problems.append(f"cycle_arm={cfg.get('cycle_arm')!r} != {arm!r}")
    if cfg.get("model") != method:
        problems.append(f"model={cfg.get('model')!r} != {method!r}")
    if cfg.get("subset") != subset:
        problems.append(f"subset={cfg.get('subset')!r} != {subset!r}")
    if cfg.get("seed") != seed:
        problems.append(f"seed={cfg.get('seed')!r} != {seed!r}")
    if any(v != 5 for v in cuts.values()):
        problems.append("some unit does not have 5 cuts")
    return {
        "arm": arm, "method": method, "subset": subset, "seed": seed,
        "path": str(d), "n_rows": rows, "n_units": len(er),
        "cuts_ok": all(v == 5 for v in cuts.values()),
        "engine_rmse": {str(u): er[u] for u in sorted(er)},
        "params": cfg.get("params"), "trainable_params": cfg.get("trainable_params"),
        "head": cfg.get("head"), "d_head": cfg.get("d_head"), "pe_proj": cfg.get("pe_proj"),
        "sham": cfg.get("sham"), "permutation_seed": cfg.get("permutation_seed"),
        "cycle_position": cfg.get("cycle_position"), "precision": cfg.get("precision"),
        "epochs": cfg.get("epochs"), "patience": cfg.get("patience"),
        "batch": cfg.get("batch"), "lr": cfg.get("lr"), "window": cfg.get("window"),
        "code_sha256": cfg.get("code_sha256"), "problems": problems,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="cycle_runs root (contains v1/ and sham/)")
    ap.add_argument("--v0-root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = []
    for arm in ("v1", "sham"):
        base = Path(a.root) / arm
        if not base.exists():
            continue
        for d in sorted(base.glob("*/FD00*/s*")):
            c = read_cell(d, arm)
            if c:
                cells.append(c)
    if a.v0_root:
        for d in sorted(Path(a.v0_root).glob("*/FD00*/s*")):
            c = read_cell(d, "v0")
            if c:
                cells.append(c)
    Path(a.out).write_text(json.dumps({"cells": cells}, indent=1), encoding="utf-8")
    print(f"cells={len(cells)}")


if __name__ == "__main__":
    main()
