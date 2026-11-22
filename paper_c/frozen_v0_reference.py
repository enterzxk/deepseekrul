"""Export the frozen fixed40 V0 runs as (a) an A-段 reconciliation table and
(b) a gate1.py baseline input.

Why this exists
---------------
经理's A 段 is "retrain the 30 V0 cells on 3090, then reconcile against the
frozen report at |delta| <= 0.2".  The frozen per-engine predictions are NOT
only on a40 -- they are committed in git on
``codex/paper-b-open-source-reproduction`` (d7bd9c17) as
``validation_predictions.csv`` inside each ``physical_runs/fixed40/<method>/
<subset>/<seed>/`` directory.  So the reconciliation reference needs no weights
and no cross-machine transfer.

This script reads those committed CSVs straight out of git and writes:

1. ``--out-cells``   per-cell reconciliation table (30 cells), each with the
                     engine-mean RMSE that A 段 must land within 0.2 of.
2. ``--out-contract`` the gate1.py baseline arm, one row per
                     (method, subset, seed, engine, cut) with the exact keys
                     ``method, subset, seed, engine, cut, y, pred``.

The ``cut`` column in the frozen CSV is a *truncation index*, while gate1.py
compares ``cut`` to EOL_CUT = 0.95 as a *fraction of life*.  We recover the
fraction from the fact that the five cuts per engine sit on the frozen grid
f in {0.40, 0.55, 0.70, 0.85, 0.95}: sorted ascending, cut_i / f_i must return
the same lifetime.  That identity is asserted per engine, so a silent
reordering cannot pass.

Self-check: the recomputed engine-mean RMSE is compared against the committed
``validation_metrics.json`` of the same cell.  A mismatch is an error, not a
warning -- A 段's 0.2 tolerance is meaningless if this table is not the same
quantity the frozen report used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

F_GRID = (0.40, 0.55, 0.70, 0.85, 0.95)
EOL_CUT = 0.95
RUL_CAP = 125
ROOT = "paper_b/experiments/open_source_matched_20261112/physical_runs/fixed40"
METHODS = ("aidgn", "two_phase", "dual_mixer", "ffve", "fcstgnn")
SUBSETS = ("FD001", "FD004")


def git(ref: str, *args: str) -> bytes:
    out = subprocess.run(["git", *args], capture_output=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({out.returncode}): "
            f"{out.stderr.decode('utf-8', 'replace').strip()}"
        )
    return out.stdout


def cell_paths(ref: str) -> list[tuple[str, str, str, str]]:
    """(method, subset, seed, csv_path) for every committed fixed40 cell."""
    raw = git(ref, "ls-tree", "-r", "--name-only", ref, "--", ROOT).decode()
    cells = []
    for line in raw.splitlines():
        parts = line.split("/")
        if not line.endswith("/validation_predictions.csv"):
            continue
        i = parts.index("fixed40")
        method, subset, seed = parts[i + 1], parts[i + 2], parts[i + 3]
        cells.append((method, subset, seed, line))
    return sorted(cells)


def read_csv(ref: str, path: str) -> tuple[list[dict], str]:
    blob = git(ref, "show", f"{ref}:{path}")
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8"))))
    return rows, hashlib.sha256(blob).hexdigest()


def recover_lifetime(cuts: list[int], cell: str, unit: str) -> int:
    """Invert cut_i == round(f_i * n) for the integer lifetime n.

    Do not use cut_i / f_i as an estimate: at f = 0.40 a one-step integer
    rounding error is amplified 2.5x, which makes a correct grid look
    inconsistent (measured spread 1.82 on aidgn/FD004/s101 unit 95).  The
    exact model is integer rounding, so we search the integers that satisfy
    all five cuts at once and require a unique solution.
    """
    top = cuts[-1]
    cands = [
        n
        for n in range(int(top / F_GRID[-1]) - 3, int(top / F_GRID[-1]) + 4)
        if all(int(round(f * n)) == c for f, c in zip(F_GRID, cuts))
    ]
    if len(cands) != 1:
        raise AssertionError(
            f"{cell} unit={unit}: cut grid {cuts} has {len(cands)} integer "
            f"lifetime solutions {cands}, want exactly 1"
        )
    return cands[0]


def assign_fractions(rows: list[dict], cell: str) -> list[dict]:
    """Recover f = cut / lifetime per engine and assert the grid identity."""
    by_unit: dict[str, list[dict]] = {}
    for r in rows:
        by_unit.setdefault(r["unit"], []).append(r)

    out = []
    for unit, pts in by_unit.items():
        pts = sorted(pts, key=lambda r: int(r["cut"]))
        if len(pts) != len(F_GRID):
            raise AssertionError(f"{cell} unit={unit}: {len(pts)} cuts, want {len(F_GRID)}")
        cuts = [int(p["cut"]) for p in pts]
        life = recover_lifetime(cuts, cell, unit)
        for p, f in zip(pts, F_GRID):
            # Independent check on the labels: RUL is min(cap, life - cut), so
            # every uncapped row must satisfy y == life - cut exactly.
            y = float(p["y_true"])
            if y < RUL_CAP and y != life - int(p["cut"]):
                raise AssertionError(
                    f"{cell} unit={unit} cut={p['cut']}: y_true {y} != "
                    f"life {life} - cut {p['cut']} = {life - int(p['cut'])}"
                )
            out.append(
                {
                    "engine": int(p["unit"]),
                    "cut": f,
                    "y": y,
                    "pred": float(p["y_pred"]),
                }
            )
    return out


def cell_metrics(points: list[dict]) -> dict:
    """pooled RMSE, engine-mean RMSE, and EOL(0.95) MAE."""
    sq = [(p["y"] - p["pred"]) ** 2 for p in points]
    pooled = (sum(sq) / len(sq)) ** 0.5

    per_engine: dict[int, list[dict]] = {}
    for p in points:
        per_engine.setdefault(p["engine"], []).append(p)
    engine_rmse = {
        u: (sum((p["y"] - p["pred"]) ** 2 for p in pts) / len(pts)) ** 0.5
        for u, pts in per_engine.items()
    }
    eol = [
        abs(p["y"] - p["pred"])
        for pts in per_engine.values()
        for p in [min(pts, key=lambda q: abs(q["cut"] - EOL_CUT))]
    ]
    return {
        "rmse_pooled": round(pooled, 6),
        "rmse_engine_mean": round(sum(engine_rmse.values()) / len(engine_rmse), 6),
        "eol_mae": round(sum(eol) / len(eol), 6),
        "n_engines": len(engine_rmse),
        "n_rows": len(points),
    }


def frozen_metrics(ref: str, csv_path: str) -> dict | None:
    path = csv_path.replace("validation_predictions.csv", "validation_metrics.json")
    out = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, check=False
    )
    if out.returncode != 0:
        return None
    return json.loads(out.stdout.decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="codex/paper-b-open-source-reproduction")
    ap.add_argument("--out-cells", required=True)
    ap.add_argument("--out-contract", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    a = ap.parse_args()

    commit = git(a.ref, "rev-parse", a.ref).decode().strip()
    cells = cell_paths(a.ref)
    print(f"ref {a.ref} -> {commit}")
    print(f"cells found: {len(cells)}")

    table, contract, problems = [], [], []
    seen = set()
    for method, subset, seed, path in cells:
        if method not in METHODS or subset not in SUBSETS:
            continue
        rows, sha = read_csv(a.ref, path)
        tag = f"{method}/{subset}/{seed}"
        points = assign_fractions(rows, tag)
        m = cell_metrics(points)
        m.update(method=method, subset=subset, seed=seed, csv_sha256=sha)
        table.append(m)
        seen.add((method, subset, seed))

        frozen = frozen_metrics(a.ref, path)
        if frozen is not None:
            got = frozen.get("val_engine_mean_rmse") or frozen.get("val_engine_rmse")
            if got is not None and abs(float(got) - m["rmse_engine_mean"]) > a.tolerance:
                problems.append(
                    f"{tag}: recomputed engine-mean RMSE {m['rmse_engine_mean']} "
                    f"!= frozen {got} (tol {a.tolerance})"
                )

        for p in points:
            contract.append(
                {
                    "method": method,
                    "subset": subset,
                    "seed": int(seed.lstrip("s")),
                    "engine": p["engine"],
                    "cut": p["cut"],
                    "y": p["y"],
                    "pred": p["pred"],
                }
            )

    expected = {(m, s) for m in METHODS for s in SUBSETS}
    got_pairs = {(c["method"], c["subset"]) for c in table}
    missing = expected - got_pairs
    if missing:
        problems.append(f"missing method/subset pairs: {sorted(missing)}")
    if len(table) != 30:
        problems.append(f"expected 30 cells, got {len(table)}")

    Path(a.out_cells).write_text(
        json.dumps(
            {
                "source_ref": a.ref,
                "source_commit": commit,
                "protocol": {
                    "split_seed": 7,
                    "rul_cap": 125,
                    "epochs": 40,
                    "window": 30,
                    "selection": "val_engine_mean_rmse",
                },
                "cells": table,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    Path(a.out_contract).write_text(
        "".join(json.dumps(r) + "\n" for r in contract), encoding="utf-8"
    )

    print(f"cells written: {len(table)}  contract rows: {len(contract)}")
    print(f"engines per cell: {sorted({c['n_engines'] for c in table})}")
    print(f"rows per cell:    {sorted({c['n_rows'] for c in table})}")
    for c in table[:3]:
        print(
            f"  {c['method']:11} {c['subset']} {c['seed']}  "
            f"engRMSE {c['rmse_engine_mean']:.4f}  EOL {c['eol_mae']:.4f}"
        )
    if problems:
        print("\nFAILED self-checks:")
        for p in problems:
            print("  -", p)
        return 2
    print("\nall self-checks passed (recomputed == frozen validation_metrics.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
