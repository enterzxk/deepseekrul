"""Ground truth for the R2.1 protocol-fidelity gates, computed from the frozen
anchor cells only.

Why this exists
---------------
Reviewing 军师's implementation by reading its output is not a review: every
number would be re-derived by the same code that is under review. This script
emits the expected values independently, from the committed anchor tree, so the
review compares two derivations instead of trusting one.

It is deliberately NOT a gate and NOT a second implementation of the criterion:
it only records what the anchors say. The judge stays gate1.py.

R2.1 gates covered (prereg v6, 66c38d41)
----------------------------------------
1. protocol fields   -- the 13 fields that must be byte-identical per cell
2. data fingerprint  -- training_data_sha256, including two_phase's paired
                        pretraining subset (FD001->FD003, FD004->FD002)
3. early-stop shape  -- epochs_run / best_epoch per cell, to be compared against
                        the rerun cell by cell, NOT against a constant
4. numeric health    -- curve length == epochs_run, no NaN, best_epoch in range

Trap this file exists to prevent: gate 3 must compare the rerun's shape to the
anchor's shape. A rule of the form "epochs_run == epochs means budget-truncated
means broken" would flag the three two_phase/FD004 anchors, which legitimately
ran to the 40-epoch ceiling. The anchors' own shapes are the reference.

Usage
-----
    python paper_c/anchor_protocol_truth.py \
        --anchor <frozen fixed40 dir> \
        --out docs/paper_c/registry/ANCHOR_PROTOCOL_TRUTH.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

PROTOCOL_FIELDS = (
    "window",
    "batch",
    "lr",
    "weight_decay",
    "epochs",
    "patience",
    "min_delta",
    "rul_cap",
    "split_seed",
    "val_frac",
    "precision",
    "tf32",
    "training_windows",
)
METHODS = ("aidgn", "two_phase", "dual_mixer", "ffve", "fcstgnn")
SUBSETS = ("FD001", "FD004")
SEEDS = ("101", "202", "303")


def cell_dir(anchor: Path, m: str, s: str, seed: str) -> Path:
    return anchor / m / s / f"s{seed}"


def protocol_sha256(cfg: dict) -> str:
    """Hash over the 13 protocol fields, so a per-cell diff is one comparison."""
    payload = json.dumps(
        {k: cfg.get(k, "<absent>") for k in PROTOCOL_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numeric_health(history: list, epochs: int, best: int) -> list[str]:
    """Numeric sanity of one cell's curve.

    `epochs` is the CEILING (40), not the number of epochs actually run: an
    early-stopped cell legitimately has 16-32 records. Comparing the curve
    length to the ceiling would flag all 25 early-stopped anchors as broken --
    that mistake was in the first version of this file. The invariants are:
    never exceed the ceiling, and never claim a best epoch the curve does not
    contain.
    """
    problems = []
    if len(history) > epochs:
        problems.append(f"curve length {len(history)} exceeds epochs ceiling {epochs}")
    if not (1 <= best <= len(history)):
        problems.append(f"best_epoch {best} outside 1..{len(history)}")
    for rec in history:
        for key in ("val_engine_mean_rmse", "val_pooled_rmse", "train_loss"):
            v = rec.get(key)
            if v is None:
                problems.append(f"epoch {rec.get('epoch')}: missing {key}")
            elif isinstance(v, float) and not math.isfinite(v):
                problems.append(f"epoch {rec.get('epoch')}: {key} not finite")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    anchor = Path(a.anchor)

    cells, problems = [], []
    for m in METHODS:
        for s in SUBSETS:
            for seed in SEEDS:
                d = cell_dir(anchor, m, s, seed)
                tag = f"{m}/{s}/s{seed}"
                try:
                    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
                    hist = json.loads((d / "history.json").read_text(encoding="utf-8"))
                    freeze = json.loads((d / "FREEZE.json").read_text(encoding="utf-8"))
                    vmet = json.loads(
                        (d / "validation_metrics.json").read_text(encoding="utf-8")
                    )
                except FileNotFoundError as exc:
                    problems.append(f"{tag}: {exc}")
                    continue

                epochs_run = len(hist)
                best = int(freeze["best_epoch"])
                health = numeric_health(hist, int(cfg["epochs"]), best)
                if health:
                    problems.append(f"{tag}: " + "; ".join(health))

                # The judge takes the anchor argmin from the curve minimum, so
                # that equality is a precondition, not a convention. Two cells
                # were reported with the wrong epoch earlier precisely because
                # this was assumed rather than asserted; verify it here so the
                # assumption is recorded rather than trusted.
                curve_vals = [float(r["val_engine_mean_rmse"]) for r in hist]
                curve_min = min(curve_vals)
                argmin_epoch = curve_vals.index(curve_min) + 1
                recorded = float(vmet["val_engine_mean_rmse"])
                selection_ok = (
                    abs(recorded - curve_min) < 1e-9 and argmin_epoch == best
                )
                if not selection_ok:
                    problems.append(
                        f"{tag}: recorded selection {recorded:.6f} vs curve min "
                        f"{curve_min:.6f} (argmin epoch {argmin_epoch} vs best {best})"
                    )

                cells.append(
                    {
                        "method": m,
                        "subset": s,
                        "seed": int(seed),
                        "protocol": {k: cfg.get(k, None) for k in PROTOCOL_FIELDS},
                        "protocol_sha256": protocol_sha256(cfg),
                        "training_data_sha256": cfg.get("training_data_sha256"),
                        "paired_pretraining_subset": cfg.get("paired_pretraining_subset"),
                        "pretraining_epochs": cfg.get("pretraining_epochs"),
                        "epochs_run": epochs_run,
                        "best_epoch": best,
                        "budget_truncated": epochs_run == int(cfg["epochs"]),
                        "curve_len": len(hist),
                        "recorded_selection": recorded,
                        "curve_min": curve_min,
                        "argmin_epoch": argmin_epoch,
                        "selection_is_curve_min": selection_ok,
                        "numeric_health": "ok" if not health else "; ".join(health),
                    }
                )

    truncated = [f"{c['method']}/{c['subset']}/s{c['seed']}" for c in cells if c["budget_truncated"]]
    per_method_epochs = {}
    for c in cells:
        per_method_epochs.setdefault(c["method"], []).append(c["epochs_run"])

    out = {
        "source": "frozen anchor cells (prereg v6 R2.1 ground truth)",
        "note": (
            "Compare each rerun cell to ITS OWN anchor row. Do not test for an "
            "absolute value: budget_truncated=true is legitimate for the cells "
            "listed below, and a 'hit the ceiling means broken' rule would "
            "false-flag them."
        ),
        "protocol_fields": list(PROTOCOL_FIELDS),
        "cells": cells,
        "budget_truncated_cells": truncated,
        "epochs_run_by_method": per_method_epochs,
        "selection_is_curve_min_all": all(c["selection_is_curve_min"] for c in cells),
        "selection_mismatches": [
            f"{c['method']}/{c['subset']}/s{c['seed']}"
            for c in cells
            if not c["selection_is_curve_min"]
        ],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"cells: {len(cells)}")
    print(f"distinct protocol_sha256: {len({c['protocol_sha256'] for c in cells})}")
    print(f"budget_truncated (legitimately at the ceiling): {len(truncated)} {truncated}")
    print("epochs_run by method:")
    for m, vals in sorted(per_method_epochs.items()):
        print(f"  {m:11} {vals}  max={max(vals)}")
    print(f"numeric health problems: {len([p for p in problems if 'numeric' in p or 'curve' in p or 'best_epoch' in p])}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 2
    print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
