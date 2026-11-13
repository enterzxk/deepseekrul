#!/usr/bin/env python3
"""Sprint-24h metric aggregation: one reproducible pass over every run dir.

Why this file exists
--------------------
The sprint produces one directory per (arm, subset, seed) with ``metrics.json``
and ``preds.csv``.  Comparing modules across those directories is only fair if
every number is recomputed the *same* way, from the same frozen definition, on
the same labels.  This script is that single pass.  It never trains, never
touches model code, and never selects on anything but the frozen validation
metric.

Frozen definitions (source of truth: ``paper_b/training/metrics.py`` and
``paper_b/hc/train_d.py::metrics_from_arrays``)
------------------------------------------------
* label capping:   ``ys = min(y_raw, cap)``; predictions clipped to ``[0, cap]``
* main metric:     ``engine_mean_rmse = mean_e rmse(pred_e, ys_e)`` -- the
                   engine is the statistical unit, 5 cut-cycles per engine
* pooled:          ``sqrt(mean((pred - ys)^2))`` over all points (reported, not
                   used for selection; always >= engine-mean by Jensen)
* NASA score:      asymmetric exponential, ``exp(-d/13) - 1`` for early
                   predictions (d < 0) and ``exp(d/10) - 1`` for late ones
* uncapped:        raw truth, predictions NOT clipped, only where ``y_raw <= cap``
                   (beyond the horizon the operator cannot represent the truth)

``cap`` is the frozen RUL cap (125).  It is not inferred from the data; it is
asserted against the run's own ``y_cap`` column so a silently different cap
cannot pass unnoticed.

Usage
-----
    python aggregate_metrics.py --runs-root ~/sprint24/experiments/hc/d \
                                --out-dir docs/paper_b_edc/sprint_24h

    # align against the experiment owner's ledger (single source of truth):
    python aggregate_metrics.py ... --run-list docs/paper_b_edc/sprint_24h/RUN_QUEUE.csv

Outputs
-------
``SEED_RESULTS.csv``    one row per (module, subset, seed)
``MODULE_RESULTS.csv``  one row per (module, subset), seed mean/sd
``PER_ENGINE.csv``      one row per (run_id, engine_id) -- paired analysis
``METRIC_AUDIT.md``     the 口径 check: recomputed vs reported, per run

Exit code is non-zero if any run's recomputed main metric disagrees with its
``metrics.json`` beyond ``--tol``; a mismatch means the reported number did not
come from the predictions on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# frozen protocol constants
# --------------------------------------------------------------------------- #
DEFAULT_CAP = 125.0          # rul_cap of the frozen protocol
DEFAULT_GLOB = "HC-D-*-D1"
RUN_RE = re.compile(r"^HC-D-(?P<arm>.+)-(?P<subset>FD00[1-4])-s(?P<seed>\d+)-D1$")

# display names: arm -> (module label, one-line description)
MODULE_LABELS = {
    "B0": ("B0_direct_rul", "direct RUL regression (Stage D baseline)"),
    "B1": ("B1_HI_fixed", "structured HI auxiliary supervision"),
    "B2": ("B2_HI_fixed_rate", "structured HI + fixed future rate"),
    "B3": ("B3_HI_var_rate", "structured HI + variable future rate"),
    "AGE": ("AGE_age_only", "age prior only, no trainable parameters"),
    "AAR": ("AAR_age_residual", "M1-B: age prior + zero-init sensor residual"),
    "M2A": ("M2_CDE_A", "CDE raw three-channel"),
    "M2B": ("M2_CDE_B", "CDE + condition-matched trend"),
    "M2C": ("M2_CDE_C", "CDE + trend + persistence"),
    "M2D": ("M2_CDE_D", "CDE + trend + evidence confidence"),
    "M3A": ("M3_ADC_A", "Stage D damage increment"),
    "M3B": ("M3_ADC_B", "age-anchored increment"),
    "M3C": ("M3_ADC_C", "age-anchored + same-age ranking"),
    "M3D": ("M3_ADC_D", "age-anchored + ranking + CDE"),
    "M4A": ("M4_CSR_A", "learned sensor pooling"),
    "M4B": ("M4_CSR_B", "original sensor attention"),
    "M4C": ("M4_CSR_C", "condition-aware pooling"),
    "M4D": ("M4_CSR_D", "condition + evidence reliability pooling"),
    "M5A": ("M5_DPC_A", "no DPC"),
    "M5B": ("M5_DPC_B", "single-span prediction"),
    "M5C": ("M5_DPC_C", "multi-span prediction"),
    "M5D": ("M5_DPC_D", "condition-matched multi-span"),
    "M6A": ("M6_EFR_A", "fixed future rate"),
    "M6B": ("M6_EFR_B", "free variable rate"),
    "M6C": ("M6_EFR_C", "age prior + fixed evidence correction"),
    "M6D": ("M6_EFR_D", "age prior + adaptive evidence correction"),
    "M6E": ("M6_EFR_E", "isoparametric cycle-concat control"),
    "M7A": ("M7_SCR_A", "Stage D original"),
    "M7B": ("M7_SCR_B", "Stage D + SCR"),
    "M7C": ("M7_SCR_C", "ADC + SCR"),
}


def module_of(arm: str) -> tuple[str, str]:
    return MODULE_LABELS.get(arm, (arm, ""))


# --------------------------------------------------------------------------- #
# metric primitives -- kept identical to paper_b/training/metrics.py
# --------------------------------------------------------------------------- #
def rmse(p, y) -> float:
    p = np.asarray(p, np.float64)
    y = np.asarray(y, np.float64)
    return float(np.sqrt(np.mean((p - y) ** 2))) if p.size else float("nan")


def mae(p, y) -> float:
    p = np.asarray(p, np.float64)
    y = np.asarray(y, np.float64)
    return float(np.mean(np.abs(p - y))) if p.size else float("nan")


def nasa_score(p, y) -> tuple[float, float]:
    d = np.asarray(p, np.float64) - np.asarray(y, np.float64)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s)), float(np.mean(s))


def per_engine_rmse(unit, y_true, y_pred) -> float:
    unit = np.asarray(unit)
    if unit.size == 0:
        return float("nan")
    vals = [rmse(y_pred[unit == u], y_true[unit == u]) for u in np.unique(unit)]
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def read_preds(path: str) -> dict[str, np.ndarray]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty predictions file: {path}")
    need = {"engine_id", "cut", "y_cap", "y_raw", "y_pred"}
    missing = need - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    out = {
        "engine_id": np.array([int(float(r["engine_id"])) for r in rows]),
        "cut": np.array([int(float(r["cut"])) for r in rows]),
        "y_cap": np.array([float(r["y_cap"]) for r in rows]),
        "y_raw": np.array([float(r["y_raw"]) for r in rows]),
        "y_pred": np.array([float(r["y_pred"]) for r in rows]),
    }
    if "split" in rows[0]:
        out["split"] = np.array([r["split"] for r in rows])
    return out


def score_run(run_dir: str, cap: float) -> dict[str, Any]:
    """Recompute every metric of one run from its predictions."""
    metrics_path = os.path.join(run_dir, "metrics.json")
    preds_path = os.path.join(run_dir, "preds.csv")
    with open(metrics_path) as fh:
        meta = json.load(fh)
    a = read_preds(preds_path)

    eng, cuts = a["engine_id"], a["cut"]
    y_raw, y_pred = a["y_raw"], a["y_pred"]
    ys = np.minimum(y_raw, cap)                     # frozen label
    clipped = np.clip(y_pred, 0.0, cap)             # frozen prediction transform

    # the cap assertion: the file's own y_cap column must equal min(y_raw, cap)
    cap_gap = float(np.max(np.abs(ys - a["y_cap"]))) if ys.size else 0.0
    if cap_gap > 1e-9:
        raise ValueError(
            f"{run_dir}: y_cap column disagrees with min(y_raw, cap={cap}) "
            f"by {cap_gap:.3e} -- the run used a different cap; pass --cap"
        )

    unc = y_raw < cap                                # censoring on the raw RUL
    capd = ~unc
    d = clipped - ys
    nasa_sum, nasa_mean = nasa_score(clipped, ys)

    row: dict[str, Any] = dict(
        n_points=int(ys.size),
        n_engines=int(np.unique(eng).size),
        n_cuts=int(np.unique(cuts).size),
        engine_mean_rmse=per_engine_rmse(eng, ys, clipped),
        pooled_rmse=rmse(clipped, ys),
        nasa_sum=nasa_sum,
        nasa_mean=nasa_mean,
        mae=mae(clipped, ys),
        bias=float(np.mean(d)),
        late_frac=float(np.mean(d > 0)),
        severe_late=float(np.mean(d > 10)),
        max_late=float(np.max(d)),
        frac_capped=float(capd.mean()),
        rmse_uncensored=(rmse(clipped[unc], y_raw[unc]) if unc.any() else float("nan")),
        rmse_uncapped=(rmse(y_pred[y_raw <= cap], y_raw[y_raw <= cap])
                       if (y_raw <= cap).any() else float("nan")),
        boundary_hit=(float(np.mean(y_pred[capd] >= cap)) if capd.any() else float("nan")),
        exact_boundary=(float(np.mean(np.abs(y_pred[capd] - cap) < 1e-6))
                        if capd.any() else float("nan")),
        cap_gap=cap_gap,
    )

    # per cut-cycle segment: cut 0 is the earliest (largest RUL), cut 4 the latest
    for c in range(int(np.unique(cuts).size)):
        m = cuts == c
        row[f"seg_rmse_{c}"] = rmse(clipped[m], ys[m])
        row[f"seg_mae_{c}"] = mae(clipped[m], ys[m])
        row[f"seg_bias_{c}"] = float(np.mean(clipped[m] - ys[m]))
        row[f"seg_rul_mean_{c}"] = float(np.mean(ys[m]))

    # provenance / budget
    params = meta.get("params") or {}
    budget = meta.get("budget") or {}
    code_fp = meta.get("code_fingerprint") or {}
    epochs_run = meta.get("epochs_run")
    best_ep = meta.get("best_ep")
    row.update(
        reported_engine_mean_rmse=meta.get("val_engine_mean_rmse"),
        reported_pooled_rmse=meta.get("val_pooled_rmse"),
        delta_vs_reported=(
            row["engine_mean_rmse"] - meta["val_engine_mean_rmse"]
            if meta.get("val_engine_mean_rmse") is not None else float("nan")
        ),
        best_ep=best_ep,
        epochs_run=epochs_run,
        budget_limited=(
            bool(best_ep is not None and epochs_run is not None and best_ep >= epochs_run)
        ),
        budget_epochs=budget.get("epochs"),
        budget_patience=budget.get("patience"),
        lr=budget.get("lr"),
        params_trainable=params.get("trainable"),
        params_backbone=params.get("backbone"),
        elapsed_s=meta.get("elapsed_s"),
        created_utc=meta.get("created_utc"),
        git_sha=code_fp.get("git_sha"),
        code_files_sha256=code_fp.get("files_sha256"),
        frozen_sha256=meta.get("frozen_sha256"),
        l_mean=meta.get("l_mean"),
    )
    row["_per_engine"] = per_engine_rows(eng, ys, clipped, run_dir)
    return row


def per_engine_rows(eng, ys, clipped, run_dir) -> list[dict[str, Any]]:
    out = []
    for u in np.unique(eng):
        m = eng == u
        out.append(dict(
            run_dir=run_dir,
            engine_id=int(u),
            n=int(m.sum()),
            rmse=rmse(clipped[m], ys[m]),
            mae=mae(clipped[m], ys[m]),
            bias=float(np.mean(clipped[m] - ys[m])),
            y_mean=float(np.mean(ys[m])),
            pred_mean=float(np.mean(clipped[m])),
        ))
    return out


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def parse_run_id(name: str) -> dict[str, Any] | None:
    m = RUN_RE.match(name)
    if not m:
        return None
    return dict(arm=m.group("arm"), subset=m.group("subset"), seed=int(m.group("seed")))


def discover(runs_root: str, pattern: str) -> list[dict[str, Any]]:
    found = []
    for name in sorted(os.listdir(runs_root)):
        path = os.path.join(runs_root, name)
        if not os.path.isdir(path):
            continue
        if not re.match(re.escape(pattern).replace(r"\*", ".*"), name):
            continue
        info = parse_run_id(name)
        if info is None:
            continue
        info["run_id"] = name
        info["run_dir"] = path
        found.append(info)
    return found


def load_run_list(path: str) -> dict[str, dict[str, str]]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    key = "run_id" if "run_id" in rows[0] else list(rows[0])[0]
    return {r[key].strip(): r for r in rows if r.get(key)}


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def mean_sd(vals: list[float]) -> tuple[float, float]:
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return float(statistics.mean(clean)), float(statistics.stdev(clean))


def write_csv(path: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


SEED_COLUMNS = [
    "run_id", "module", "arm", "subset", "seed", "status",
    "n_points", "n_engines", "n_cuts",
    "engine_mean_rmse", "pooled_rmse", "nasa_sum", "nasa_mean", "mae", "bias",
    "late_frac", "severe_late", "max_late",
    "rmse_uncensored", "rmse_uncapped", "frac_capped",
    "boundary_hit", "exact_boundary",
    "seg_rmse_0", "seg_rmse_1", "seg_rmse_2", "seg_rmse_3", "seg_rmse_4",
    "seg_rul_mean_0", "seg_rul_mean_1", "seg_rul_mean_2", "seg_rul_mean_3", "seg_rul_mean_4",
    "reported_engine_mean_rmse", "reported_pooled_rmse", "delta_vs_reported",
    "best_ep", "epochs_run", "budget_limited", "budget_epochs", "budget_patience", "lr",
    "params_trainable", "params_backbone", "elapsed_s", "created_utc",
    "git_sha", "code_files_sha256", "frozen_sha256", "l_mean",
    "preds_path", "in_ledger",
]

MODULE_COLUMNS = [
    "module", "arm", "subset", "n_seeds", "seeds",
    "engine_mean_rmse_mean", "engine_mean_rmse_sd", "engine_mean_rmse_min",
    "engine_mean_rmse_max", "engine_mean_rmse_cv_pct",
    "nasa_mean_mean", "mae_mean", "bias_mean",
    "rmse_uncensored_mean", "frac_capped_mean",
    "delta_vs_B0_abs", "delta_vs_B0_pct",
    "budget_limited_any", "best_ep_max", "epochs_run_max",
    "params_trainable", "status", "description",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.expanduser("~/sprint24/experiments/hc/d"))
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--run-list", default=None,
                    help="ledger CSV with a run_id column; marks missing/unlisted runs")
    ap.add_argument("--glob", default=DEFAULT_GLOB, help="run dir name pattern")
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP, help="frozen RUL cap")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="tolerance for recomputed vs reported main metric")
    args = ap.parse_args(argv)

    runs = discover(args.runs_root, args.glob)
    ledger = load_run_list(args.run_list) if args.run_list else None
    if ledger is not None:
        known = {r["run_id"] for r in runs}
        for rid, rec in ledger.items():
            if rid not in known:
                info = parse_run_id(rid)
                if info:
                    runs.append(dict(run_id=rid, run_dir=None, **info))

    seed_rows: list[dict[str, Any]] = []
    per_engine: list[dict[str, Any]] = []
    problems: list[str] = []

    for r in runs:
        rid = r["run_id"]
        module, desc = module_of(r["arm"])
        base = dict(
            run_id=rid, module=module, arm=r["arm"], subset=r["subset"],
            seed=r["seed"], description=desc,
            in_ledger=("yes" if (ledger is not None and rid in ledger) else
                       ("no" if ledger is not None else "n/a")),
        )
        if not r["run_dir"] or not os.path.isdir(r["run_dir"]):
            seed_rows.append({**base, "status": "missing"})
            continue
        metrics_path = os.path.join(r["run_dir"], "metrics.json")
        preds_path = os.path.join(r["run_dir"], "preds.csv")
        if not os.path.exists(preds_path):
            seed_rows.append({**base, "status": "no_preds",
                              "preds_path": preds_path})
            continue
        if not os.path.exists(metrics_path):
            seed_rows.append({**base, "status": "no_metrics",
                              "preds_path": preds_path})
            continue
        try:
            row = score_run(r["run_dir"], args.cap)
        except Exception as exc:                     # noqa: BLE001 - report, do not abort
            problems.append(f"{rid}: {exc}")
            seed_rows.append({**base, "status": f"error: {exc}", "preds_path": preds_path})
            continue
        pe = row.pop("_per_engine")
        for e in pe:
            per_engine.append({**e, "run_id": rid, "module": module,
                               "subset": r["subset"], "seed": r["seed"]})
        seed_rows.append({**base, **row, "status": "done", "preds_path": preds_path})
        if not math.isnan(row["delta_vs_reported"]) and abs(row["delta_vs_reported"]) > args.tol:
            problems.append(
                f"{rid}: recomputed engine_mean_rmse={row['engine_mean_rmse']:.9f} vs "
                f"reported={row['reported_engine_mean_rmse']:.9f} "
                f"(delta {row['delta_vs_reported']:.3e})"
            )

    # ---- module x subset aggregation -------------------------------------- #
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        if row.get("status") != "done":
            continue
        groups.setdefault((row["module"], row["subset"]), []).append(row)

    baseline: dict[str, float] = {}
    for (mod, subset), rows in groups.items():
        if mod == "B0_direct_rul":
            v, _ = mean_sd([x["engine_mean_rmse"] for x in rows])
            baseline[subset] = v

    module_rows: list[dict[str, Any]] = []
    for (mod, subset), rows in sorted(groups.items()):
        vals = [x["engine_mean_rmse"] for x in rows]
        mean, sd = mean_sd(vals)
        nasa_mean, _ = mean_sd([x["nasa_mean"] for x in rows])
        mae_mean, _ = mean_sd([x["mae"] for x in rows])
        bias_mean, _ = mean_sd([x["bias"] for x in rows])
        unc_mean, _ = mean_sd([x["rmse_uncensored"] for x in rows])
        capped_mean, _ = mean_sd([x["frac_capped"] for x in rows])
        b0 = baseline.get(subset)
        module_rows.append(dict(
            module=mod, arm=rows[0]["arm"], subset=subset,
            n_seeds=len(rows),
            seeds=";".join(str(x["seed"]) for x in sorted(rows, key=lambda y: y["seed"])),
            engine_mean_rmse_mean=mean, engine_mean_rmse_sd=sd,
            engine_mean_rmse_min=min(vals), engine_mean_rmse_max=max(vals),
            engine_mean_rmse_cv_pct=(100.0 * sd / mean if mean else float("nan")),
            nasa_mean_mean=nasa_mean, mae_mean=mae_mean, bias_mean=bias_mean,
            rmse_uncensored_mean=unc_mean, frac_capped_mean=capped_mean,
            delta_vs_B0_abs=(mean - b0 if b0 is not None else float("nan")),
            delta_vs_B0_pct=(100.0 * (mean - b0) / b0
                             if b0 not in (None, 0) else float("nan")),
            budget_limited_any=any(x["budget_limited"] for x in rows),
            best_ep_max=max((x["best_ep"] or 0) for x in rows),
            epochs_run_max=max((x["epochs_run"] or 0) for x in rows),
            params_trainable=rows[0]["params_trainable"],
            status="done", description=rows[0]["description"],
        ))

    os.makedirs(args.out_dir, exist_ok=True)
    seed_path = os.path.join(args.out_dir, "SEED_RESULTS.csv")
    module_path = os.path.join(args.out_dir, "MODULE_RESULTS.csv")
    engine_path = os.path.join(args.out_dir, "PER_ENGINE.csv")
    write_csv(seed_path, seed_rows, SEED_COLUMNS)
    write_csv(module_path, module_rows, MODULE_COLUMNS)
    write_csv(engine_path, per_engine,
              ["run_id", "module", "subset", "seed", "engine_id", "n",
               "rmse", "mae", "bias", "y_mean", "pred_mean"])

    # ---- audit note ------------------------------------------------------- #
    done = [r for r in seed_rows if r.get("status") == "done"]
    max_gap = max((abs(r["delta_vs_reported"]) for r in done
                   if r.get("delta_vs_reported") is not None), default=float("nan"))
    lines = [
        "# Metric audit -- recomputed vs reported",
        "",
        "Generated by `aggregate_metrics.py`; do not hand-edit.",
        "",
        f"- runs scored: **{len(done)}**",
        f"- runs in ledger but missing on disk: "
        f"**{sum(1 for r in seed_rows if r.get('status') == 'missing')}**",
        f"- runs on disk but not in ledger: "
        f"**{sum(1 for r in seed_rows if r.get('in_ledger') == 'no')}**",
        f"- max |recomputed - reported| engine-mean RMSE: **{max_gap:.3e}** "
        f"(tolerance {args.tol:g})",
        f"- frozen cap asserted: **{args.cap:g}**",
        "",
        "## Definition check",
        "",
        "| quantity | definition used here | source |",
        "|---|---|---|",
        "| main metric | mean over engines of RMSE(clip(pred,0,cap), min(y_raw,cap)) | "
        "`train_d.py::metrics_from_arrays` |",
        "| pooled | sqrt(mean((clip(pred,0,cap) - min(y_raw,cap))^2)) | `metrics.py::rmse` |",
        "| NASA | sum/mean of exp(-d/13)-1 (d<0), exp(d/10)-1 (d>=0) | "
        "`metrics.py::nasa_score` |",
        "| uncapped | RMSE(pred, y_raw) where y_raw <= cap, predictions unclipped | "
        "`metrics_from_arrays` |",
        "| segments | cut-cycle 0 (earliest, largest RUL) .. 4 (latest) | "
        "`metrics_from_arrays['per_cut']` |",
        "",
        "## Per-run check",
        "",
        "| run_id | n | engines | recomputed | reported | delta | NASA mean | MAE | bias | best_ep/epochs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(done, key=lambda x: x["run_id"]):
        lines.append(
            f"| {r['run_id']} | {r['n_points']} | {r['n_engines']} | "
            f"{r['engine_mean_rmse']:.6f} | {r['reported_engine_mean_rmse']:.6f} | "
            f"{r['delta_vs_reported']:.2e} | {r['nasa_mean']:.4f} | "
            f"{r['mae']:.4f} | {r['bias']:+.4f} | {r['best_ep']}/{r['epochs_run']} |"
        )
    if problems:
        lines += ["", "## Problems", ""] + [f"- {p}" for p in problems]
    with open(os.path.join(args.out_dir, "METRIC_AUDIT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"scored {len(done)} runs -> {seed_path}")
    print(f"module rows: {len(module_rows)} -> {module_path}")
    print(f"per-engine rows: {len(per_engine)} -> {engine_path}")
    print(f"max |delta vs reported| = {max_gap:.3e}")
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
