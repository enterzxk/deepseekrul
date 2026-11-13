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

Run identity
------------
``run_id`` is NOT unique: the 80-epoch probes live in a second root
(``probe80/``) and reuse the formal run ids.  Identity is therefore
``(root, run_id)``, and every aggregation keeps the root.

Usable runs only
----------------
A directory can exist and still not be evidence.  Three cases are excluded from
``MODULE_RESULTS`` and kept in ``SEED_RESULTS`` with a status:

* ``smoke_override`` -- ``budget_override`` is set (a 1-epoch smoke that landed
  in the formal root under a formal run id);
* ``interrupted``    -- training stopped before the budget without patience
  firing (``epochs_run - best_ep < patience`` and ``epochs_run < budget``);
* ``unusable``       -- the ledger says ``usable=no``.

``n/a_by_design`` (AGE: zero trainable parameters) and ``early_stopped``
(patience fired) are legitimate and DO enter the aggregation.

Usage
-----
    python aggregate_metrics.py --out-dir docs/paper_b_edc/sprint_24h

    # align against the experiment owner's ledger (single source of truth):
    python aggregate_metrics.py --run-list .../RUN_QUEUE.csv \
        --runs-root ~/sprint24/experiments/hc/d \
        --runs-root ~/sprint24/experiments/hc/d80 \
        --runs-root ~/sprint24/experiments/hc/d/probe80

Outputs
-------
``SEED_RESULTS.csv``    one row per (root, module, subset, seed)
``MODULE_RESULTS.csv``  one row per (module, subset, budget), seed mean/sd
``PER_ENGINE.csv``      one row per (run, engine) -- paired analysis
``METRIC_AUDIT.md``     the 口径 check: recomputed vs reported, per run

Exit code is non-zero if any run's recomputed main metric disagrees with its
``metrics.json`` beyond ``--tol``; a mismatch means the reported number did not
come from the predictions on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
DEFAULT_ROOTS = [
    "~/sprint24/experiments/hc/d",
    "~/sprint24/experiments/hc/d80",
    "~/sprint24/experiments/hc/d/probe80",
    # diagnostic roots: a run is not less real because it lives under a
    # different budget, and skipping these roots silently kept M3B's 160-epoch
    # result (11.88) out of every table.
    "~/sprint24/experiments/hc/d80diag",
    "~/sprint24/experiments/hc/d40diag",
    "~/sprint24/experiments/hc/d80b160",
    # reproduction-gate roots: scanned so their runs are recorded, flagged
    # archived so they never enter a statistic
    "~/sprint24/experiments/hc/_gate",
    "~/sprint24/experiments/hc/_gate160",
]
RUN_RE = re.compile(r"^HC-D-(?P<arm>.+)-(?P<subset>FD00[1-4])-s(?P<seed>\d+)-D1$")

# Roots whose runs are kept as records but must never enter statistics.  A
# reproduction gate re-runs a cell that already exists: if its predictions match
# it is a duplicate, and if they differ it silently doubles n_seeds and rewrites
# the very sd the gate was meant to validate.  Gates are therefore archived, not
# aggregated.
ARCHIVED_ROOTS = {"d80gate", "d80b160gate"}


def is_archived_root(root: str) -> bool:
    return root in ARCHIVED_ROOTS or os.path.basename(root.rstrip("/")).startswith("_")

# display names: arm -> module label.  Longest key wins, so "M2B" is matched
# before any shorter arm that happens to be a prefix.
MODULE_LABELS = {
    "B0": "B0_direct_rul",
    "B1": "B1_HI_fixed",
    "B2": "B2_HI_fixed_rate",
    "B3": "B3_HI_var_rate",
    "AGE": "AGE_age_only",
    "AAR": "AAR_age_residual",
}
MODULE_DESC = {
    "B0_direct_rul": "direct RUL regression (Stage D baseline)",
    "B1_HI_fixed": "structured HI auxiliary supervision",
    "B2_HI_fixed_rate": "structured HI + fixed future rate",
    "B3_HI_var_rate": "structured HI + variable future rate",
    "AGE_age_only": "age prior only, no trainable parameters",
    "AAR_age_residual": "M1-B: age prior + zero-init sensor residual",
}

# The primary comparison for a module is its ablation parent, not B0.  A module
# added on top of B3 has to be judged against B3: comparing it to B0 answers
# "is the structured-HI route good", which we already know, and hides "did this
# module help its parent".
MODULE_BASELINE = {
    "B1_HI_fixed": "B0_direct_rul",
    "B2_HI_fixed_rate": "B3_HI_var_rate",
    "B3_HI_var_rate": "B0_direct_rul",
    "AGE_age_only": "B0_direct_rul",
    "AAR_age_residual": "B0_direct_rul",
    "M2": "B0_direct_rul",
    "M3B": "B3_HI_var_rate",
    "M3C": "M3B",
    "M4": "B0_direct_rul",
    "M5": "B0_direct_rul",
    "M6": "B3_HI_var_rate",
    "M7": "B3_HI_var_rate",
}


def baseline_for(module: str) -> str:
    return MODULE_BASELINE.get(module, "B0_direct_rul")


def module_of(arm: str) -> str:
    """Map an arm name to its module label; unknown arms pass through as-is."""
    if arm in MODULE_LABELS:
        return MODULE_LABELS[arm]
    # fall back to a prefix match so M2A/M2B/... land under one module family
    for key in sorted(MODULE_LABELS, key=len, reverse=True):
        if arm.startswith(key):
            return MODULE_LABELS[key]
    return arm


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


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_fingerprint(run_dir: str) -> str:
    """Short hash of the run's per-file code fingerprint, or '' when absent.

    The gate cells are the reference: they were launched from the current HEAD, so
    a cell sharing their fingerprint was run by the same code.  A cell that does
    not share it is not wrong -- the gates exist precisely because some old cells
    turned out to be numerically identical across fingerprints -- but it has not
    been *shown* equivalent, and that difference is what `verified` records.
    """
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path) as fh:
            cf = (json.load(fh).get("code_fingerprint") or {})
    except Exception:                                 # noqa: BLE001
        return ""
    files = cf.get("files_sha256") or {}
    if not files:
        return ""
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]


def read_history(run_dir: str) -> list[float]:
    """Per-epoch validation curve, or [] when the run has no history file."""
    path = os.path.join(run_dir, "history.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            hist = json.load(fh)
    except Exception:                                 # noqa: BLE001
        return []
    out = []
    for h in hist:
        v = h.get("val_engine_mean_rmse")
        if v is not None:
            out.append(float(v))
    return out


EMA_ALPHA = 0.3


def ema_min(curve: list[float], alpha: float = EMA_ALPHA) -> float:
    """Minimum of the exponential moving average of the validation curve.

    A compromise between the two biased extremes: the raw minimum favours a
    noisy arm (a minimum order statistic), and the tail mean is distorted by
    patience (every arm stops about 12 epochs after its best).  Smoothing keeps
    the location of the dip but discounts a single lucky epoch.
    """
    if not curve:
        return float("nan")
    e = curve[0]
    best = e
    for x in curve[1:]:
        e = alpha * x + (1.0 - alpha) * e
        best = min(best, e)
    return float(best)


def curve_stats(curve: list[float]) -> dict[str, Any]:
    """Convergence-level statistics for one arm.

    ``best_ep`` is the minimum of the curve, i.e. a minimum order statistic: the
    noisier the arm, the lower its minimum is pushed, so best-vs-best comparisons
    systematically favour the unstable arm.  The tail mean measures the level the
    arm actually sits at once it stops improving, and the epoch-to-epoch sd says
    how much of that curve is noise.  Both are reported; neither replaces the
    frozen engine-mean RMSE, which still scores the selected checkpoint.
    """
    if not curve:
        return dict(n_epochs=0, tail10_mean=float("nan"), tail10_sd=float("nan"),
                    epoch_sd=float("nan"), best_val=float("nan"),
                    best_minus_tail10=float("nan"), ema_min=float("nan"),
                    gain_last1=float("nan"), gain_last3=float("nan"),
                    gain_last5=float("nan"), gain_last6=float("nan"),
                    gain_last12=float("nan"))
    tail = curve[-10:] if len(curve) >= 10 else curve
    best = min(curve)
    t_mean, t_sd = mean_sd(tail)

    def gain(k: int) -> float:
        """How much the best improved inside the final k epochs.

        ``best(1..B-k) - best(1..B)``: zero means the last k epochs bought
        nothing.  This is the number that decides whether a run that reached its
        budget without firing patience actually left anything on the table -- and
        it is measured, not inferred from the cosine formula.
        """
        if len(curve) <= k:
            return float("nan")
        return min(curve[:-k]) - best

    return dict(
        n_epochs=len(curve),
        tail10_mean=t_mean,
        tail10_sd=t_sd,
        epoch_sd=(float(statistics.stdev(curve)) if len(curve) > 1 else 0.0),
        best_val=best,
        best_minus_tail10=(t_mean - best),
        ema_min=ema_min(curve),
        gain_last1=gain(1),
        gain_last3=gain(3),
        gain_last5=gain(5),
        gain_last6=gain(6),
        gain_last12=gain(12),
    )


def score_run(run_dir: str, cap: float) -> dict[str, Any]:
    """Recompute every metric of one run from its predictions."""
    with open(os.path.join(run_dir, "metrics.json")) as fh:
        meta = json.load(fh)
    preds_path = os.path.join(run_dir, "preds.csv")
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
        preds_md5=file_md5(preds_path),
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
        budget_epochs=budget.get("epochs"),
        budget_patience=budget.get("patience"),
        budget_override=meta.get("budget_override"),
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
    row["fp_short"] = files_fingerprint(run_dir)
    # reproducible: an independent recomputation from preds.csv lands on the
    # number the run itself reported.  It says nothing about which code ran.
    d_rep = row.get("delta_vs_reported")
    row["reproducible"] = (
        "yes" if isinstance(d_rep, (int, float)) and d_rep == d_rep
        and abs(d_rep) <= 1e-6 else "no")
    row["status_detail"] = classify(row)
    ov = row.get("budget_override")
    er1 = row.get("epochs_run")
    if not ov:
        row["override_kind"] = ""
    elif er1 is None or er1 <= 1:
        row["override_kind"] = "smoke"
    elif ov.startswith(("m4fix", "m3crank")):
        row["override_kind"] = "retest"
    elif ov.startswith(("codecheck", "audit_")):
        row["override_kind"] = "audit"
    else:
        row["override_kind"] = "diag_budget"
    row.update(curve_stats(read_history(run_dir)))
    row["converged"] = (
        "yes" if (epochs_run is not None and row.get("budget_epochs") is not None
                  and epochs_run < row["budget_epochs"]) else "no")
    # A run that reached its budget without patience firing never turned around:
    # every one of the last `patience` epochs held an improvement, so `best_ep`
    # sits inside the final window and the reported score is a lower bound on a
    # converged score, not an estimate of it.  The obvious test
    # (`best_ep == epochs_run`) misses almost every such run: M3B-FD004-s101 is
    # 39/40 and slips through, while `epochs_run >= budget` catches it.  The two
    # columns stay separate because a truncated run is still evidence -- it just
    # cannot be compared against a converged run as if the two were the same
    # statistic.
    row["early_stop_margin"] = (
        None if (epochs_run is None or best_ep is None) else epochs_run - best_ep)
    row["truncated"] = (
        "yes" if (epochs_run is not None and row.get("budget_epochs") is not None
                  and epochs_run >= row["budget_epochs"]) else "")
    # The one column a reader needs in order to know which two rows may be
    # differenced.  A pair is only interpretable when both sides share a status:
    # comparing a converged run with a budget-limited one mixes two different
    # quantities and cannot be repaired by a wider threshold.
    sd0 = row.get("status_detail")
    if sd0 == "n/a_by_design":
        row["budget_status"] = "n/a"
    elif epochs_run is None or row.get("budget_epochs") is None:
        row["budget_status"] = "unknown"
    elif epochs_run >= row["budget_epochs"]:
        row["budget_status"] = "budget_limited"
    else:
        row["budget_status"] = "converged"
    row["_per_engine"] = per_engine_rows(eng, ys, clipped)
    return row


def classify(row: dict[str, Any]) -> str:
    """Decide whether a finished-looking directory is evidence.

    Order matters: a smoke override is never evidence even though its budget
    was met; a zero-parameter arm is legitimate even though it ran one epoch;
    and a one-epoch probe without an override is still a probe -- the
    ``budget_override`` guard alone does not catch it, because a probe launched
    by hand with ``--seed 999`` has no override at all.
    """
    if row.get("budget_override"):
        # budget_override started as the smoke marker, but it is now also used to
        # tag legitimate re-runs (m4fix3, m3b160, m4fd004).  A run that trained
        # past one epoch is not a smoke, whatever its override says -- excluding
        # it silently is how M4's 14.5396 disappeared and made the table look like
        # it had cherry-picked the good seed.
        er0 = row.get("epochs_run")
        if er0 is None or er0 <= 1:
            return "smoke_override"
        return "override_run"
    if row.get("params_trainable") == 0:
        return "n/a_by_design"
    be, er = row.get("best_ep"), row.get("epochs_run")
    budget = row.get("budget_epochs")
    patience = row.get("budget_patience") or 0
    if be is None or er is None:
        return "unknown"
    if er <= 1 and (budget is None or budget < 40):
        return "one_epoch_probe"
    # A run whose best epoch is 1 or 2 and then stops improving for a whole
    # patience window never trained: patience fired on a curve that only went
    # up.  That is a broken run, not a statement about the module.
    if be is not None and be <= 2 and er >= 5:
        return "degenerate_curve"
    if budget is not None and er >= budget:
        return "complete"
    if patience and (er - be) >= patience:
        return "early_stopped"
    return "interrupted"


def per_engine_rows(eng, ys, clipped) -> list[dict[str, Any]]:
    out = []
    for u in np.unique(eng):
        m = eng == u
        out.append(dict(
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
# discovery / ledger
# --------------------------------------------------------------------------- #
def parse_run_id(name: str) -> dict[str, Any] | None:
    m = RUN_RE.match(name)
    if not m:
        return None
    return dict(arm=m.group("arm"), subset=m.group("subset"), seed=int(m.group("seed")))


def discover(root: str, pattern: str) -> list[dict[str, Any]]:
    root = os.path.expanduser(root)
    label = os.path.basename(root.rstrip("/"))
    found = []
    if not os.path.isdir(root):
        return found
    rx = re.compile("^" + re.escape(pattern).replace(r"\*", ".*") + "$")
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not rx.match(name):
            continue
        info = parse_run_id(name)
        if info is None:
            continue
        info.update(run_id=name, run_dir=path, root=label, in_ledger="no",
                    archived=("yes" if is_archived_root(label) else ""))
        found.append(info)
    return found


def load_ledger(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def derive_root(run_dir: str, base: str) -> str:
    """Root label from the path, not from the ledger's ``root`` column.

    A ledger can carry ``root=d80`` while its ``run_dir`` points at ``d/`` (that
    happened: two rows did, and trusting the label invented a duplicate cell
    that did not exist on disk).  The directory is the identity, so the label
    has to follow the directory.
    """
    rel = os.path.relpath(os.path.realpath(run_dir), os.path.realpath(base))
    parts = rel.split(os.sep)
    if len(parts) >= 4 and parts[0] == "experiments" and parts[1] == "hc":
        return "/".join(parts[2:-1])
    return os.path.basename(os.path.dirname(rel))


# --------------------------------------------------------------------------- #
# aggregation helpers
# --------------------------------------------------------------------------- #
def mean_sd(vals: list[float]) -> tuple[float, float]:
    clean = [v for v in vals
             if v is not None and not (isinstance(v, float) and math.isnan(v))]
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
    "run_id", "root", "root_ledger", "module", "arm", "subset", "seed",
    "status", "status_detail", "in_ledger", "usable", "archived", "preds_md5",
    "n_points", "n_engines", "n_cuts",
    "engine_mean_rmse", "pooled_rmse", "nasa_sum", "nasa_mean", "mae", "bias",
    "late_frac", "severe_late", "max_late",
    "rmse_uncensored", "rmse_uncapped", "frac_capped",
    "boundary_hit", "exact_boundary",
    "seg_rmse_0", "seg_rmse_1", "seg_rmse_2", "seg_rmse_3", "seg_rmse_4",
    "seg_rul_mean_0", "seg_rul_mean_1", "seg_rul_mean_2", "seg_rul_mean_3", "seg_rul_mean_4",
    "reported_engine_mean_rmse", "reported_pooled_rmse", "delta_vs_reported",
    "best_ep", "epochs_run", "budget_epochs", "budget_patience", "budget_override", "override_kind",
    "converged", "truncated", "budget_status", "early_stop_margin",
    "reproducible", "verified", "verified_note", "comparable", "fp_short",
    "gain_last1", "gain_last3", "gain_last5", "gain_last6", "gain_last12",
    "n_epochs", "tail10_mean", "tail10_sd", "epoch_sd",
    "best_val", "best_minus_tail10", "ema_min",
    "lr", "params_trainable", "params_backbone", "elapsed_s", "created_utc",
    "git_sha", "code_files_sha256", "frozen_sha256", "l_mean",
    "run_dir", "preds_path",
]

MODULE_COLUMNS = [
    "module", "arm", "subset", "budget_epochs", "roots", "n_seeds", "seeds",
    "engine_mean_rmse_mean", "engine_mean_rmse_sd", "engine_mean_rmse_min",
    "engine_mean_rmse_max", "engine_mean_rmse_cv_pct",
    "engine_mean_rmse_se_over_seeds", "engine_mean_rmse_ci95_lo",
    "engine_mean_rmse_ci95_hi",
    "tail10_mean", "tail10_sd", "tail10_ci95_lo", "tail10_ci95_hi",
    "best_mean", "ema_min_mean", "epoch_sd_mean", "best_minus_tail10_mean",
    "tail10_delta_vs_B0_abs", "tail10_delta_vs_B0_pct",
    "best_delta_vs_B0_abs", "ema_delta_vs_B0_abs", "consistent",
    "nasa_mean_mean", "nasa_sum_mean", "mae_mean", "bias_mean",
    "rmse_uncensored_mean", "frac_capped_mean",
    "delta_vs_B0_abs", "delta_vs_B0_pct",
    "status_mix", "n_truncated", "n_budget_limited",
    "reproducible_cells", "verified_cells", "comparable_cells", "converged_cells",
    "best_ep_max", "epochs_run_max",
    "params_trainable", "parent", "params_parent", "params_delta", "params_pct", "description",
]

USABLE = {"complete", "early_stopped", "n/a_by_design", "override_run"}

# two-sided 95% t quantiles for small df; 1.96 (normal) beyond the table
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t975(df: int) -> float:
    return T975.get(df, 1.96)


def paired_stats(diffs: list[float]) -> dict[str, Any]:
    """Mean paired difference with its SE, t, and 95% CI."""
    n = len(diffs)
    if n == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), se=float("nan"),
                    t=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"))
    mean = float(statistics.mean(diffs))
    if n == 1:
        return dict(n=1, mean=mean, sd=0.0, se=float("nan"),
                    t=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"))
    sd = float(statistics.stdev(diffs))
    se = sd / math.sqrt(n)
    t = mean / se if se else float("nan")
    half = t975(n - 1) * se
    return dict(n=n, mean=mean, sd=sd, se=se, t=t,
                ci_lo=mean - half, ci_hi=mean + half)


def ensemble_vectors(seed_rows: list[dict[str, Any]], cap: float,
                     min_seeds: int = 2) -> dict[tuple[str, str, Any], dict[str, Any]]:
    """Ensemble prediction vector per cell, aligned on (engine_id, cut)."""
    groups: dict[tuple[str, str, Any], list[dict[str, Any]]] = {}
    for r in seed_rows:
        if r.get("status") != "usable" or not r.get("preds_path"):
            continue
        groups.setdefault(
            (r["module"], r["subset"], r.get("budget_epochs")), []).append(r)
    out: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for key, rows in groups.items():
        if len(rows) < min_seeds:
            continue
        try:
            per = []
            labels: dict[tuple[int, int], float] = {}
            for r in rows:
                a = read_preds(r["preds_path"])
                per.append({(int(e), int(c)): float(p) for e, c, p in
                            zip(a["engine_id"], a["cut"], a["y_pred"])})
                labels.update({(int(e), int(c)): float(y) for e, c, y in
                               zip(a["engine_id"], a["cut"], a["y_raw"])})
            keys = sorted(labels)
            if any(set(p) != set(keys) for p in per):
                continue
            preds = np.mean([np.array([p[k] for k in keys], np.float64)
                             for p in per], axis=0)
        except Exception:                             # noqa: BLE001
            continue
        out[key] = dict(
            keys=keys,
            preds=preds,
            ys=np.array([min(labels[k], cap) for k in keys], np.float64),
            unit=np.array([k[0] for k in keys]),
            n_seeds=len(rows))
    return out


def build_ensemble_vs(seed_rows: list[dict[str, Any]], cap: float,
                      baseline_module: str) -> list[dict[str, Any]]:
    """Ensemble against ensemble, paired per engine.

    This is the test the deployment question actually needs: the seed-level CI
    says how sure we are about a mean over trainings, but the shipped artefact is
    the averaged prediction, and its error is a different random variable.
    """
    vecs = ensemble_vectors(seed_rows, cap)
    out: list[dict[str, Any]] = []
    for (mod, subset, budget), mine in sorted(
            vecs.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        if mod == baseline_module:
            continue
        ref = vecs.get((baseline_module, subset, budget))
        if ref is None or set(mine["keys"]) != set(ref["keys"]):
            continue
        idx = {k: i for i, k in enumerate(mine["keys"])}
        pos = [idx[k] for k in ref["keys"]]
        units = np.unique(ref["unit"])
        diffs = []
        for u in units:
            m = ref["unit"] == u
            pm = np.clip(mine["preds"][pos][m], 0.0, cap)
            pb = np.clip(ref["preds"][m], 0.0, cap)
            ys = ref["ys"][m]
            diffs.append(rmse(pm, ys) - rmse(pb, ys))
        st = paired_stats(diffs)
        em_m = per_engine_rmse(ref["unit"], ref["ys"], np.clip(mine["preds"][pos], 0, cap))
        em_b = per_engine_rmse(ref["unit"], ref["ys"], np.clip(ref["preds"], 0, cap))
        out.append(dict(
            module=mod, subset=subset, budget_epochs=budget,
            n_seeds_module=mine["n_seeds"], n_seeds_b0=ref["n_seeds"],
            ensemble_module_em=em_m, ensemble_b0_em=em_b,
            diff_em=em_m - em_b,
            n_engines=st["n"], mean_diff=st["mean"], sd_diff=st["sd"],
            se=st["se"], t=st["t"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"],
            pct_vs_b0=(100.0 * (em_m - em_b) / em_b if em_b else float("nan")),
            significant=("yes" if (not math.isnan(st["ci_lo"])
                                   and (st["ci_lo"] > 0 or st["ci_hi"] < 0)) else "no"),
        ))
    return out


def cell_stats(seed_rows: list[dict[str, Any]],
               module: str) -> dict[tuple[str, Any], dict[str, Any]]:
    """Per-cell summary of one arm: n, sd, and the mean of each of the three readings."""
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for r in seed_rows:
        if r.get("status") != "usable" or r.get("module") != module:
            continue
        groups.setdefault((r["subset"], r.get("budget_epochs")), []).append(r)
    out = {}
    for key, rows in groups.items():
        best, sd_best = mean_sd([x["best_val"] for x in rows])
        ema, sd_ema = mean_sd([x["ema_min"] for x in rows])
        tail, sd_tail = mean_sd([x["tail10_mean"] for x in rows])
        em, sd = mean_sd([x["engine_mean_rmse"] for x in rows])
        out[key] = dict(n=len(rows), mean=em, sd=sd, best=best, ema=ema, tail=tail,
                        sd_best=sd_best, sd_ema=sd_ema, sd_tail=sd_tail,
                        rows={x["seed"]: x for x in rows})
    return out


def build_screen(seed_rows: list[dict[str, Any]],
                 resolution: list[dict[str, Any]],
                 baseline_module: str) -> list[dict[str, Any]]:
    """The screening table: one row per run plus one paired row per cell.

    ``baseline`` is the module's ablation parent, so a module added on top of B3
    is judged against B3.  ``threshold_kind`` declares the comparison: a single
    run against a multi-seed mean is not paired and gets the independent-run
    screen, while matched seeds get a paired half-width computed from the observed
    spread of the differences and the number of PAIRS.
    """
    def verdict(deltas: list[float], thresholds: list[float]) -> tuple[str, str]:
        """Direction and the metric that drove it.

        The two directions are deliberately NOT symmetric:
          * "worse" needs only ONE reading beyond its own width -- a single clear
            regression is enough to say an arm is not harmless;
          * "better" needs ALL THREE beyond their own widths -- one flattering
            reading must not carry a promotion.
        Each reading is compared with the width measured for that same reading.
        """
        if any(math.isnan(d) for d in deltas) or any(math.isnan(t) for t in thresholds):
            return "n/a", ""
        signs = {d > 0 for d in deltas if abs(d) >= 0.01}
        if len(signs) > 1:
            return "undetermined (metrics disagree)", ""
        if not signs:
            return "indistinguishable (flat)", ""
        names = ("best", "ema_min", "tail10")
        beyond = [n for n, d, t in zip(names, deltas, thresholds) if abs(d) > t]
        worse = True in signs
        if worse:
            return ("worse" if beyond else "indistinguishable"), ";".join(beyond)
        return ("better" if len(beyond) == 3 else "indistinguishable"), ";".join(beyond)

    def decide(v: str, n_evidence: int) -> tuple[str, str]:
        """Map a verdict to a decision, with the promotion floor enforced here.

        A promotion needs at least three seeds on the comparison.  With one or two
        the interval is computed from a single spread estimate (df=1 gives t=12.7),
        so a narrow width can be an artefact of two differences that happened to
        agree -- M2/FD003 flipped twice that way.  The floor is in code, not in a
        reviewer's memory.
        """
        if v == "better":
            if n_evidence < 3:
                return "hold", (f"would promote, but only {n_evidence} seed(s): "
                                "the floor is 3, a narrow width from 1-2 seeds is not evidence")
            return "promote", "three readings agree and the gap exceeds the threshold"
        if v == "worse":
            return "eliminate", "three readings agree and the gap exceeds the threshold"
        if v == "indistinguishable":
            return "eliminate", "gap within threshold: no evidence of gain"
        if v == "indistinguishable (flat)":
            return "eliminate", "all three readings flat against the parent"
        if v == "undetermined (metrics disagree)":
            return "eliminate", "best / ema_min / tail10 point in different directions"
        return "hold", "not comparable (missing or single reading)"

    stats = {m: cell_stats(seed_rows, m) for m in
             ({r["module"] for r in seed_rows if r.get("status") == "usable"}
              | set(MODULE_BASELINE.values()))}

    out: list[dict[str, Any]] = []
    for mod in sorted(stats):
        if mod == baseline_module or mod.startswith("B"):
            continue
        parent = baseline_for(mod)
        for key, mine in sorted(stats[mod].items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            subset, budget = key
            base = stats.get(parent, {}).get(key)
            used_parent = parent
            if base is None:
                # The ablation parent may not have been run at this budget (B3
                # only exists at 80 epochs, M3B was also run at 160).  Falling
                # back to the route-level baseline keeps the cell visible and
                # labels the fallback instead of dropping it.
                used_parent = baseline_module
                base = stats.get(baseline_module, {}).get(key)
            if base is None or base["n"] < 1:
                continue
            k = 2.0 * math.sqrt(2.0)
            thr3 = ([k * base["sd_best"], k * base["sd_ema"], k * base["sd_tail"]]
                    if base["n"] > 1 else [float("nan")] * 3)
            single_thr = thr3[0]
            for seed, row in sorted(mine["rows"].items()):
                # Prefer the parent run on the SAME seed: seed difficulty spans
                # 11.05-13.96 on B0, so comparing a single run to the parent's
                # mean mixes "the module helped" with "this seed was easy".
                same = base["rows"].get(seed)
                if same is not None:
                    d = [row["best_val"] - same["best_val"],
                         row["ema_min"] - same["ema_min"],
                         row["tail10_mean"] - same["tail10_mean"]]
                    cmp_kind = "same seed"
                else:
                    d = [row["best_val"] - base["best"], row["ema_min"] - base["ema"],
                         row["tail10_mean"] - base["tail"]]
                    cmp_kind = "vs parent mean"
                v, driver = verdict(d, thr3)
                dec, why = decide(v, base["n"])
                out.append(dict(
                    arm=mod, baseline=used_parent, subset=subset, seed=seed, budget=budget,
                    comparison=cmp_kind,
                    n_baseline=base["n"], sd_baseline=base["sd"],
                    threshold_kind="single-run screen", threshold=single_thr,
                    best=row["best_val"], ema_min=row["ema_min"],
                    tail10=row["tail10_mean"],
                    delta_best=d[0], delta_ema=d[1], delta_tail10=d[2],
                    consistent=("yes" if len({x > 0 for x in d if abs(x) >= 0.01}) <= 1
                                else "no"),
                    verdict=v, decision=dec, reason=why, decision_driver=driver,
                    threshold_ema=thr3[1], threshold_tail=thr3[2],
                    converged=row.get("converged")))

            shared = sorted(set(mine["rows"]) & set(base["rows"]))
            if len(shared) >= 2:
                db = [mine["rows"][s]["best_val"] - base["rows"][s]["best_val"] for s in shared]
                de = [mine["rows"][s]["ema_min"] - base["rows"][s]["ema_min"] for s in shared]
                dt = [mine["rows"][s]["tail10_mean"] - base["rows"][s]["tail10_mean"]
                      for s in shared]
                st = paired_stats(db)
                tf = t975(len(shared) - 1) / math.sqrt(len(shared))
                half3 = [tf * paired_stats(x)["sd"] for x in (db, de, dt)]
                half = half3[0]
                mean_d = [st["mean"], float(np.mean(de)), float(np.mean(dt))]
                v, driver = verdict(mean_d, half3)
                dec, why = decide(v, len(shared))
                # A paired interval from one or two pairs has df=1, where t=12.7:
                # it is so wide that it rules nothing out.  Too little data is not
                # a result, so neither promote nor eliminate is allowed -- the row
                # is held, and the reason says why it has no resolution.
                if len(shared) < 3 and v in ("indistinguishable", "undetermined (metrics disagree)"):
                    dec, why = "hold", (f"only {len(shared)} pair(s): half-widths "
                                        f"{half3[0]:.2f}/{half3[1]:.2f}/{half3[2]:.2f} "
                                        "are too wide to promote or eliminate")
                if v == "indistinguishable" and len({x > 0 for x in db}) > 1:
                    why += (f"; direction disagrees across seeds "
                            f"({sum(1 for x in db if x < 0)}/{len(db)} favour the module)")
                capped = [s for s in shared if mine["rows"][s].get("converged") == "no"]
                if capped:
                    why += (f"; {len(capped)}/{len(shared)} seed(s) hit the budget "
                            "without early stop, so those values are bounds")
                out.append(dict(
                    arm=mod, baseline=parent, subset=subset,
                    seed=f"paired({len(shared)})", budget=budget,
                    comparison="paired",
                    n_baseline=base["n"], sd_baseline=base["sd"],
                    threshold_kind="paired half-width", threshold=half,
                    threshold_ema=half3[1], threshold_tail=half3[2],
                    best=mean_sd([mine["rows"][s]["best_val"] for s in shared])[0],
                    ema_min=mean_sd([mine["rows"][s]["ema_min"] for s in shared])[0],
                    tail10=mean_sd([mine["rows"][s]["tail10_mean"] for s in shared])[0],
                    delta_best=mean_d[0], delta_ema=mean_d[1], delta_tail10=mean_d[2],
                    consistent=("yes" if len({x > 0 for x in mean_d if abs(x) >= 0.01}) <= 1
                                else "no"),
                    verdict=v, decision=dec, reason=why, decision_driver=driver,
                    converged=""))
    # When a cell has a paired row, that row is the module-level statement (it
    # uses every seed).  The per-seed rows are components; leaving them with
    # their own promote/eliminate invites picking whichever single row reads
    # best, which is exactly the failure the rules forbid.
    paired_cells = {(r["arm"], r["subset"], r["budget"]) for r in out
                    if str(r["seed"]).startswith("paired(")}
    for r in out:
        if (r["arm"], r["subset"], r["budget"]) in paired_cells and not str(r["seed"]).startswith("paired("):
            r["decision"] = "component"
            # A component label must not swallow the per-metric verdict: the row
            # can be clearly worse on the robust readings while the paired row has
            # no power, and both facts have to stay visible.
            parts = []
            for nm, dk, tk in (("best", "delta_best", "threshold"),
                               ("ema_min", "delta_ema", "threshold_ema"),
                               ("tail10", "delta_tail10", "threshold_tail")):
                dv, tv = r.get(dk), r.get(tk)
                if dv is None or tv is None or (isinstance(tv, float) and math.isnan(tv)):
                    parts.append(f"{nm} n/a")
                    continue
                parts.append(f"{nm} {dv:+.2f} vs {tv:.2f} "
                             f"{'out' if abs(dv) > tv else 'in'}")
            n_out = sum(1 for p in parts if p.endswith(" out"))
            r["reason"] = (f"component; module verdict is the paired row; "
                           + " | ".join(parts) + f" -> {n_out}/3 beyond")
    return out


def build_gate_check(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare each archived reproduction gate with the formal cell it repeats.

    A gate exists to prove the formal cell is reproducible from the current code.
    Two levels matter and they are not the same: identical ``preds_md5`` means the
    bytes came back, while a small value difference with a different md5 usually
    means the formatting moved (row order, decimals) rather than the numerics.
    Both are reported; only the value check can tell those apart.
    """
    formal: dict[tuple, dict[str, Any]] = {}
    for r in seed_rows:
        if r.get("status") == "usable":
            formal.setdefault(
                (r["module"], r["subset"], r["seed"], r.get("budget_epochs")), r)
    out = []
    for r in seed_rows:
        if r.get("status") != "archived":
            continue
        key = (r["module"], r["subset"], r["seed"], r.get("budget_epochs"))
        ref = formal.get(key)
        if ref is None:
            out.append(dict(gate_run=r["run_id"], gate_root=r["root"], module=r["module"],
                            subset=r["subset"], seed=r["seed"],
                            budget_epochs=r.get("budget_epochs"),
                            formal_run="", formal_root="",
                            gate_md5=r.get("preds_md5"), formal_md5="",
                            md5_match="", delta_value=float("nan"),
                            verdict="no formal twin"))
            continue
        same_md5 = (r.get("preds_md5") == ref.get("preds_md5"))
        dv = (r.get("engine_mean_rmse") - ref.get("engine_mean_rmse")
              if r.get("engine_mean_rmse") is not None else float("nan"))
        if same_md5:
            verdict = "PASS (identical bytes)"
        elif not math.isnan(dv) and abs(dv) < 1e-3:
            verdict = "PASS (value identical, bytes differ: formatting)"
        else:
            verdict = "FAIL (numerics differ)"
        out.append(dict(
            gate_run=r["run_id"], gate_root=r["root"], module=r["module"],
            subset=r["subset"], seed=r["seed"], budget_epochs=r.get("budget_epochs"),
            formal_run=ref["run_id"], formal_root=ref["root"],
            gate_md5=r.get("preds_md5"), formal_md5=ref.get("preds_md5"),
            md5_match=("yes" if same_md5 else "no"), delta_value=dv,
            gate_epochs=r.get("epochs_run"), formal_epochs=ref.get("epochs_run"),
            epochs_match=("yes" if r.get("epochs_run") == ref.get("epochs_run") else "no"),
            verdict=verdict))
    return out


def build_budget_pairs(per_engine: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same arm, same seed, two budgets -- is the extra budget worth it?

    A budget change also changes the LR schedule, so this compares two whole
    training recipes rather than "more steps".  Pairing on seed and engine keeps
    the comparison honest: the seed difficulty cancels and the CI is over seeds,
    which is the interval a decision needs.
    """
    runs: dict[tuple[str, str, int, Any], dict[int, float]] = {}
    for e in per_engine:
        key = (e["module"], e["subset"], int(e["seed"]), e.get("budget_epochs"))
        runs.setdefault(key, {})[e["engine_id"]] = e["rmse"]

    cells: dict[tuple[str, str], dict[int, dict[Any, dict[int, float]]]] = {}
    for (mod, subset, seed, budget), per in runs.items():
        cells.setdefault((mod, subset), {}).setdefault(seed, {})[budget] = per

    out: list[dict[str, Any]] = []
    for (mod, subset), by_seed in sorted(cells.items()):
        budgets = sorted({b for d in by_seed.values() for b in d},
                         key=lambda b: (b is None, b))
        for lo, hi in zip(budgets, budgets[1:]):
            seed_diffs: list[float] = []
            seed_txt: list[str] = []
            for seed, d in sorted(by_seed.items()):
                if lo not in d or hi not in d:
                    continue
                a, b = d[lo], d[hi]
                common = sorted(set(a) & set(b))
                if not common:
                    continue
                diff = float(np.mean([b[i] - a[i] for i in common]))
                seed_diffs.append(diff)
                seed_txt.append(f"{seed}:{diff:+.3f}")
            if not seed_diffs:
                continue
            st = paired_stats(seed_diffs)
            out.append(dict(
                module=mod, subset=subset, budget_lo=lo, budget_hi=hi,
                n_seeds=len(seed_diffs),
                seeds=";".join(seed_txt),
                mean_diff=st["mean"], sd_diff=st["sd"], se=st["se"],
                t=st["t"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"],
                verdict=("higher budget worse" if st["mean"] > 0
                         else "higher budget better"),
                significant=("yes" if (not math.isnan(st["ci_lo"])
                                       and (st["ci_lo"] > 0 or st["ci_hi"] < 0)) else "no"),
            ))
    return out


def mad(vals: list[float]) -> float:
    """Median absolute deviation: ``median(|x - median(x)|)``.

    This is the definition the sprint's stability claim uses -- B0 0.3596 against
    M2 0.1387 is 2.592x, the "2.59x" quoted in the documents. The *mean* absolute
    deviation of the same two cells is 0.7801 against 0.2910, i.e. 2.681x. The two
    statistics differ by 3% here and by much more elsewhere, so the definition
    must be stated wherever the number is quoted: mixing them silently produces a
    ratio that reproduces neither.
    """
    if not vals:
        return float("nan")
    m = float(statistics.median(vals))
    return float(statistics.median([abs(v - m) for v in vals]))


def build_stability(seed_rows: list[dict[str, Any]],
                    baseline_module: str) -> list[dict[str, Any]]:
    """Seed dispersion per cell, with MAD ratios gated on equal sample size.

    MAD is the most outlier-sensitive dispersion here: with four points a single
    bad seed carries 1/4 of the weight, with six only 1/6.  In this sprint the
    four-seed B0 median absolute deviation is 0.7302 against 0.3596 at six seeds,
    so the same comparison reads 5.26x instead of 2.59x -- **a mismatch of sample
    size alone doubles the ratio, in the direction the claim wants it**.

    The general statement is the one to keep: the error is not random.  Its size
    and sign depend on which seeds happen to be missing, so an n-mismatched ratio
    is neither a conservative estimate nor a noisy one -- it is a number whose
    bias cannot be bounded at all, which is why no amount of care in the prose
    substitutes for refusing to print it.

    ``comparison_kind`` separates the two things this table now holds: a
    ``baseline_selfcheck`` row compares the baseline with itself and is 1.0 by
    construction -- it tests the arithmetic, not a module; only
    ``module_vs_baseline`` rows are evidence about a module.
    """
    groups: dict[tuple[str, str, Any], list[dict[str, Any]]] = {}
    for r in seed_rows:
        if r.get("status") != "usable":
            continue
        groups.setdefault((r["module"], r["subset"], r.get("budget_epochs")), []).append(r)

    base_n: dict[tuple[str, Any], int] = {}
    base_mad: dict[tuple[str, Any], float] = {}
    base_sd: dict[tuple[str, Any], float] = {}
    for (mod, subset, budget), rs in groups.items():
        if mod != baseline_module:
            continue
        vals = [x["engine_mean_rmse"] for x in rs]
        base_n[(subset, budget)] = len(vals)
        base_mad[(subset, budget)] = mad(vals)
        base_sd[(subset, budget)] = mean_sd(vals)[1]

    out = []
    for (mod, subset, budget), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], str(kv[0][2]), kv[0][0])):
        vals = [x["engine_mean_rmse"] for x in rs]
        n = len(vals)
        m, sd = mean_sd(vals)
        bn = base_n.get((subset, budget))
        bmad = base_mad.get((subset, budget), float("nan"))
        bsd = base_sd.get((subset, budget), float("nan"))
        same_n = (bn is not None and n == bn)
        md = mad(vals)
        if same_n and n >= 3 and bmad:
            ratio_ok, ratio, reason = "yes", md / bmad, ""
        elif bn is None:
            ratio_ok, ratio, reason = "no", float("nan"), "no baseline cell at this budget"
        else:
            ratio_ok = "no"
            ratio = float("nan")
            reason = ("n mismatch: %d vs baseline %d -- MAD is outlier-driven, "
                      "so this ratio would be an artefact of sample size" % (n, bn))
        out.append(dict(
            module=mod, subset=subset, budget_epochs=budget, n_seeds=n,
            comparison_kind=("baseline_selfcheck" if mod == baseline_module
                             else "module_vs_baseline"),
            seeds=";".join(sorted(str(x["seed"]) for x in rs)),
            mean=m, sd=sd, mad=md,
            baseline_n=bn, baseline_sd=bsd, baseline_mad=bmad,
            sd_ratio=(sd / bsd if bsd else float("nan")),
            mad_ratio=ratio, mad_ratio_ok=ratio_ok, mad_ratio_reason=reason,
            # A selfcheck ratio is 1.0 by construction -- the same numbers
            # compared with themselves.  It tests the arithmetic, not a module,
            # so it must never be counted as evidence for one.
            informative=("yes" if ratio_ok == "yes" and mod != baseline_module
                         else "no"),
        ))
    return out


def build_resolution(seed_rows: list[dict[str, Any]],
                     baseline_module: str) -> list[dict[str, Any]]:
    """How small a difference can this cell actually resolve?

    Derived from the baseline's own seed spread, so the thresholds move with the
    data instead of being hand-carried constants:

    * ``screen_single`` -- two independent single-seed runs of equal noise differ
      with sd ``sqrt(2)*sd``; 2 sd is the practical screen.  A module whose
      single-seed delta is below this cannot be promoted on one seed.
    * ``ci_mean`` -- the baseline's own 3-seed mean interval.
    * ``paired_3seed`` -- an arm-vs-arm difference averaged over the same seeds
      cancels the shared seed difficulty, so it is ``sqrt(2)*sd/sqrt(n)`` wide.

    All three scale with ``sd``, which is per-cell: a module with a larger spread
    gets a wider threshold.  A single global constant is therefore wrong.
    """
    groups: dict[tuple[str, Any], list[float]] = {}
    for r in seed_rows:
        if r.get("status") != "usable" or r.get("module") != baseline_module:
            continue
        groups.setdefault((r["subset"], r.get("budget_epochs")), []).append(
            r["engine_mean_rmse"])
    out = []
    for (subset, budget), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        n = len(vals)
        mean, sd = mean_sd(vals)
        if n < 2:
            out.append(dict(subset=subset, budget_epochs=budget, n_seeds=n,
                            mean=mean, sd=float("nan"), ci_mean=float("nan"),
                            screen_single=float("nan"), paired_3seed=float("nan"),
                            sd_first3=float("nan"), screen_first3=float("nan"),
                            paired_first3=float("nan")))
            continue
        t = t975(n - 1)
        first3 = vals[:3]                      # the thresholds the sprint started with
        _, sd3 = mean_sd(first3) if len(first3) > 1 else (float("nan"), float("nan"))
        out.append(dict(
            subset=subset, budget_epochs=budget, n_seeds=n, mean=mean, sd=sd,
            ci_mean=t * sd / math.sqrt(n),
            screen_single=2.0 * math.sqrt(2.0) * sd,
            paired_3seed=t * math.sqrt(2.0) * sd / math.sqrt(n),
            sd_first3=sd3,
            screen_first3=(2.0 * math.sqrt(2.0) * sd3 if len(first3) > 1 else float("nan")),
            paired_first3=(t975(len(first3) - 1) * math.sqrt(2.0) * sd3 / math.sqrt(len(first3))
                           if len(first3) > 1 else float("nan"))))
    return out


def build_verdict_shift(module_rows: list[dict[str, Any]],
                        resolution: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Which verdicts change when the seed variance is estimated properly?

    The sprint's thresholds came from three seeds.  With five the baseline sd is
    4.4x larger, so a gap that was 'clearly worse' under the old width can be
    'indistinguishable' under the new one.  Listing those flips is the honest
    accounting: it shows how an underestimated seed variance manufactures
    significant-looking conclusions.
    """
    res = {(r["subset"], r["budget_epochs"]): r for r in resolution}

    def verdict(delta: float, half: float) -> str:
        if math.isnan(half):
            return "n/a"
        if abs(delta) <= half:
            return "indistinguishable"
        return "worse" if delta > 0 else "better"

    out = []
    for m in module_rows:
        if m["module"] == "B0_direct_rul":
            continue
        r = res.get((m["subset"], m["budget_epochs"]))
        if r is None or math.isnan(r["paired_first3"]):
            continue
        delta = m["engine_mean_rmse_mean"] - r["mean"]
        # The right width depends on the comparison, not on habit.  A single run
        # against a multi-seed mean is NOT paired: its error carries the run's own
        # seed noise plus the uncertainty of the baseline mean, so the screen for
        # independent single runs applies.  Only when several seeds are matched
        # one-to-one does the paired width apply.
        single = m["n_seeds"] < 2
        half_old = r["screen_first3"] if single else r["paired_first3"]
        half_new = r["screen_single"] if single else r["paired_3seed"]
        if not single:
            # paired width for THIS many pairs, not for the baseline's seed count
            n_mod = m["n_seeds"]
            half_old = t975(min(n_mod, 3) - 1) * math.sqrt(2.0) * r["sd_first3"] / math.sqrt(n_mod)
            half_new = t975(n_mod - 1) * math.sqrt(2.0) * r["sd"] / math.sqrt(n_mod)
        v_old = verdict(delta, half_old)
        v_new = verdict(delta, half_new)
        out.append(dict(
            module=m["module"], subset=m["subset"], budget_epochs=m["budget_epochs"],
            n_seeds_module=m["n_seeds"], n_seeds_b0=r["n_seeds"],
            comparison=("single vs mean" if single else "paired"),
            module_mean=m["engine_mean_rmse_mean"], b0_mean=r["mean"],
            delta=delta,
            half_first3=half_old, half_all=half_new,
            verdict_first3=v_old, verdict_all=v_new,
            flips=("yes" if v_old != v_new else "no"),
        ))
    return sorted(out, key=lambda r: (r["flips"] != "yes", r["module"], r["subset"]))


def build_ensemble(seed_rows: list[dict[str, Any]], cap: float) -> list[dict[str, Any]]:
    """Seed-ensemble baseline: average predictions of the same cell over seeds.

    Averaging happens on the raw predictions, then the frozen clip is applied
    once, so the ensemble is scored exactly like a single run.  Labels are
    checked to be identical across seeds -- if they differ, the runs are not the
    same split and averaging them would be meaningless.

    The pairwise columns exist to answer a different question from the headline
    gain: if a 2-seed ensemble already reaches the 3-seed value, the gain is
    saturated and adding seeds buys nothing.
    """
    groups: dict[tuple[str, str, Any], list[dict[str, Any]]] = {}
    for r in seed_rows:
        if r.get("status") != "usable" or not r.get("preds_path"):
            continue
        groups.setdefault(
            (r["module"], r["subset"], r.get("budget_epochs")), []).append(r)

    out: list[dict[str, Any]] = []
    for (mod, subset, budget), rows in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        if len(rows) < 2:
            continue
        rows.sort(key=lambda x: x["seed"])
        per_seed: list[dict[tuple[int, int], float]] = []
        labels: dict[tuple[int, int], float] = {}
        engines: set[int] = set()
        cuts: set[int] = set()
        try:
            for r in rows:
                a = read_preds(r["preds_path"])
                d = {(int(e), int(c)): float(p)
                     for e, c, p in zip(a["engine_id"], a["cut"], a["y_pred"])}
                lab = {(int(e), int(c)): float(y)
                       for e, c, y in zip(a["engine_id"], a["cut"], a["y_raw"])}
                if labels and set(lab) != set(labels):
                    raise ValueError("different (engine_id, cut) sets across seeds")
                for k, v in lab.items():
                    if k in labels and abs(labels[k] - v) > 1e-9:
                        raise ValueError(f"label mismatch at {k}: {labels[k]} vs {v}")
                labels.update(lab)
                per_seed.append(d)
                engines.update(e for e, _ in d)
                cuts.update(c for _, c in d)
        except Exception as exc:                      # noqa: BLE001
            out.append(dict(module=mod, subset=subset, budget_epochs=budget,
                            n_seeds=len(rows), status=f"error: {exc}"))
            continue

        keys = sorted(labels)
        ys = np.array([min(labels[k], cap) for k in keys], np.float64)
        unit = np.array([k[0] for k in keys])

        def score(preds: np.ndarray) -> dict[str, float]:
            clipped = np.clip(preds, 0.0, cap)
            n_sum, n_mean = nasa_score(clipped, ys)
            return dict(em=per_engine_rmse(unit, ys, clipped), nasa_mean=n_mean,
                        mae=mae(clipped, ys), bias=float(np.mean(clipped - ys)))

        seed_ems = [r["engine_mean_rmse"] for r in rows]
        mean_em, sd_em = mean_sd(seed_ems)
        ens = np.mean([np.array([s[k] for k in keys], np.float64) for s in per_seed], axis=0)
        ens_stats = score(ens)

        # pairwise ensembles: the saturation check
        pair_ems: list[float] = []
        for i in range(len(per_seed)):
            for j in range(i + 1, len(per_seed)):
                p = (np.array([per_seed[i][k] for k in keys], np.float64)
                     + np.array([per_seed[j][k] for k in keys], np.float64)) / 2.0
                pair_ems.append(score(p)["em"])
        p_mean, p_sd = mean_sd(pair_ems)

        out.append(dict(
            module=mod, subset=subset, budget_epochs=budget,
            n_seeds=len(rows),
            seeds=";".join(str(r["seed"]) for r in rows),
            seed_ems=";".join(f"{v:.4f}" for v in seed_ems),
            mean_seed_em=mean_em, sd_seed_em=sd_em,
            ensemble_em=ens_stats["em"],
            gain_abs=(mean_em - ens_stats["em"]),
            gain_pct=(100.0 * (mean_em - ens_stats["em"]) / mean_em if mean_em else float("nan")),
            pair_mean_em=p_mean, pair_sd_em=p_sd,
            pair_min_em=(min(pair_ems) if pair_ems else float("nan")),
            pair_max_em=(max(pair_ems) if pair_ems else float("nan")),
            ensemble_nasa_mean=ens_stats["nasa_mean"],
            ensemble_mae=ens_stats["mae"],
            ensemble_bias=ens_stats["bias"],
            n_engines=len(engines), n_cuts=len(cuts), n_points=len(keys),
            status="ok",
        ))
    return out


HEADROOM_TOL = 0.2


def judge_pair(module_status: Any, baseline_status: Any,
               module_gain5: Any, baseline_gain5: Any) -> tuple[str, str, float]:
    """Can these two rows be differenced, judged by measured headroom?

    An early stop that did not fire looks alarming, but the label alone says
    nothing about how much was left on the table.  ``gain_last5`` measures that
    directly: ``best(1..B-5) - best(1..B)``.  A pair whose residual headroom is
    negligible on both sides measures the same thing whatever its formal status,
    because the cosine schedule has already annealed the LR to ``eta_min`` by the
    last epoch.

    The gate is therefore quantitative, not a binary label: a pair is usable when
    neither side gained more than ``HEADROOM_TOL`` inside its final five epochs.
    ``budget_status`` stays as a description of how the run ended.
    """
    if any(s in (None, "", "unknown", "n/a") for s in (module_status, baseline_status)):
        return "unknown", "no", float("nan")
    gains = [g for g in (module_gain5, baseline_gain5)
             if isinstance(g, (int, float)) and g == g]
    if not gains:
        return "unknown", "no", float("nan")
    headroom = max(gains)
    if headroom > HEADROOM_TOL:
        return "headroom_high", "no", headroom
    return "same_schedule", "yes", headroom


def build_paired(seed_rows: list[dict[str, Any]],
                 per_engine: list[dict[str, Any]],
                 baseline_module: str) -> list[dict[str, Any]]:
    """Same-seed, same-engine paired comparison against the baseline arm.

    The unpaired SE answers "how noisy is this quantity"; the paired SE answers
    "is the difference stable when seed difficulty cancels".  Both are needed,
    and with three seeds only the engine-level pairing has usable degrees of
    freedom.
    """
    by_run: dict[tuple[str, str], dict[int, float]] = {}
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for e in per_engine:
        key = (e["root"], e["run_id"])
        by_run.setdefault(key, {})[e["engine_id"]] = e["rmse"]
    for r in seed_rows:
        if r.get("status") != "usable":
            continue
        meta[(r["root"], r["run_id"])] = dict(
            module=r["module"], subset=r["subset"], seed=r["seed"],
            root=r["root"],
            budget=r.get("budget_epochs"),
            bs=r.get("budget_status"),
            g5=r.get("gain_last5"),
            g12=r.get("gain_last12"),
            em=r["engine_mean_rmse"])

    # index the baseline by (subset, budget, seed)
    base_idx: dict[tuple[str, Any, int], tuple[str, str]] = {}
    for key, m in meta.items():
        if m["module"] == baseline_module:
            base_idx[(m["subset"], m["budget"], m["seed"])] = key

    out: list[dict[str, Any]] = []
    for key, m in meta.items():
        if m["module"] == baseline_module:
            continue
        bkey = base_idx.get((m["subset"], m["budget"], m["seed"]))
        if bkey is None or bkey not in by_run or key not in by_run:
            continue
        mine, theirs = by_run[key], by_run[bkey]
        common = sorted(set(mine) & set(theirs))
        if not common:
            continue
        diffs = [mine[i] - theirs[i] for i in common]
        st = paired_stats(diffs)
        b0_em = meta[bkey]["em"]
        pair_status, judgeable, headroom = judge_pair(
            m["bs"], meta[bkey]["bs"], m["g5"], meta[bkey]["g5"])
        out.append(dict(
            level="seed",
            module=m["module"], subset=m["subset"], budget_epochs=m["budget"],
            seed=m["seed"], root=m["root"], n_seeds=1, seeds=str(m["seed"]),
            n_engines=st["n"], mean_diff=st["mean"], sd_diff=st["sd"],
            se=st["se"], t=st["t"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"],
            module_em=m["em"], b0_em=b0_em,
            module_budget_status=m["bs"], baseline_budget_status=meta[bkey]["bs"],
            module_gain_last5=m["g5"], baseline_gain_last5=meta[bkey]["g5"],
            module_gain_last12=m["g12"], baseline_gain_last12=meta[bkey]["g12"],
            headroom=headroom,
            headroom12=max([g for g in (m["g12"], meta[bkey]["g12"])
                            if isinstance(g, (int, float)) and g == g], default=float("nan")),
            headroom12_over_effect=(
                max([g for g in (m["g12"], meta[bkey]["g12"])
                     if isinstance(g, (int, float)) and g == g], default=float("nan"))
                / abs(st["mean"]) if st["mean"] else float("nan")),
            pair_status=pair_status, judgeable=judgeable,
            pct_vs_b0=(100.0 * st["mean"] / b0_em if b0_em else float("nan")),
            verdict=("worse" if st["mean"] > 0 else "better"),
            significant=("yes" if (st["ci_lo"] is not None and not math.isnan(st["ci_lo"])
                                   and (st["ci_lo"] > 0 or st["ci_hi"] < 0)) else "no"),
        ))

    # ---- cross-seed roll-up: the second variance source ------------------- #
    groups: dict[tuple[str, str, Any], list[dict[str, Any]]] = {}
    for r in out:
        groups.setdefault((r["module"], r["subset"], r["budget_epochs"]), []).append(r)
    for (mod, subset, budget), rows in groups.items():
        seed_means = [r["mean_diff"] for r in rows]
        st = paired_stats(seed_means)                 # df = n_seeds - 1
        b0_em, _ = mean_sd([r["b0_em"] for r in rows])
        # A roll-up inherits the weakest link: one confounded seed pair makes the
        # whole module-level difference uninterpretable, however many clean pairs
        # sit beside it.
        statuses = {r["pair_status"] for r in rows}
        if "headroom_high" in statuses:
            pair_status = "headroom_high"
        elif len(statuses) == 1:
            only = next(iter(statuses))
            pair_status = only if only == "same_schedule" else "unknown"
        else:
            pair_status = "unknown"
        judgeable = ("yes" if pair_status == "same_schedule"
                     and all(r["judgeable"] == "yes" for r in rows) else "no")
        heads = [r["headroom"] for r in rows
                 if isinstance(r.get("headroom"), (int, float)) and r["headroom"] == r["headroom"]]
        headroom = max(heads) if heads else float("nan")
        out.append(dict(
            level="module",
            module=mod, subset=subset, budget_epochs=budget,
            seed="", root=";".join(sorted({r["root"] for r in rows})),
            n_seeds=len(rows),
            seeds=";".join(str(r["seed"]) for r in sorted(rows, key=lambda x: x["seed"])),
            n_engines=rows[0]["n_engines"],
            mean_diff=st["mean"], sd_diff=st["sd"], se=st["se"], t=st["t"],
            ci_lo=st["ci_lo"], ci_hi=st["ci_hi"],
            module_em=float("nan"), b0_em=b0_em,
            module_budget_status=";".join(sorted({str(r["module_budget_status"]) for r in rows})),
            baseline_budget_status=";".join(sorted({str(r["baseline_budget_status"]) for r in rows})),
            module_gain_last5=max((r["module_gain_last5"] for r in rows
                                   if isinstance(r.get("module_gain_last5"), (int, float))
                                   and r["module_gain_last5"] == r["module_gain_last5"]),
                                  default=float("nan")),
            baseline_gain_last5=max((r["baseline_gain_last5"] for r in rows
                                     if isinstance(r.get("baseline_gain_last5"), (int, float))
                                     and r["baseline_gain_last5"] == r["baseline_gain_last5"]),
                                    default=float("nan")),
            headroom=headroom,
            module_gain_last12=max((r["module_gain_last12"] for r in rows
                                    if isinstance(r.get("module_gain_last12"), (int, float))
                                    and r["module_gain_last12"] == r["module_gain_last12"]),
                                   default=float("nan")),
            baseline_gain_last12=max((r["baseline_gain_last12"] for r in rows
                                      if isinstance(r.get("baseline_gain_last12"), (int, float))
                                      and r["baseline_gain_last12"] == r["baseline_gain_last12"]),
                                     default=float("nan")),
            headroom12=max((r["headroom12"] for r in rows
                            if isinstance(r.get("headroom12"), (int, float))
                            and r["headroom12"] == r["headroom12"]), default=float("nan")),
            headroom12_over_effect=max((r["headroom12_over_effect"] for r in rows
                                        if isinstance(r.get("headroom12_over_effect"), (int, float))
                                        and r["headroom12_over_effect"] == r["headroom12_over_effect"]),
                                       default=float("nan")),
            pair_status=pair_status, judgeable=judgeable,
            pct_vs_b0=(100.0 * st["mean"] / b0_em if b0_em else float("nan")),
            verdict=("worse" if st["mean"] > 0 else "better"),
            significant=("yes" if (st["ci_lo"] is not None and not math.isnan(st["ci_lo"])
                                   and (st["ci_lo"] > 0 or st["ci_hi"] < 0)) else "no"),
        ))
    return sorted(out, key=lambda r: (r["module"], r["subset"], str(r["budget_epochs"]),
                                      r["level"] != "module", r["seed"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", action="append", default=None,
                    help="run root; repeatable (default: the three sprint roots)")
    ap.add_argument("--base", default="~/sprint24",
                    help="base dir that ledger run_dir paths are relative to")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--run-list", default=None,
                    help="ledger CSV (RUN_QUEUE.csv); authoritative run list")
    ap.add_argument("--glob", default=DEFAULT_GLOB, help="run dir name pattern")
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP, help="frozen RUL cap")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="tolerance for recomputed vs reported main metric")
    ap.add_argument("--no-unlisted", action="store_true",
                    help="ignore runs found on disk but absent from the ledger")
    ap.add_argument("--paired-baseline", default="B0_direct_rul",
                    help="module used as the paired baseline (default B0)")
    args = ap.parse_args(argv)

    roots = [os.path.expanduser(r) for r in (args.runs_root or DEFAULT_ROOTS)]
    base = os.path.expanduser(args.base)

    # ---- build the work list: ledger first, then unlisted disk runs -------- #
    work: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    noncanon: list[dict[str, str]] = []
    dir_owner: dict[str, str] = {}
    root_mismatch: list[str] = []

    if args.run_list:
        for rec in load_ledger(args.run_list):
            rid = (rec.get("run_id") or "").strip()
            if not rid:
                continue
            info = parse_run_id(rid)
            if info is None:
                # not silently dropped: a hand-named directory can hold a real
                # number that nobody should ever aggregate under a formal arm
                noncanon.append(rec)
                continue
            ledger_root = (rec.get("root") or "formal").strip()
            run_dir = (rec.get("run_dir") or "").strip()
            if run_dir:
                # the ledger's path is authoritative: if it does not exist yet
                # (a running cell), the run is not there.  Searching other roots
                # for the same run id would silently substitute a *different*
                # cell -- d80/HC-D-B0-FD004-s101 is running while
                # d/HC-D-B0-FD004-s101 is finished, and they are not the same run.
                run_dir = os.path.join(base, run_dir)
                if not os.path.isdir(run_dir):
                    run_dir = None
            else:
                for r in roots:
                    cand = os.path.join(r, rid)
                    if os.path.isdir(cand):
                        run_dir = cand
                        break
            root = derive_root(run_dir, base) if run_dir else ledger_root
            if run_dir and root != ledger_root:
                root_mismatch.append(dict(
                    run_id=rid, ledger_root=ledger_root, path_root=root,
                    rel=os.path.relpath(run_dir, base)))
            dup_of = None
            if run_dir:
                rp = os.path.realpath(run_dir)
                if rp in dir_owner:
                    dup_of = dir_owner[rp]
                else:
                    dir_owner[rp] = f"{root}/{rid}"
            work.append(dict(
                run_id=rid, run_dir=run_dir, root=root, root_ledger=ledger_root,
                dup_of=dup_of, in_ledger="yes",
                archived=("yes" if (is_archived_root(root)
                                    or (rec.get("archived") or "").strip().lower() == "yes")
                          else ""),
                ledger_usable=(rec.get("usable") or "yes").strip().lower(),
                ledger_status=(rec.get("status") or "").strip(), **info))
            seen.add((root, rid))

    if not args.no_unlisted:
        # Identity is the directory, not the run id: probe80/ reuses formal run
        # ids, so a run_id-only dedup would either drop or double-count runs.
        ledger_dirs = {os.path.realpath(w["run_dir"])
                       for w in work if w.get("run_dir")}
        for r in roots:
            for info in discover(r, args.glob):
                if os.path.realpath(info["run_dir"]) in ledger_dirs:
                    continue
                if (info["root"], info["run_id"]) in seen:
                    continue
                info["ledger_usable"] = None
                info["ledger_status"] = ""
                work.append(info)
                seen.add((info["root"], info["run_id"]))

    # ---- score ------------------------------------------------------------ #
    seed_rows: list[dict[str, Any]] = []
    per_engine: list[dict[str, Any]] = []
    problems: list[str] = []

    for w in work:
        module = module_of(w["arm"])
        base_row = dict(
            run_id=w["run_id"], root=w["root"], module=module, arm=w["arm"],
            subset=w["subset"], seed=w["seed"], in_ledger=w["in_ledger"],
            archived=w.get("archived") or "",
            root_ledger=w.get("root_ledger"),
            usable=("" if w["ledger_usable"] is None else w["ledger_usable"]),
        )
        run_dir = w.get("run_dir")
        if w.get("dup_of"):
            seed_rows.append({**base_row, "status": "duplicate_ledger_row",
                              "status_detail": f"duplicate_ledger_row: same directory as {w['dup_of']}",
                              "run_dir": run_dir})
            continue
        if not run_dir or not os.path.isdir(run_dir):
            # a queued/running cell has no directory yet; say so instead of
            # calling it missing
            pending = (w.get("ledger_status") or "").lower()
            status = pending if pending in ("running", "queued") else "missing"
            seed_rows.append({**base_row, "status": status,
                              "status_detail": status, "run_dir": run_dir})
            continue
        metrics_path = os.path.join(run_dir, "metrics.json")
        preds_path = os.path.join(run_dir, "preds.csv")
        if not os.path.exists(metrics_path):
            seed_rows.append({**base_row, "status": "no_metrics",
                              "status_detail": "no_metrics", "run_dir": run_dir})
            continue
        if not os.path.exists(preds_path):
            seed_rows.append({**base_row, "status": "no_preds",
                              "status_detail": "no_preds", "run_dir": run_dir})
            continue
        try:
            row = score_run(run_dir, args.cap)
        except Exception as exc:                     # noqa: BLE001 - report, do not abort
            problems.append(f"{w['root']}/{w['run_id']}: {exc}")
            seed_rows.append({**base_row, "status": f"error",
                              "status_detail": f"error: {exc}", "run_dir": run_dir})
            continue

        detail = row.pop("status_detail")
        # the ledger's usable flag wins over the local heuristic
        if w.get("archived") == "yes":
            # still scored, so the gate can be compared with its formal twin,
            # but never aggregated
            status = "archived"
        elif w["ledger_usable"] == "no":
            status = "unusable"
        else:
            status = "usable" if detail in USABLE else "excluded"
        pe = row.pop("_per_engine")
        for e in pe:
            per_engine.append({**e, "run_id": w["run_id"], "root": w["root"],
                               "module": module, "subset": w["subset"],
                               "seed": w["seed"], "budget_epochs": row.get("budget_epochs")})
        seed_rows.append({**base_row, **row, "status": status,
                          "status_detail": detail,
                          "run_dir": run_dir, "preds_path": preds_path})
        if not math.isnan(row["delta_vs_reported"]) and abs(row["delta_vs_reported"]) > args.tol:
            problems.append(
                f"{w['root']}/{w['run_id']}: recomputed {row['engine_mean_rmse']:.9f} vs "
                f"reported {row['reported_engine_mean_rmse']:.9f} "
                f"(delta {row['delta_vs_reported']:.3e})")

    # ---- module x subset x budget ---------------------------------------- #
    for rec in noncanon:
        rid = (rec.get("run_id") or "").strip()
        # A smoke that was moved aside keeps its trace under the _smoke_ prefix.
        # Calling that "non canonical" would be true but misleading: the reason
        # it is not evidence is that it is a smoke.
        smoke = rid.startswith("_smoke_")
        seed_rows.append(dict(
            run_id=rid,
            root=(rec.get("root") or "").strip(),
            module=module_of((rec.get("arm") or "").strip()),
            arm=(rec.get("arm") or "").strip(),
            subset=(rec.get("subset") or "").strip(),
            seed=(rec.get("seed") or "").strip(),
            in_ledger="yes", usable=(rec.get("usable") or "").strip().lower(),
            status="smoke_dir" if smoke else "non_canonical_id",
            status_detail="smoke_dir" if smoke else "non_canonical_id",
            run_dir=(rec.get("run_dir") or "").strip(),
            reported_engine_mean_rmse=rec.get("val_engine_mean_rmse"),
        ))

    # ---- one number per artifact, then one number per cell ----------------- #
    # First net: identical bytes are a copy, whatever the label says.  Second
    # net: two different artifacts claiming one cell is a conflict -- there is
    # no way to tell which number is real, so neither is used.
    ROOT_PREF = {"d": 0, "d80": 1, "d/probe80": 2, "probe80": 3}
    artifacts: dict[str, list[dict[str, Any]]] = {}
    for r in seed_rows:
        if r.get("status") != "usable" or not r.get("preds_md5"):
            continue
        artifacts.setdefault(r["preds_md5"], []).append(r)
    for _md5, rows in artifacts.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: (ROOT_PREF.get(r["root"], 9),
                                 str(r.get("created_utc") or "")))
        keep = rows[0]
        for r in rows[1:]:
            r["status"] = "duplicate_artifact"
            r["status_detail"] = (
                f"duplicate_artifact: identical preds.csv as {keep['root']}/{keep['run_id']}")

    cells: dict[tuple, list[dict[str, Any]]] = {}
    for r in seed_rows:
        if r.get("status") != "usable":
            continue
        cells.setdefault(
            (r["module"], r["subset"], r["seed"], r.get("budget_epochs")), []).append(r)
    for _key, rows in cells.items():
        if len(rows) < 2:
            for r in rows:
                r["status_detail"] = r.get("status_detail") or "ok"
            continue
        for r in rows:
            r["status"] = "cell_conflict"
            r["status_detail"] = (
                "cell_conflict: same (module,subset,seed,budget), different predictions")
    live = {(r["root"], r["run_id"]) for r in seed_rows if r.get("status") == "usable"}
    per_engine = [e for e in per_engine if (e["root"], e["run_id"]) in live]

    # ---- the four independent axes ---------------------------------------- #
    # reproducible / verified / comparable / converged are independent: a number
    # can be any combination.  Keeping them apart is what turns tonight's
    # arguments ("is 2.59 verified or just reproducible?", "is -0.713 confounded
    # by budget?") into a table lookup instead of a discussion.
    # `verified` is NOT "same fingerprint as the gate".  The gates exist because
    # a different fingerprint can still be numerically identical -- G1 proved
    # exactly that -- so the test is a formal twin: a run in a `_gate` root with
    # the same (module, subset, seed, budget) and the same predictions.  That is
    # the only sense in which an old cell is known to be current-HEAD equivalent.
    gate_twins: dict[tuple, dict[str, Any]] = {}
    for r in seed_rows:
        if str(r.get("root", "")).startswith("_gate") and r.get("preds_md5"):
            gate_twins[(r["module"], r["subset"], r.get("budget_epochs"), r["seed"])] = r
    for r in seed_rows:
        twin = gate_twins.get((r["module"], r["subset"], r.get("budget_epochs"), r["seed"]))
        if twin is None or twin is r:
            r["verified"] = "no"
            r["verified_note"] = "no gate twin"
            continue
        md5_ok = twin.get("preds_md5") == r.get("preds_md5")
        v_a, v_b = twin.get("engine_mean_rmse"), r.get("engine_mean_rmse")
        val_ok = (v_a is not None and v_b is not None and abs(v_a - v_b) <= 1e-6)
        ep_ok = twin.get("epochs_run") == r.get("epochs_run")
        be_ok = twin.get("best_ep") == r.get("best_ep")
        # All four, not just the bytes.  Early stopping has drifted independently
        # of the values before, so a matching md5 with a different epochs_run
        # would mean the two runs are not the same experiment even though they
        # produced the same predictions.
        ok = md5_ok and val_ok and ep_ok and be_ok
        r["verified"] = "yes" if ok else "no"
        r["verified_note"] = ("md5+val+ep+best" if ok else
                              "partial: " + "+".join(
                                  n for n, f in (("md5", md5_ok), ("val", val_ok),
                                                 ("ep", ep_ok), ("best", be_ok)) if f))

    # comparable: the parent has a cell for this subset at this budget, so the
    # two rows can be differenced under one schedule.
    present = {(r["module"], r["subset"], r.get("budget_epochs"))
               for r in seed_rows if r.get("status") == "usable"}
    for r in seed_rows:
        parent = baseline_for(r["module"])
        r["comparable"] = (
            "yes" if (parent, r["subset"], r.get("budget_epochs")) in present
            else "no")

    groups: dict[tuple[str, str, Any], list[dict[str, Any]]] = {}
    for row in seed_rows:
        if row.get("status") != "usable":
            continue
        groups.setdefault(
            (row["module"], row["subset"], row.get("budget_epochs")), []).append(row)

    baseline: dict[tuple[str, Any], float] = {}
    baseline_tail: dict[tuple[str, Any], float] = {}
    baseline_best: dict[tuple[str, Any], float] = {}
    baseline_ema: dict[tuple[str, Any], float] = {}
    for (mod, subset, budget), rows in groups.items():
        if mod == "B0_direct_rul":
            v, _ = mean_sd([x["engine_mean_rmse"] for x in rows])
            baseline[(subset, budget)] = v
            baseline_tail[(subset, budget)], _ = mean_sd(
                [x["tail10_mean"] for x in rows])
            baseline_best[(subset, budget)], _ = mean_sd([x["best_val"] for x in rows])
            baseline_ema[(subset, budget)], _ = mean_sd([x["ema_min"] for x in rows])

    module_rows: list[dict[str, Any]] = []
    for (mod, subset, budget), rows in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        vals = [x["engine_mean_rmse"] for x in rows]
        mean, sd = mean_sd(vals)
        nasa_mean, _ = mean_sd([x["nasa_mean"] for x in rows])
        nasa_sum, _ = mean_sd([x["nasa_sum"] for x in rows])
        mae_mean, _ = mean_sd([x["mae"] for x in rows])
        bias_mean, _ = mean_sd([x["bias"] for x in rows])
        unc_mean, _ = mean_sd([x["rmse_uncensored"] for x in rows])
        capped_mean, _ = mean_sd([x["frac_capped"] for x in rows])
        b0 = baseline.get((subset, budget))
        tail_mean, tail_sd = mean_sd([x["tail10_mean"] for x in rows])
        tail_se = (tail_sd / math.sqrt(len(rows))) if len(rows) > 1 else float("nan")
        tail_half = (t975(len(rows) - 1) * tail_se if len(rows) > 1 else float("nan"))
        b0_tail = baseline_tail.get((subset, budget))
        best_mean, _ = mean_sd([x["best_val"] for x in rows])
        ema_mean, _ = mean_sd([x["ema_min"] for x in rows])
        d_best = best_mean - baseline_best.get((subset, budget), float("nan"))
        d_ema = ema_mean - baseline_ema.get((subset, budget), float("nan"))
        d_tail = tail_mean - b0_tail if b0_tail is not None else float("nan")
        # The three readings are biased in different directions, so a conclusion
        # needs the same sign in all three.  Flat (within 0.01) counts as no
        # direction, and the baseline is not compared with itself.
        deltas = [d_best, d_ema, d_tail]
        if mod == "B0_direct_rul":
            consistent = "n/a (baseline)"
        elif any(math.isnan(d) for d in deltas):
            consistent = "unknown"
        elif all(abs(d) < 0.01 for d in deltas):
            consistent = "flat"
        elif all(d > 0 for d in deltas):
            consistent = "yes (worse)"
        elif all(d < 0 for d in deltas):
            consistent = "yes (better)"
        else:
            consistent = "no (flips)"
        # cross-seed uncertainty: the second variance source.  A single-seed
        # engine-level CI says nothing about it, and with 3 seeds df=2 the
        # interval is wide -- that width is the honest picture, not a defect.
        se_seeds = (sd / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        half = t975(len(vals) - 1) * se_seeds if len(vals) > 1 else float("nan")
        module_rows.append(dict(
            module=mod, arm=rows[0]["arm"], subset=subset, budget_epochs=budget,
            roots=";".join(sorted({x["root"] for x in rows})),
            n_seeds=len(rows),
            seeds=";".join(str(x["seed"]) for x in sorted(rows, key=lambda y: y["seed"])),
            engine_mean_rmse_mean=mean, engine_mean_rmse_sd=sd,
            engine_mean_rmse_min=min(vals), engine_mean_rmse_max=max(vals),
            engine_mean_rmse_cv_pct=(100.0 * sd / mean if mean else float("nan")),
            engine_mean_rmse_se_over_seeds=se_seeds,
            engine_mean_rmse_ci95_lo=mean - half,
            engine_mean_rmse_ci95_hi=mean + half,
            tail10_mean=tail_mean, tail10_sd=tail_sd,
            tail10_ci95_lo=tail_mean - tail_half,
            tail10_ci95_hi=tail_mean + tail_half,
            epoch_sd_mean=mean_sd([x["epoch_sd"] for x in rows])[0],
            best_minus_tail10_mean=mean_sd([x["best_minus_tail10"] for x in rows])[0],
            best_mean=best_mean, ema_min_mean=ema_mean,
            best_delta_vs_B0_abs=d_best,
            ema_delta_vs_B0_abs=d_ema,
            consistent=consistent,
            tail10_delta_vs_B0_abs=d_tail,
            tail10_delta_vs_B0_pct=(100.0 * (tail_mean - b0_tail) / b0_tail
                                    if b0_tail not in (None, 0) else float("nan")),
            nasa_mean_mean=nasa_mean, nasa_sum_mean=nasa_sum,
            mae_mean=mae_mean, bias_mean=bias_mean,
            rmse_uncensored_mean=unc_mean, frac_capped_mean=capped_mean,
            delta_vs_B0_abs=(mean - b0 if b0 is not None else float("nan")),
            delta_vs_B0_pct=(100.0 * (mean - b0) / b0
                             if b0 not in (None, 0) else float("nan")),
            status_mix=";".join(sorted({x["status_detail"] for x in rows})),
            n_truncated=sum(1 for x in rows if x.get("truncated") == "yes"),
            n_budget_limited=sum(1 for x in rows
                                 if x.get("budget_status") == "budget_limited"),
            reproducible_cells="%d/%d" % (
                sum(1 for x in rows if x.get("reproducible") == "yes"), len(rows)),
            verified_cells="%d/%d" % (
                sum(1 for x in rows if x.get("verified") == "yes"), len(rows)),
            comparable_cells="%d/%d" % (
                sum(1 for x in rows if x.get("comparable") == "yes"), len(rows)),
            converged_cells="%d/%d" % (
                sum(1 for x in rows if x.get("budget_status") == "converged"), len(rows)),
            best_ep_max=max((x["best_ep"] or 0) for x in rows),
            epochs_run_max=max((x["epochs_run"] or 0) for x in rows),
            params_trainable=rows[0]["params_trainable"],
            parent=baseline_for(mod),
            description=MODULE_DESC.get(mod, ""),
        ))

    # Parameter cost is only meaningful against the module's own parent: the
    # arms sit on three different bases (B0, B3, M3B), so one global denominator
    # produces numbers that are arithmetically right and conceptually wrong.
    params_by_module: dict[str, float] = {}
    for r in module_rows:
        p = r.get("params_trainable")
        if p is not None:
            params_by_module.setdefault(r["module"], p)
    for r in module_rows:
        parent = r["parent"]
        p = r.get("params_trainable")
        pp = params_by_module.get(parent)
        if p is None or pp in (None, 0):
            r["params_parent"] = pp
            r["params_delta"] = float("nan")
            r["params_pct"] = float("nan")
            continue
        r["params_parent"] = pp
        r["params_delta"] = p - pp
        r["params_pct"] = 100.0 * (p - pp) / pp

    os.makedirs(args.out_dir, exist_ok=True)
    seed_path = os.path.join(args.out_dir, "SEED_RESULTS.csv")
    module_path = os.path.join(args.out_dir, "MODULE_RESULTS.csv")
    engine_path = os.path.join(args.out_dir, "PER_ENGINE.csv")
    paired_path = os.path.join(args.out_dir, "PAIRED_VS_B0.csv")
    write_csv(seed_path, seed_rows, SEED_COLUMNS)
    write_csv(module_path, module_rows, MODULE_COLUMNS)
    write_csv(engine_path, per_engine,
              ["run_id", "root", "module", "subset", "budget_epochs", "seed",
               "engine_id", "n", "rmse", "mae", "bias", "y_mean", "pred_mean"])
    paired = build_paired(seed_rows, per_engine, args.paired_baseline)
    write_csv(paired_path, paired,
              ["level", "module", "subset", "budget_epochs", "seed", "root",
               "n_seeds", "seeds", "n_engines",
               "mean_diff", "sd_diff", "se", "t", "ci_lo", "ci_hi",
               "module_em", "b0_em", "pct_vs_b0", "verdict", "significant",
               "module_budget_status", "baseline_budget_status",
               "module_gain_last5", "baseline_gain_last5", "headroom",
               "module_gain_last12", "baseline_gain_last12", "headroom12",
               "headroom12_over_effect",
               "pair_status", "judgeable"])
    ensemble = build_ensemble(seed_rows, args.cap)
    # stage-2 bar: a module is only a contribution if its ensemble beats the
    # baseline ensemble, not the baseline mean -- averaging seeds is free.
    base_ens = {(e["subset"], e["budget_epochs"]): e["ensemble_em"]
                for e in ensemble if e.get("module") == args.paired_baseline
                and e.get("status") == "ok"}
    for e in ensemble:
        ref = base_ens.get((e.get("subset"), e.get("budget_epochs")))
        if ref and e.get("status") == "ok" and e.get("module") != args.paired_baseline:
            e["vs_b0_ensemble_abs"] = e["ensemble_em"] - ref
            e["vs_b0_ensemble_pct"] = 100.0 * (e["ensemble_em"] - ref) / ref
        else:
            e["vs_b0_ensemble_abs"] = float("nan")
            e["vs_b0_ensemble_pct"] = float("nan")
    write_csv(os.path.join(args.out_dir, "ENSEMBLE_RESULTS.csv"), ensemble,
              ["module", "subset", "budget_epochs", "n_seeds", "seeds",
               "seed_ems", "mean_seed_em", "sd_seed_em", "ensemble_em",
               "gain_abs", "gain_pct", "pair_mean_em", "pair_sd_em",
               "pair_min_em", "pair_max_em",
               "vs_b0_ensemble_abs", "vs_b0_ensemble_pct",
               "ensemble_nasa_mean", "ensemble_mae", "ensemble_bias",
               "n_engines", "n_cuts", "n_points", "status"])
    resolution = build_resolution(seed_rows, args.paired_baseline)
    stability = build_stability(seed_rows, args.paired_baseline)
    write_csv(os.path.join(args.out_dir, "STABILITY.csv"), stability,
              ["module", "subset", "budget_epochs", "n_seeds", "comparison_kind", "seeds",
               "mean", "sd", "mad", "baseline_n", "baseline_sd", "baseline_mad",
               "sd_ratio", "mad_ratio", "mad_ratio_ok", "mad_ratio_reason",
               "informative"])
    write_csv(os.path.join(args.out_dir, "RESOLUTION.csv"), resolution,
              ["subset", "budget_epochs", "n_seeds", "mean", "sd",
               "ci_mean", "screen_single", "paired_3seed",
               "sd_first3", "screen_first3", "paired_first3"])
    shift = build_verdict_shift(module_rows, resolution)
    write_csv(os.path.join(args.out_dir, "VERDICT_SHIFT.csv"), shift,
              ["module", "subset", "budget_epochs", "n_seeds_module", "n_seeds_b0",
               "comparison", "module_mean", "b0_mean", "delta",
               "half_first3", "half_all",
               "verdict_first3", "verdict_all", "flips"])
    screen = build_screen(seed_rows, resolution, args.paired_baseline)
    write_csv(os.path.join(args.out_dir, "SCREEN.csv"), screen,
              ["arm", "baseline", "subset", "seed", "budget", "comparison", "n_baseline",
               "sd_baseline", "threshold_kind", "threshold",
               "best", "ema_min", "tail10",
               "delta_best", "delta_ema", "delta_tail10", "consistent",
               "verdict", "decision", "decision_driver", "reason", "converged"])
    gates = build_gate_check(seed_rows)
    write_csv(os.path.join(args.out_dir, "GATE_CHECK.csv"), gates,
              ["gate_run", "gate_root", "module", "subset", "seed", "budget_epochs",
               "formal_run", "formal_root", "gate_md5", "formal_md5", "md5_match",
               "delta_value", "verdict"])
    budget_pairs = build_budget_pairs(per_engine)
    write_csv(os.path.join(args.out_dir, "BUDGET_VS.csv"), budget_pairs,
              ["module", "subset", "budget_lo", "budget_hi", "n_seeds", "seeds",
               "mean_diff", "sd_diff", "se", "t", "ci_lo", "ci_hi",
               "verdict", "significant"])
    ens_vs = build_ensemble_vs(seed_rows, args.cap, args.paired_baseline)
    write_csv(os.path.join(args.out_dir, "ENSEMBLE_VS_B0.csv"), ens_vs,
              ["module", "subset", "budget_epochs", "n_seeds_module", "n_seeds_b0",
               "ensemble_module_em", "ensemble_b0_em", "diff_em",
               "n_engines", "mean_diff", "sd_diff", "se", "t", "ci_lo", "ci_hi",
               "pct_vs_b0", "significant"])

    # ---- audit note ------------------------------------------------------- #
    scored = [r for r in seed_rows if r.get("status") in ("usable", "excluded", "unusable")]
    usable = [r for r in seed_rows if r.get("status") == "usable"]
    max_gap = max((abs(r["delta_vs_reported"]) for r in scored
                   if r.get("delta_vs_reported") is not None), default=float("nan"))
    counts: dict[str, int] = {}
    for r in seed_rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    lines = [
        "# Metric audit -- recomputed vs reported",
        "",
        "Generated by `aggregate_metrics.py`; do not hand-edit.",
        "",
        f"- runs scored from predictions: **{len(scored)}**",
        f"- runs usable in MODULE_RESULTS: **{len(usable)}**",
        f"- status counts: " + ", ".join(f"`{k}` {v}" for k, v in sorted(counts.items())),
        f"- max |recomputed - reported| engine-mean RMSE: **{max_gap:.3e}** "
        f"(tolerance {args.tol:g})",
        f"- frozen cap asserted: **{args.cap:g}**",
        f"- ledger: `{args.run_list or 'none (disk discovery only)'}`",
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
        "## Why some directories are not evidence",
        "",
        "| status | rule | enters MODULE_RESULTS |",
        "|---|---|---|",
        "| `complete` | `epochs_run >= budget.epochs` | yes |",
        "| `early_stopped` | patience fired: `epochs_run - best_ep >= patience` | yes |",
        "| `n/a_by_design` | zero trainable parameters (AGE) | yes |",
        "| `interrupted` | stopped early without patience firing | no |",
        "| `degenerate_curve` | `best_ep <= 2` and the run continued past 5 epochs -- never trained | no |",
        "| `smoke_override` | `budget_override` set (smoke run) | no |",
        "| `one_epoch_probe` | `epochs_run <= 1` and budget < 40, no override | no |",
        "| `non_canonical_id` | run id is not `HC-D-{arm}-{subset}-s{seed}-D1` | no |",
        "| `smoke_dir` | run id carries the `_smoke_` prefix (moved-aside smoke) | no |",
        "| `duplicate_ledger_row` | ledger lists one directory twice | no, counted once |",
        "| `duplicate_artifact` | identical `preds.csv` md5 (a copy, whatever the label) | no, counted once |",
        "| `cell_conflict` | same cell, different predictions (cannot tell which is right) | no, both dropped |",
        "| `unusable` | ledger marks `usable=no` | no |",
        "",
        "## Truncated runs (`truncated=yes`)",
        "",
        "A run with `epochs_run >= budget.epochs` never fired patience, so each of",
        "the last `patience` epochs still held an improvement. Its `best_ep` sits",
        "inside that final window and the reported score is a **lower bound** on the",
        "converged score, not an estimate of it. Such a run stays evidence and stays",
        "in `MODULE_RESULTS`, but it cannot be differenced against an",
        "`early_stopped` run as if both measured the same quantity: the truncated",
        "side is inflated, so the converged side looks better than it is.",
        "",
        "Test used: `epochs_run >= budget.epochs`. The intuitive test",
        "`best_ep == epochs_run` catches only the extreme case -- M3B-FD004-s101 is",
        "39/40 and would pass it. `early_stop_margin = epochs_run - best_ep` is",
        "exported alongside so the gap to the boundary is visible.",
        "",
        "### No direction is assigned, on purpose",
        "",
        "It is tempting to say a truncated score is an upper bound on the converged",
        "score, because `best` is `min(epoch 1..B)` and a larger B can only lower a",
        "minimum. That argument is wrong here: the schedule is",
        "`CosineAnnealingLR(T_max=epochs)`, so 40 and 80 epochs are two different",
        "curves, not one curve and its continuation. A budget-limited run is neither",
        "optimistic nor pessimistic -- it is a **different protocol**. The only",
        "correct statement is `result under the 40ep protocol, not extrapolated`.",
        "",
        "### The four independent axes",
        "",
        "`reproducible` / `verified` / `comparable` / `converged` are independent;",
        "a number can be any combination, and `n` is a fifth axis kept in its own",
        "column. Keeping them apart turns an argument into a table lookup.",
        "",
        "| axis | question | how it is decided |",
        "|---|---|---|",
        "| `reproducible` | can another person get this number from these files? |",
        "  recomputation from `preds.csv` matches `metrics.json` |",
        "| `verified` | was this cell shown current-HEAD equivalent? |",
        "  a `_gate` twin exists at the same (module, subset, seed, budget) and",
        "  **all four** of `preds_md5`, `val`, `epochs_run`, `best_ep` match |",
        "| `comparable` | can the two sides be differenced? | the parent has a cell",
        "  at the same subset and budget |",
        "| `converged` | did patience fire? | `epochs_run < budget` |",
        "",
        "`verified` is not a fingerprint comparison. The gates exist because a",
        "different fingerprint can be numerically identical -- G1 proved exactly",
        "that -- so judging by fingerprint would mark the one verified cell as",
        "unverified. All four items are required because early stopping has drifted",
        "independently of the values: matching bytes with a different `epochs_run`",
        "means the two runs are not the same experiment.",
        "",
        "### `budget_status`, `headroom` and `judgeable`",
        "",
        "`budget_status` describes how a run ended: `converged` (patience fired),",
        "`budget_limited` (reached the budget with patience still pending), `n/a`",
        "(zero-parameter arm), `unknown`. **It is a description, not a verdict.** A",
        "`budget_limited` run did not stop with a large amount of learning left: the",
        "schedule is `CosineAnnealingLR(T_max=epochs, eta_min=1e-5)`, so the LR has",
        "annealed to `eta_min` by the final epoch and the last few epochs move the",
        "weights very little.",
        "",
        "The gate is therefore measured, not asserted. `gain_lastk =",
        "best(1..B-k) - best(1..B)` is the improvement the final k epochs actually",
        "bought; k = 1, 3, 5, 6, 12 are exported. Across the 21 `budget_limited`",
        "cells in this sprint: `gain_last5` median 0.0000 / max 0.1552, `gain_last12`",
        "median 0.2621 / max 1.4696. `gain_last12` is the widest window that can",
        "matter, because patience=12 means any improvement a budget-limited run made",
        "must lie inside its final 12 epochs.",
        "",
        "Two disciplines apply to these columns:",
        "",
        "1. **They are statements, not bounds.** An observed recent gain does not",
        "   bound the remaining headroom. Write `the final 5 epochs gained at most",
        "   0.1552`, never `the residual is at most 0.1552`.",
        "2. **`gain_last12` is exactly 0.0000 for all 43 converged cells, by",
        "   construction**: a run that stopped at `best_ep + patience` has no",
        "   improvement inside its final patience window. The column carries",
        "   information only for budget-limited cells; on converged cells it is a",
        "   consistency check on the early-stop machinery, nothing more.",
        "",
        "`PAIRED_VS_B0.csv` carries `module_budget_status`,",
        "`baseline_budget_status`, `module_gain_last5`, `baseline_gain_last5`,",
        "`headroom` and `judgeable`. Only `judgeable=yes` rows belong in a judgment",
        "table:",
        "",
        "| pair_status | meaning | usable |",
        "|---|---|---|",
        "| `same_schedule` | same budget, and neither side gained more than",
        "  `HEADROOM_TOL=0.2` in its final five epochs | yes |",
        "| `headroom_high` | one side was still gaining > 0.2 at its boundary | no |",
        "| `unknown` | a status or a history curve is missing | no |",
        "",
        "Pairs are matched on `(subset, budget, seed)` by construction, so every pair",
        "here is a same-schedule pair; the real gate is the measured headroom.",
        "",
        "## Run identity comes from the path, not from the ledger label",
        "",
        "The ledger's `root` column is not trusted.  A row can say `root=d80` while",
        "its `run_dir` points into `d/`; believing the label turns one directory into",
        "two cells and inflates the seed count.  `root` is therefore derived from",
        "`run_dir` (`d`, `d80`, `d/probe80`), rows that resolve to the same real path",
        "are collapsed as `duplicate_ledger_row`, and any label/path disagreement is",
        "listed here.",
        "",
        "| run_id | ledger says | path says | path |",
        "|---|---|---|---|",
    ] + [
        f"| {m['run_id']} | `{m['ledger_root']}` | `{m['path_root']}` | `{m['rel']}` |"
        for m in root_mismatch
    ] + [
        "",
        "## Paired vs " + args.paired_baseline + " (same seed, same engines)",
        "",
        "A negative `mean_diff` means the module beats the baseline.  **Two variance",
        "sources are reported separately and must not be confused:**",
        "",
        "* `level=seed` -- CI over **engines** within one seed.  It answers \"is there a",
        "  systematic engine-level offset for this seed?\".  It does **not** cover seed",
        "  noise, so it cannot establish that two arms are equivalent.",
        "* `level=module` -- CI over **seeds** (df = n_seeds - 1) of the seed-level mean",
        "  differences.  This is the interval a paper claim needs; with 3 seeds df=2 and",
        "  t=4.303, so it is wide, and that width is the honest picture.",
        "",
        "### Per seed (engine-level CI)",
        "",
        "| module | subset | budget | seed | engines | mean diff | 95% CI | t | vs B0 | sig |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ] + [
        f"| {p['module']} | {p['subset']} | {p['budget_epochs']} | {p['seed']} | "
        f"{p['n_engines']} | {p['mean_diff']:+.4f} | "
        f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] | {p['t']:+.2f} | "
        f"{p['pct_vs_b0']:+.1f}% | {p['significant']} |"
        for p in paired if p["level"] == "seed"
    ] + [
        "",
        "### Across seeds (seed-level CI)",
        "",
        "| module | subset | budget | seeds | mean diff | 95% CI | t | df | vs B0 | sig |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ] + [
        f"| {p['module']} | {p['subset']} | {p['budget_epochs']} | {p['seeds']} | "
        f"{p['mean_diff']:+.4f} | "
        + (f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]" if p["n_seeds"] > 1
           else "n/a (1 seed)")
        + f" | {p['t']:+.2f} | {p['n_seeds'] - 1} | {p['pct_vs_b0']:+.1f}% | "
          f"{p['significant']} |"
        for p in paired if p["level"] == "module"
    ] + [
        "",
        "## Resolution (what difference can this cell resolve)",
        "",
        "Derived from the baseline's own seed spread, so it refreshes with the data.",
        "`screen_single` is the two-sigma width of a difference between two",
        "independent single-seed runs; `paired_3seed` is the width of an arm-vs-arm",
        "difference averaged over the same seeds.  Both scale with `sd`, which is",
        "per-cell: a module with a larger spread needs a wider threshold, so no",
        "single global constant applies to every module.",
        "",
        "| subset | budget | seeds | mean | sd | mean CI half-width | single-seed screen | paired 3-seed half-width |",
        "|---|---|---|---|---|---|---|---|",
    ] + [
        f"| {r['subset']} | {r['budget_epochs']} | {r['n_seeds']} | "
        + (f"{r['mean']:.4f} | {r['sd']:.4f} | ±{r['ci_mean']:.3f} | "
           f"±{r['screen_single']:.3f} | ±{r['paired_3seed']:.3f} |"
           if r["n_seeds"] > 1 else
           f"{r['mean']:.4f} | n/a (1 seed) | n/a | n/a | n/a |")
        for r in resolution
    ] + [
        "",
        "## Per-run check",
        "",
        "| root | run_id | n | eng | recomputed | reported | delta | NASA mean | MAE | bias | ep | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(scored, key=lambda x: (x["root"], x["run_id"])):
        lines.append(
            f"| {r['root']} | {r['run_id']} | {r.get('n_points','')} | "
            f"{r.get('n_engines','')} | {r.get('engine_mean_rmse', float('nan')):.6f} | "
            f"{r.get('reported_engine_mean_rmse', float('nan')):.6f} | "
            f"{r.get('delta_vs_reported', float('nan')):.2e} | "
            f"{r.get('nasa_mean', float('nan')):.4f} | "
            f"{r.get('mae', float('nan')):.4f} | {r.get('bias', 0.0):+.4f} | "
            f"{r.get('best_ep','')}/{r.get('epochs_run','')} | {r['status_detail']} |"
        )
    skipped = [r for r in seed_rows if r.get("status") not in ("usable", "excluded", "unusable")]
    if skipped:
        lines += ["", "## Runs that could not be scored", ""]
        for r in sorted(skipped, key=lambda x: (x["root"], x["run_id"])):
            lines.append(f"- `{r['root']}/{r['run_id']}` -- {r['status_detail']}")
    if problems:
        lines += ["", "## Problems", ""] + [f"- {p}" for p in problems]
    with open(os.path.join(args.out_dir, "METRIC_AUDIT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"scored {len(scored)} runs ({len(usable)} usable) -> {seed_path}")
    print(f"module rows: {len(module_rows)} -> {module_path}")
    print(f"per-engine rows: {len(per_engine)} -> {engine_path}")
    print(f"paired rows: {len(paired)} -> {paired_path}")
    print(f"status counts: {counts}")
    print(f"max |delta vs reported| = {max_gap:.3e}")
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
