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
]
RUN_RE = re.compile(r"^HC-D-(?P<arm>.+)-(?P<subset>FD00[1-4])-s(?P<seed>\d+)-D1$")

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
    row["status_detail"] = classify(row)
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
        return "smoke_override"
    if row.get("params_trainable") == 0:
        return "n/a_by_design"
    be, er = row.get("best_ep"), row.get("epochs_run")
    budget = row.get("budget_epochs")
    patience = row.get("budget_patience") or 0
    if be is None or er is None:
        return "unknown"
    if er <= 1 and (budget is None or budget < 40):
        return "one_epoch_probe"
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
        info.update(run_id=name, run_dir=path, root=label, in_ledger="no")
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
    "status", "status_detail", "in_ledger", "usable", "preds_md5",
    "n_points", "n_engines", "n_cuts",
    "engine_mean_rmse", "pooled_rmse", "nasa_sum", "nasa_mean", "mae", "bias",
    "late_frac", "severe_late", "max_late",
    "rmse_uncensored", "rmse_uncapped", "frac_capped",
    "boundary_hit", "exact_boundary",
    "seg_rmse_0", "seg_rmse_1", "seg_rmse_2", "seg_rmse_3", "seg_rmse_4",
    "seg_rul_mean_0", "seg_rul_mean_1", "seg_rul_mean_2", "seg_rul_mean_3", "seg_rul_mean_4",
    "reported_engine_mean_rmse", "reported_pooled_rmse", "delta_vs_reported",
    "best_ep", "epochs_run", "budget_epochs", "budget_patience", "budget_override",
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
    "nasa_mean_mean", "nasa_sum_mean", "mae_mean", "bias_mean",
    "rmse_uncensored_mean", "frac_capped_mean",
    "delta_vs_B0_abs", "delta_vs_B0_pct",
    "status_mix", "best_ep_max", "epochs_run_max",
    "params_trainable", "description",
]

USABLE = {"complete", "early_stopped", "n/a_by_design"}

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
        out.append(dict(
            level="seed",
            module=m["module"], subset=m["subset"], budget_epochs=m["budget"],
            seed=m["seed"], root=m["root"], n_seeds=1, seeds=str(m["seed"]),
            n_engines=st["n"], mean_diff=st["mean"], sd_diff=st["sd"],
            se=st["se"], t=st["t"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"],
            module_em=m["em"], b0_em=b0_em,
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
        if w["ledger_usable"] == "no":
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

    groups: dict[tuple[str, str, Any], list[dict[str, Any]]] = {}
    for row in seed_rows:
        if row.get("status") != "usable":
            continue
        groups.setdefault(
            (row["module"], row["subset"], row.get("budget_epochs")), []).append(row)

    baseline: dict[tuple[str, Any], float] = {}
    for (mod, subset, budget), rows in groups.items():
        if mod == "B0_direct_rul":
            v, _ = mean_sd([x["engine_mean_rmse"] for x in rows])
            baseline[(subset, budget)] = v

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
            nasa_mean_mean=nasa_mean, nasa_sum_mean=nasa_sum,
            mae_mean=mae_mean, bias_mean=bias_mean,
            rmse_uncensored_mean=unc_mean, frac_capped_mean=capped_mean,
            delta_vs_B0_abs=(mean - b0 if b0 is not None else float("nan")),
            delta_vs_B0_pct=(100.0 * (mean - b0) / b0
                             if b0 not in (None, 0) else float("nan")),
            status_mix=";".join(sorted({x["status_detail"] for x in rows})),
            best_ep_max=max((x["best_ep"] or 0) for x in rows),
            epochs_run_max=max((x["epochs_run"] or 0) for x in rows),
            params_trainable=rows[0]["params_trainable"],
            description=MODULE_DESC.get(mod, ""),
        ))

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
               "module_em", "b0_em", "pct_vs_b0", "verdict", "significant"])
    ensemble = build_ensemble(seed_rows, args.cap)
    write_csv(os.path.join(args.out_dir, "ENSEMBLE_RESULTS.csv"), ensemble,
              ["module", "subset", "budget_epochs", "n_seeds", "seeds",
               "seed_ems", "mean_seed_em", "sd_seed_em", "ensemble_em",
               "gain_abs", "gain_pct", "pair_mean_em", "pair_sd_em",
               "pair_min_em", "pair_max_em",
               "ensemble_nasa_mean", "ensemble_mae", "ensemble_bias",
               "n_engines", "n_cuts", "n_points", "status"])

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
        "| `smoke_override` | `budget_override` set (smoke run) | no |",
        "| `one_epoch_probe` | `epochs_run <= 1` and budget < 40, no override | no |",
        "| `non_canonical_id` | run id is not `HC-D-{arm}-{subset}-s{seed}-D1` | no |",
        "| `smoke_dir` | run id carries the `_smoke_` prefix (moved-aside smoke) | no |",
        "| `duplicate_ledger_row` | ledger lists one directory twice | no, counted once |",
        "| `duplicate_artifact` | identical `preds.csv` md5 (a copy, whatever the label) | no, counted once |",
        "| `cell_conflict` | same cell, different predictions (cannot tell which is right) | no, both dropped |",
        "| `unusable` | ledger marks `usable=no` | no |",
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
