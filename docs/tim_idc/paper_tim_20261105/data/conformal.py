"""Lifetime-conditioned Mondrian conformal layer for RUL intervals.

Calibrates |y - yhat| quantiles per predicted-RUL bin on the outer-validation
windows, evaluates conditional coverage by cross-conformal across the five
outer folds, and freezes the full-calibration quantiles for the official test.
Never reads RUL_FD00X.txt.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evidence import FLEETS  # noqa: E402

SEEDS = [101, 202, 303]
ARMS = ["M0", "OPS", "FULL", "CET"]
ALPHA = 0.10
BINS = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0]
BIN_LABELS = ["[0,25)", "[25,50)", "[50,75)", "[75,100)", "[100,125]"]
MIN_BIN = 30
RUL_MAX = 125.0


def arm_path(fleet: str, seed: int, fold: int, arm: str) -> Path:
    if seed == 101:
        return HERE / "runs" / fleet / str(fold) / f"{arm}.csv"
    return HERE / "replicas" / str(seed) / fleet / str(fold) / f"{arm}.csv"


def load_arm(fleet: str, seed: int, arm: str):
    frames = []
    for fold in range(5):
        path = arm_path(fleet, seed, fold, arm)
        if not path.is_file():
            return None
        frame = pd.read_csv(path)
        frame["fold"] = fold
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["seed"] = seed
    data["arm"] = arm
    return data


def load_ensemble(fleet: str, arm: str):
    parts = []
    for seed in SEEDS:
        data = load_arm(fleet, seed, arm)
        if data is None:
            return None
        parts.append(data.set_index(["fold", "unit", "end_cycle"])[["y_true", "y_pred"]])
    index = parts[0].index
    for part in parts[1:]:
        index = index.intersection(part.index)
    truths = [part.loc[index, "y_true"].to_numpy() for part in parts]
    assert all(np.array_equal(truths[0], truth) for truth in truths[1:])
    preds = np.mean([part.loc[index, "y_pred"].to_numpy() for part in parts], axis=0)
    out = pd.DataFrame({"y_true": truths[0], "y_pred": preds}, index=index).reset_index()
    out["seed"] = 0
    out["arm"] = arm + "_ens"
    return out


def bin_id(pred):
    bins = np.digitize(np.asarray(pred, dtype=float), BINS[1:-1], right=False)
    return np.clip(bins, 0, len(BIN_LABELS) - 1)


def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        return None
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(scores[k - 1])


def fit_quantiles(data, alpha):
    scores = np.abs(data["y_true"].to_numpy() - data["y_pred"].to_numpy())
    bins = bin_id(data["y_pred"].to_numpy())
    global_q = conformal_quantile(scores, alpha)
    out = {}
    for b, label in enumerate(BIN_LABELS):
        s = scores[bins == b]
        if len(s) >= MIN_BIN:
            out[label] = {"n": int(len(s)), "q": conformal_quantile(s, alpha), "fallback": False}
        else:
            out[label] = {"n": int(len(s)), "q": global_q, "fallback": True}
    out["_global"] = {"n": int(len(scores)), "q": global_q, "fallback": False}
    return out


def cross_conformal(data, alpha):
    rows = []
    for fold in range(5):
        cal = data[data["fold"] != fold]
        ev = data[data["fold"] == fold]
        quantiles = fit_quantiles(cal, alpha)
        bins = bin_id(ev["y_pred"].to_numpy())
        q = np.array([quantiles[BIN_LABELS[b]]["q"] for b in bins])
        lo = np.clip(ev["y_pred"].to_numpy() - q, 0.0, RUL_MAX)
        hi = np.clip(ev["y_pred"].to_numpy() + q, 0.0, RUL_MAX)
        cover = (ev["y_true"].to_numpy() >= lo) & (ev["y_true"].to_numpy() <= hi)
        width = hi - lo
        for b, label in enumerate(BIN_LABELS):
            mask = bins == b
            if not mask.any():
                continue
            rows.append({"fold": fold, "bin": label, "n": int(mask.sum()),
                         "coverage": float(cover[mask].mean()),
                         "mean_width": float(width[mask].mean())})
        rows.append({"fold": fold, "bin": "all", "n": int(len(ev)),
                     "coverage": float(cover.mean()),
                     "mean_width": float(width.mean())})
    frame = pd.DataFrame(rows)
    summary = frame.groupby("bin", sort=False).agg(
        n=("n", "sum"), coverage=("coverage", "mean"),
        mean_width=("mean_width", "mean")).reset_index()
    return {"per_fold": frame.to_dict(orient="records"),
            "summary": summary.to_dict(orient="records")}


def main():
    result = {
        "alpha": ALPHA,
        "bins": BIN_LABELS,
        "seeds": SEEDS,
        "arms": ARMS,
        "method": ("Mondrian split conformal on |y-yhat| with bins by predicted RUL; "
                   "finite-sample quantile ceil((n+1)(1-alpha))/n; global fallback n<30; "
                   "intervals clipped to [0,125]"),
        "official_labels_read": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subsets": {},
    }
    for fleet in FLEETS:
        result["subsets"][fleet] = {}
        for arm in ARMS:
            for seed in SEEDS:
                data = load_arm(fleet, seed, arm)
                if data is None:
                    continue
                result["subsets"][fleet][f"{arm}_s{seed}"] = {
                    "n_windows": int(len(data)),
                    "cross_conformal": cross_conformal(data, ALPHA),
                    "quantiles": fit_quantiles(data, ALPHA),
                }
            data = load_ensemble(fleet, arm)
            if data is not None:
                result["subsets"][fleet][f"{arm}_ens"] = {
                    "n_windows": int(len(data)),
                    "cross_conformal": cross_conformal(data, ALPHA),
                    "quantiles": fit_quantiles(data, ALPHA),
                }
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE / "conformal_results.json").write_text(json.dumps(result, indent=2))
    protocol = {
        "method": result["method"],
        "alpha": ALPHA,
        "bins": BIN_LABELS,
        "seeds": SEEDS,
        "arms": ARMS,
        "code_sha256": result["code_sha256"],
        "frozen_at_utc": result["frozen_at_utc"],
        "official_labels_read": False,
        "quantiles": {
            fleet: {key: value["quantiles"] for key, value in arms.items()}
            for fleet, arms in result["subsets"].items()
        },
    }
    (HERE / "conformal_protocol.json").write_text(json.dumps(protocol, indent=2))
    print("wrote conformal_results.json / conformal_protocol.json")
    for fleet in FLEETS:
        for key in ("FULL_s101", "FULL_ens"):
            if key in result["subsets"][fleet]:
                summary = result["subsets"][fleet][key]["cross_conformal"]["summary"]
                overall = [row for row in summary if row["bin"] == "all"][0]
                print(f"{fleet} {key}: coverage={overall['coverage']:.3f} "
                      f"width={overall['mean_width']:.2f} n={overall['n']}")


if __name__ == "__main__":
    main()
