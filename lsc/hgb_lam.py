"""Out-of-fold HGB lifetime estimates -> per-window Lambda-hat (LSC 'hgb' arm).

The stage-1 gate's HistGradientBoosting probe predicted TOTAL lifetime from
prefix statistics at cutoff ages c in {50..200} under 5-fold OOF: every unit
appears in exactly one fold's test split, so its predictions never come from
a model trained on that unit.  That makes the estimates leak-free regardless
of which fold split the PCDT trainer uses.

This module turns the per-unit (c -> prediction) curves into a per-window
Lambda-hat by piecewise-linear interpolation on the window's age, with
constant extension outside the observed c range (early windows have no
prefix information anyway).
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

_CACHE: dict = {}


def load_unit_curves(pred_file: str, subset: str = "FD002", arm: str = "sensor"):
    """unit -> (cs_sorted, preds_sorted) from the OOF gate predictions."""
    key = (pred_file, subset, arm)
    if key in _CACHE:
        return _CACHE[key]
    with open(pred_file, encoding="utf-8") as f:
        d = json.load(f)[subset]
    by: dict = defaultdict(lambda: ([], []))
    for r in d:
        by[r["unit"]][0].append(float(r["c"]))
        by[r["unit"]][1].append(float(r[arm]))
    curves = {}
    for u, (cs, ps) in by.items():
        order = np.argsort(np.asarray(cs))
        curves[u] = (np.asarray(cs, float)[order], np.asarray(ps, float)[order])
    _CACHE[key] = curves
    return curves


def lam_windows(pred_file: str, subset: str, eng: np.ndarray, cyc: np.ndarray,
                arm: str = "sensor") -> np.ndarray:
    """Per-window Lambda-hat, one float per row of (eng, cyc)."""
    curves = load_unit_curves(pred_file, subset, arm)
    eng = np.asarray(eng)
    cyc = np.asarray(cyc, float)
    out = np.empty(len(eng), np.float32)
    missing = []
    for u in np.unique(eng):
        m = eng == u
        cur = curves.get(int(u))
        if cur is None:
            missing.append(int(u))
            continue
        cs, ps = cur
        out[m] = np.interp(cyc[m], cs, ps)   # constant extension at both ends
    if missing:
        raise KeyError(f"no OOF HGB predictions for units {missing[:8]} "
                       f"(subset={subset}, file={pred_file})")
    return out
