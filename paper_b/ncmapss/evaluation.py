"""Flight-level evaluation for N-CMAPSS DS02 (Paper B cross-dataset validation).

Owner: 燕子 (evaluation protocol). This module only scores predictions; it never
trains, never reads raw DS02 arrays, and never opens the official test engines
unless a freeze manifest authorises it.

Rules fixed by the 2026-11-12 brief
----------------------------------
R1  Dev engines are evaluated by leave-one-engine-out over [2, 5, 10, 16, 18, 20]:
    each dev engine is the validation fold once, the other five are training.
    Every fitted statistic (scaler, operating-condition encoder, RUL normaliser)
    must be fitted on the current training fold only. ``assert_fold_discipline``
    makes that checkable instead of a promise.
R2  The evaluation unit is the **flight**, i.e. one row per (unit, cycle) at the
    end of that flight. A 1 Hz sample-level score is rejected by construction
    because it would weight long flights more than short ones.
R3  ``capped`` (label = min(RUL, 125)) and ``raw`` (label = RUL) are two
    criteria. Every result carries ``criterion`` and ``rul_cap``, and
    ``assert_single_criterion`` refuses a ranking that mixes them.
R4  Official test engines [11, 14, 15] stay closed until ``assert_test_allowed``
    finds them in a frozen manifest.

Typical use
-----------
>>> folds = loeo_folds()
>>> for fold in folds:
...     assert_fold_discipline(fold, fit_units=fold.train_units)
...     pred = my_model_fit_predict(fold.train_units, fold.val_units)
...     res = evaluate(pred, criterion="capped")
>>> summary = aggregate_seeds([r1, r2, r3])

Run ``python -m paper_b.ncmapss.evaluation --self-test`` to check the module
without any DS02 data on disk.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Protocol constants (do not change without a new freeze manifest)
# --------------------------------------------------------------------------
DEV_UNITS: tuple[int, ...] = (2, 5, 10, 16, 18, 20)
TEST_UNITS: tuple[int, ...] = (11, 14, 15)
RUL_CAP: float = 125.0

#: Segment rules. Two are always reported, never one alone:
#:   relative  (PRIMARY)  thirds of each unit's own remaining-life fraction, so the
#:                        split stays scale-free across units with different lifetimes.
#:   absolute  (SUPPORT)  fixed RUL cut points in cycles, the rule the C-MAPSS tables
#:                        already use. Needed to align early/mid/late *shapes* across
#:                        datasets; it is not the primary rule.
SEGMENT_EDGES_REL: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
SEGMENT_EDGES_ABS: tuple[float, float] = (75.0, 25.0)  # cycles: early >= 75, mid 25..75, late < 25
SEGMENT_NAMES: tuple[str, str, str] = ("late", "mid", "early")
SEGMENT_RULES: dict[str, str] = {
    "relative": "relative_thirds_primary",
    "absolute": "absolute_rul_supplementary",
}
DEFAULT_SEGMENT_RULES: tuple[str, ...] = ("relative", "absolute")

COL_UNIT = "unit"
COL_CYCLE = "cycle"
COL_TRUE = "y_true"
COL_PRED = "y_pred"

#: A prediction later than the truth by more than this many cycles counts as a
#: "severe late" error. It is a reporting convention, not a protocol threshold:
#: it exists so the number has a name and a home instead of being a literal
#: buried in an expression.
SEVERE_LATE_CYCLES: float = 10.0

#: D043 fixes the primary metric as per-engine equal weight; the flight-pooled
#: RMSE is reported next to it as a supplement. Every table must say which is
#: which, because the two differ by Jensen's inequality and are not
#: interchangeable.
PRIMARY_METRIC: str = "engine_mean_rmse"
SUPPLEMENTARY_METRIC: str = "rmse_pooled"


# --------------------------------------------------------------------------
# Fold construction and discipline checks
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    """One leave-one-engine-out fold over the dev engines."""

    index: int
    val_unit: int
    train_units: tuple[int, ...]

    @property
    def fit_units(self) -> tuple[int, ...]:
        """Units the preprocessing is allowed to see (training fold only)."""
        return self.train_units


def loeo_folds(dev_units: Sequence[int] = DEV_UNITS) -> list[Fold]:
    """Leave-one-engine-out folds over the dev engines, in sorted order."""
    units = tuple(sorted(int(u) for u in dev_units))
    if len(units) < 2:
        raise ValueError("LOEO needs at least two dev engines")
    if set(units) & set(TEST_UNITS):
        raise ValueError(f"dev units overlap the official test set: {sorted(set(units) & set(TEST_UNITS))}")
    return [
        Fold(index=i, val_unit=u, train_units=tuple(x for x in units if x != u))
        for i, u in enumerate(units)
    ]


def assert_fold_discipline(fold: Fold, fit_units: Iterable[int]) -> None:
    """Raise unless the fitted units are exactly the fold's training units.

    Passing anything else (all engines, or the validation engine itself) is the
    classic silent leak: the scaler then knows the validation range.
    """
    fitted = tuple(sorted(int(u) for u in fit_units))
    if fitted != fold.train_units:
        leaked = sorted(set(fitted) & {fold.val_unit})
        raise ValueError(
            f"fold {fold.index}: fitted units {fitted} != train units {fold.train_units}"
            + (f" (validation engine {leaked} leaked into fitting)" if leaked else "")
        )


def assert_no_test_units(units: Iterable[int], context: str = "") -> None:
    """Dev-side code must never touch the official test engines."""
    bad = sorted(set(int(u) for u in units) & set(TEST_UNITS))
    if bad:
        raise ValueError(f"official test engines {bad} used in {context or 'a dev-side call'}")


# --------------------------------------------------------------------------
# Flight-level frame
# --------------------------------------------------------------------------
def flight_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a flight-level frame.

    Three things are enforced, each aimed at one specific way of getting the
    unit of analysis wrong:

    1. Exactly one prediction per (unit, cycle). Duplicates mean somebody handed
       us 1 Hz samples, which would silently reweight the score by flight length.
    2. ``cycle`` is an integer. A fractional cycle is a sample timestamp.
    3. ``cycle`` strictly increases within each unit. A 1 Hz sample index passed
       off as a cycle number fails here as soon as the flights are collapsed,
       because the surviving values are spaced by the flight length, not by one.
    """
    need = {COL_UNIT, COL_CYCLE, COL_TRUE, COL_PRED}
    missing = need - set(rows.columns)
    if missing:
        raise ValueError(f"prediction frame is missing columns: {sorted(missing)}")
    df = rows.loc[:, [COL_UNIT, COL_CYCLE, COL_TRUE, COL_PRED]].copy()
    for c in (COL_UNIT, COL_CYCLE):
        raw = df[c].to_numpy()
        if not np.issubdtype(df[c].dtype, np.integer):
            if not np.all(np.isfinite(raw.astype(float))) or not np.all(raw.astype(float) == np.floor(raw.astype(float))):
                raise ValueError(
                    f"{c} must be integer-valued: got non-integer values, which means a "
                    "sample index or timestamp is being passed as a flight cycle"
                )
        df[c] = df[c].astype(int)
    for c in (COL_TRUE, COL_PRED):
        df[c] = df[c].astype(float)
    if df.empty:
        raise ValueError("prediction frame is empty")
    if not np.isfinite(df[[COL_TRUE, COL_PRED]].to_numpy()).all():
        raise ValueError("non-finite values in y_true/y_pred")
    if (df[COL_CYCLE] < 0).any():
        raise ValueError("negative cycle values")
    dup = df.duplicated([COL_UNIT, COL_CYCLE]).sum()
    if dup:
        raise ValueError(
            f"{dup} duplicated (unit, cycle) rows: evaluation must be flight-level, "
            "not 1 Hz sample-level"
        )
    for unit, block in df.groupby(COL_UNIT):
        cyc = np.sort(block[COL_CYCLE].to_numpy())
        if np.any(np.diff(cyc) <= 0):
            raise ValueError(
                f"unit {unit}: cycle is not strictly increasing; a non-monotonic cycle "
                "usually means a 1 Hz sample index is being used as the cycle number"
            )
    return df.sort_values([COL_UNIT, COL_CYCLE]).reset_index(drop=True)


def cycle_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Per-unit cycle bookkeeping, to prove the column really is a flight counter.

    ``gaps`` counts missing cycle numbers inside a unit's range. A contiguous
    1..N range is what a flight counter looks like; large gaps mean a sample
    index survived the collapse.
    """
    recs = []
    for unit, block in df.groupby(COL_UNIT):
        cyc = np.sort(block[COL_CYCLE].to_numpy())
        span = int(cyc[-1] - cyc[0] + 1)
        recs.append(dict(unit=int(unit), n_flights=int(len(cyc)), first_cycle=int(cyc[0]),
                         last_cycle=int(cyc[-1]), span=span, gaps=int(span - len(cyc))))
    return pd.DataFrame(recs).sort_values("unit").reset_index(drop=True)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.asarray(pred, float) - np.asarray(true, float)


def rmse(pred, true) -> float:
    e = _err(pred, true)
    return float(np.sqrt(np.mean(e**2)))


def mae(pred, true) -> float:
    return float(np.mean(np.abs(_err(pred, true))))


def nasa_score(pred, true) -> tuple[float, float]:
    """Asymmetric score: exp(-d/13)-1 early, exp(d/10)-1 late, d = pred - true."""
    d = _err(pred, true)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s)), float(np.mean(s))


def score(pred, true) -> dict[str, float]:
    """Flight-level metrics for one prediction set."""
    e = _err(pred, true)
    nasa_sum, nasa_mean = nasa_score(pred, true)
    return dict(
        n_flights=int(len(e)),
        rmse=rmse(pred, true),
        mae=mae(pred, true),
        nasa_sum=nasa_sum,
        nasa_mean=nasa_mean,
        bias=float(np.mean(e)),
        late_frac=float(np.mean(e > 0)),
        severe_late=float(np.mean(e > SEVERE_LATE_CYCLES)),
        max_late=float(np.max(e)) if e.size else float("nan"),
    )


def apply_criterion(df: pd.DataFrame, criterion: str, rul_cap: float = RUL_CAP) -> pd.DataFrame:
    """Return a copy whose ``y_true`` follows the requested criterion.

    ``capped`` -> min(RUL, rul_cap); ``raw`` -> RUL unchanged. Predictions are
    never clipped, matching the C-MAPSS convention so the two datasets stay
    comparable in shape.
    """
    if criterion not in ("capped", "raw"):
        raise ValueError(f"criterion must be 'capped' or 'raw', got {criterion!r}")
    out = df.copy()
    if criterion == "capped":
        out[COL_TRUE] = np.minimum(out[COL_TRUE], float(rul_cap))
    return out


def segment_labels(
    df: pd.DataFrame,
    rule: str = "relative",
    edges_rel: Sequence[float] = SEGMENT_EDGES_REL,
    edges_abs: Sequence[float] = SEGMENT_EDGES_ABS,
    unit_lifetime: dict[int, float] | None = None,
) -> pd.Series:
    """Label each flight early/mid/late under one segment rule.

    ``relative`` (primary) divides by the unit's **full-lifetime** maximum RUL,
    which must be supplied by the data layer. Deriving the denominator from the
    frame being scored is wrong whenever that frame does not span the whole
    lifetime: on the official test set each unit contributes a single flight,
    so ``frame.max()`` equals that flight's own RUL, every fraction becomes 1.0
    and the whole test set silently collapses into ``early``.

    ``absolute`` (supplementary) uses fixed RUL cut points in cycles.
    """
    if rule not in SEGMENT_RULES:
        raise ValueError(f"segment rule must be one of {sorted(SEGMENT_RULES)}, got {rule!r}")

    if rule == "relative":
        if unit_lifetime is None:
            raise ValueError(
                "relative segment rule needs unit_lifetime (per-unit max RUL from the full "
                "label trajectory); without it a single-flight-per-unit frame degenerates "
                "to frac=1.0 for every row"
            )
        lo, hi = float(edges_rel[0]), float(edges_rel[1])
        frac = np.empty(len(df), dtype=float)
        for unit, idx in df.groupby(COL_UNIT).groups.items():
            unit = int(unit)
            if unit not in unit_lifetime:
                raise ValueError(f"unit_lifetime is missing unit {unit}")
            denom = float(unit_lifetime[unit])
            if not np.isfinite(denom) or denom <= 0:
                raise ValueError(f"unit {unit}: lifetime must be a positive number, got {denom}")
            block = df.loc[idx, COL_TRUE].to_numpy()
            if block.size and block.max() > denom + 1e-9:
                raise ValueError(
                    f"unit {unit}: RUL {block.max():.3f} in the evaluated frame exceeds the "
                    f"declared lifetime {denom:.3f}; the lifetime is not from this unit"
                )
            frac[df.index.get_indexer(idx)] = block / denom
        labels = np.where(frac < lo, "late", np.where(frac < hi, "mid", "early"))
    else:
        hi, lo = float(edges_abs[0]), float(edges_abs[1])  # (75, 25)
        y = df[COL_TRUE].to_numpy()
        labels = np.where(y < lo, "late", np.where(y < hi, "mid", "early"))

    return pd.Series(labels, index=df.index, name="segment")


def evaluate(
    rows: pd.DataFrame,
    criterion: str,
    rul_cap: float = RUL_CAP,
    edges_rel: Sequence[float] = SEGMENT_EDGES_REL,
    edges_abs: Sequence[float] = SEGMENT_EDGES_ABS,
    segment_rules: Sequence[str] = DEFAULT_SEGMENT_RULES,
    unit_lifetime: dict[int, float] | None = None,
) -> dict:
    """Full flight-level evaluation under one criterion.

    ``criterion`` is positional and has no default on purpose: a forgotten
    argument must fail loudly rather than silently score the wrong label
    convention. D043 makes ``raw`` the primary criterion and ``capped`` the
    fixed supplement, so neither can be an implicit default.

    ``unit_lifetime`` maps unit -> max RUL over that unit's full label
    trajectory. It is required whenever the relative segment rule is used,
    because a frame holding one flight per unit cannot supply the scale itself.

    Returns the overall metrics, the per-engine table, and the early/mid/late
    table under **both** segment rules, every block stamped with the criterion
    it used and each segment row stamped with the rule that produced it.
    """
    df = apply_criterion(flight_frame(rows), criterion, rul_cap)

    overall = score(df[COL_PRED], df[COL_TRUE])
    overall.update(criterion=criterion, rul_cap_used=(criterion == "capped"), rul_cap=float(rul_cap))

    per_engine = []
    for unit, block in df.groupby(COL_UNIT):
        m = score(block[COL_PRED], block[COL_TRUE])
        m.update(unit=int(unit), criterion=criterion, rul_cap_used=(criterion == "capped"))
        per_engine.append(m)
    per_engine = pd.DataFrame(per_engine).sort_values("unit").reset_index(drop=True)

    seg_tables = []
    for rule in segment_rules:
        labels = segment_labels(df, rule, edges_rel, edges_abs, unit_lifetime)
        block_df = df.assign(segment=labels)
        for name in SEGMENT_NAMES:
            block = block_df[block_df["segment"] == name]
            if block.empty:
                continue
            m = score(block[COL_PRED], block[COL_TRUE])
            m.update(segment=name, segment_rule=SEGMENT_RULES[rule], criterion=criterion,
                     rul_cap_used=(criterion == "capped"))
            seg_tables.append(m)
    segments = pd.DataFrame(seg_tables)

    engine_mean_rmse = float(per_engine["rmse"].mean()) if len(per_engine) else float("nan")
    return dict(
        overall=overall,
        per_engine=per_engine,
        segments=segments,
        # D043: per-engine equal weight is the primary number, the flight-pooled
        # RMSE is the supplement. Both are returned, clearly labelled, because
        # they differ (Jensen) and must not be swapped.
        engine_mean_rmse=engine_mean_rmse,
        rmse_pooled=overall["rmse"],
        primary_metric=PRIMARY_METRIC,
        supplementary_metric=SUPPLEMENTARY_METRIC,
        criterion=criterion,
        rul_cap_used=(criterion == "capped"),
        n_engines=int(df[COL_UNIT].nunique()),
        n_flights=int(len(df)),
        segment_edges_rel=list(edges_rel),
        segment_edges_abs=list(edges_abs),
    )


def evaluate_both(
    rows: pd.DataFrame,
    rul_cap: float = RUL_CAP,
    edges_rel: Sequence[float] = SEGMENT_EDGES_REL,
    edges_abs: Sequence[float] = SEGMENT_EDGES_ABS,
    unit_lifetime: dict[int, float] | None = None,
) -> dict:
    """Both criteria side by side, never merged into one score."""
    return {
        "capped": evaluate(rows, "capped", rul_cap, edges_rel, edges_abs, unit_lifetime=unit_lifetime),
        "raw": evaluate(rows, "raw", rul_cap, edges_rel, edges_abs, unit_lifetime=unit_lifetime),
    }


# --------------------------------------------------------------------------
# Multi-seed aggregation and criterion hygiene
# --------------------------------------------------------------------------
def aggregate_seeds(results: Sequence[dict], keys: Sequence[str] = ("rmse", "mae", "nasa_sum", "bias")) -> dict:
    """Mean and sample sd over >= 3 seeds for each requested metric."""
    if len(results) < 3:
        raise ValueError(f"need >= 3 seeds for a mean+-sd, got {len(results)}")
    crit = {r["criterion"] for r in results}
    if len(crit) != 1:
        raise ValueError(f"aggregating mixed criteria {sorted(crit)}; run each criterion separately")
    out: dict[str, object] = {"n_seeds": len(results), "criterion": crit.pop(),
                              "rul_cap_used": bool(results[0]["rul_cap_used"])}
    for k in keys:
        vals = np.array([r["overall"][k] for r in results], dtype=float)
        out[k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1))}
    return out


def assert_single_criterion(table: pd.DataFrame, column: str = "criterion") -> None:
    """Refuse a table/ranking that mixes capped and raw rows."""
    if column not in table.columns:
        raise ValueError(f"table has no {column!r} column: a ranking must declare its criterion")
    seen = sorted(set(table[column].astype(str)))
    if len(seen) > 1:
        raise ValueError(
            f"table mixes criteria {seen}; capped and raw (and C-MAPSS vs N-CMAPSS) "
            "must never share one ranking"
        )


# --------------------------------------------------------------------------
# Test-set gate (mirrors paper_b/evaluation/evaluate_test.py)
# --------------------------------------------------------------------------
def assert_test_allowed(freeze_manifest: str | Path | None, tag: str, seed: int) -> None:
    """Block official test evaluation unless the run is in a frozen manifest."""
    if freeze_manifest is None:
        raise SystemExit("no freeze manifest: official test evaluation is blocked")
    path = Path(freeze_manifest)
    if not path.exists():
        raise SystemExit(f"freeze manifest not found: {path}")
    man = json.loads(path.read_text())
    if not man.get("frozen"):
        raise SystemExit(f"manifest {path} is not frozen")
    allowed = {(str(r["tag"]), int(r["seed"])) for r in man.get("runs", [])}
    if (str(tag), int(seed)) not in allowed:
        raise SystemExit(f"run ({tag}, seed={seed}) is not in the freeze manifest")


# --------------------------------------------------------------------------
# Self-test (no DS02 data needed)
# --------------------------------------------------------------------------
def _self_test() -> int:
    folds = loeo_folds()
    assert len(folds) == 6
    assert [f.val_unit for f in folds] == list(DEV_UNITS)
    assert all(f.val_unit not in f.train_units for f in folds)
    assert all(len(f.train_units) == 5 for f in folds)

    # fold discipline: exact train units pass, anything else fails
    assert_fold_discipline(folds[0], folds[0].train_units)
    for bad in (DEV_UNITS, (*folds[0].train_units, folds[0].val_unit)):
        try:
            assert_fold_discipline(folds[0], bad)
        except ValueError:
            pass
        else:
            raise AssertionError("fold discipline check failed to reject a leaky fit set")

    try:
        assert_no_test_units([2, 11])
    except ValueError:
        pass
    else:
        raise AssertionError("test-engine guard did not fire")

    # synthetic flights: 6 units, 12 flights each, RUL 120 -> 10
    rng = np.random.default_rng(0)
    recs = []
    for u in DEV_UNITS:
        n = 12
        for c in range(n):
            true = float((n - c) * 10)
            recs.append(dict(unit=u, cycle=c + 1, y_true=true, y_pred=true + rng.normal(0, 3)))
    rows = pd.DataFrame(recs)
    lifetime = {int(u): 120.0 for u in DEV_UNITS}

    frame = flight_frame(rows)
    assert len(frame) == len(rows)

    # 1 Hz rows must be rejected
    dup = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    try:
        flight_frame(dup)
    except ValueError as e:
        assert "flight-level" in str(e)
    else:
        raise AssertionError("sample-level input was not rejected")

    # criterion has no default: forgetting it must fail loudly
    try:
        evaluate(frame)  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("evaluate() accepted a missing criterion")

    # relative segments need an explicit lifetime
    try:
        evaluate(frame, "capped")
    except ValueError as e:
        assert "unit_lifetime" in str(e)
    else:
        raise AssertionError("relative segments ran without unit_lifetime")

    capped = evaluate(frame, "capped", unit_lifetime=lifetime)
    raw = evaluate(frame, "raw", unit_lifetime=lifetime)
    assert capped["overall"]["criterion"] == "capped" and capped["overall"]["rul_cap_used"] is True
    assert raw["overall"]["rul_cap_used"] is False
    assert capped["overall"]["n_flights"] == len(rows)
    assert capped["n_engines"] == 6
    # max true RUL is 120 < cap, so capped and raw must agree exactly here
    assert math.isclose(capped["overall"]["rmse"], raw["overall"]["rmse"], rel_tol=1e-12)

    # primary vs supplementary: labelled, and pooled >= per-engine mean (Jensen)
    assert capped["primary_metric"] == "engine_mean_rmse"
    assert capped["rmse_pooled"] == capped["overall"]["rmse"]
    assert capped["engine_mean_rmse"] <= capped["rmse_pooled"] + 1e-12

    # REGRESSION (the official-test shape): one flight per unit. The relative
    # rule must keep its scale from the lifetime, not from the single row.
    last = frame.groupby(COL_UNIT).tail(1).copy()
    one = evaluate(last, "raw", unit_lifetime=lifetime)
    rel = one["segments"][one["segments"]["segment_rule"] == SEGMENT_RULES["relative"]]
    assert len(rel) == 1, rel
    assert set(rel["segment"]) == {"late"}, f"single-flight frame degenerated to {set(rel['segment'])}"
    try:
        evaluate(last, "raw")
    except ValueError:
        pass
    else:
        raise AssertionError("single-flight frame was scored without a lifetime")

    # a lifetime smaller than an observed RUL belongs to another unit
    bad_lifetime = dict(lifetime)
    bad_lifetime[int(DEV_UNITS[0])] = 10.0
    try:
        evaluate(frame, "raw", unit_lifetime=bad_lifetime)
    except ValueError as e:
        assert "exceeds the declared lifetime" in str(e)
    else:
        raise AssertionError("impossible lifetime was accepted")

    # a unit with RUL above the cap must differ between criteria
    long_rows = frame.copy()
    long_rows.loc[long_rows[COL_UNIT] == 2, COL_TRUE] += 40.0
    lifetime_long = dict(lifetime)
    lifetime_long[2] = 160.0
    c2 = evaluate(long_rows, "capped", unit_lifetime=lifetime_long)
    r2 = evaluate(long_rows, "raw", unit_lifetime=lifetime_long)
    assert r2["overall"]["rmse"] > c2["overall"]["rmse"]

    # segments: both rules present, each covering every flight exactly once
    seg = capped["segments"]
    assert set(seg["segment"]) <= set(SEGMENT_NAMES)
    assert set(seg["segment_rule"]) == set(SEGMENT_RULES.values())
    for rule, block in seg.groupby("segment_rule"):
        assert block["n_flights"].sum() == len(rows), f"{rule} segments do not cover all flights"

    prov = cycle_provenance(frame)
    assert len(prov) == 6 and int(prov["gaps"].sum()) == 0

    # the two segment rules must genuinely differ, otherwise the supplementary
    # table would be a copy of the primary one
    rel_lab = segment_labels(frame, "relative", unit_lifetime=lifetime)
    abs_lab = segment_labels(frame, "absolute")
    assert (rel_lab != abs_lab).any(), "relative and absolute segment rules agree everywhere"

    # cycle column must be an integer flight counter, not a sample index
    bad = frame.copy()
    bad[COL_CYCLE] = bad[COL_CYCLE].astype(float) + 0.5
    try:
        flight_frame(bad)
    except ValueError as e:
        assert "integer-valued" in str(e)
    else:
        raise AssertionError("fractional cycle was accepted")

    # a collapsed 1 Hz index leaves gaps; non-monotonic input must be rejected
    gap = frame.copy()
    gap.loc[gap[COL_UNIT] == 2, COL_CYCLE] = gap.loc[gap[COL_UNIT] == 2, COL_CYCLE] * 100
    prov_gap = cycle_provenance(flight_frame(gap))
    assert int(prov_gap.loc[prov_gap["unit"] == 2, "gaps"].iloc[0]) > 0

    nonmono = frame.copy()
    idx = nonmono.index[nonmono[COL_UNIT] == 5]
    nonmono.loc[idx[1], COL_CYCLE] = nonmono.loc[idx[0], COL_CYCLE]  # duplicate cycle
    try:
        flight_frame(nonmono)
    except ValueError:
        pass
    else:
        raise AssertionError("non-monotonic cycle was accepted")

    # multi-seed aggregation
    seeds = [evaluate(frame, "capped", unit_lifetime=lifetime) for _ in range(3)]
    agg = aggregate_seeds(seeds)
    assert agg["n_seeds"] == 3 and agg["criterion"] == "capped"
    assert agg["rmse"]["sd"] == 0.0
    try:
        aggregate_seeds(seeds[:2])
    except ValueError:
        pass
    else:
        raise AssertionError("2 seeds should not be aggregated into mean+-sd")

    # criterion hygiene
    mixed = pd.DataFrame([{"criterion": "capped"}, {"criterion": "raw"}])
    try:
        assert_single_criterion(mixed)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed-criterion table was accepted")

    print("SELF-TEST OK")
    print(f"  folds: {[f.val_unit for f in folds]}")
    print(f"  flights scored: {capped['overall']['n_flights']} over {capped['n_engines']} engines")
    print(f"  primary {capped['primary_metric']}={capped['engine_mean_rmse']:.3f} | "
          f"supplementary {capped['supplementary_metric']}={capped['rmse_pooled']:.3f}")
    print(f"  capped rmse {capped['overall']['rmse']:.3f} | raw rmse {raw['overall']['rmse']:.3f}")
    for rule, block in seg.groupby("segment_rule"):
        print(f"  {rule}: {dict(zip(block['segment'], block['n_flights']))}")
    print(f"  single-flight (test shape) relative segment: {sorted(set(rel['segment']))}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", help="check the module without DS02 data")
    ap.add_argument("--predictions", help="CSV with unit,cycle,y_true,y_pred (flight-level)")
    ap.add_argument("--criterion", choices=("capped", "raw", "both"), default="both")
    ap.add_argument("--rul-cap", type=float, default=RUL_CAP)
    ap.add_argument("--lifetime-json",
                    help='JSON {"unit": max_rul_over_full_trajectory}. Required for the '
                         'relative segment rule; the scored frame cannot supply it.')
    ap.add_argument("--absolute-only", action="store_true",
                    help="skip the relative segment rule (absolute cut points only)")
    ap.add_argument("--out-dir", default=None, help="write summary/per-engine/segment CSVs here")
    a = ap.parse_args(argv)

    if a.self_test:
        return _self_test()
    if not a.predictions:
        ap.error("--predictions is required unless --self-test is given")

    lifetime = None
    if a.lifetime_json:
        raw = Path(a.lifetime_json).read_text(encoding="utf-8-sig")
        lifetime = {int(k): float(v) for k, v in json.loads(raw).items()}
    if not a.absolute_only and lifetime is None:
        ap.error("--lifetime-json is required for the relative segment rule "
                 "(or pass --absolute-only)")

    rows = pd.read_csv(a.predictions)
    rules = ("absolute",) if a.absolute_only else DEFAULT_SEGMENT_RULES
    kw = dict(rul_cap=a.rul_cap, unit_lifetime=lifetime)
    if a.criterion == "both":
        results = {
            "capped": evaluate(rows, "capped", segment_rules=rules, **kw),
            "raw": evaluate(rows, "raw", segment_rules=rules, **kw),
        }
    else:
        results = {a.criterion: evaluate(rows, a.criterion, segment_rules=rules, **kw)}

    for name, res in results.items():
        print(f"[{name}] flights={res['n_flights']} engines={res['n_engines']} "
              f"{res['primary_metric']}={res['engine_mean_rmse']:.3f} "
              f"(primary) {res['supplementary_metric']}={res['rmse_pooled']:.3f} (supplementary) "
              f"MAE={res['overall']['mae']:.3f} NASA={res['overall']['nasa_sum']:.1f} "
              f"bias={res['overall']['bias']:+.3f}")
    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, res in results.items():
            res["per_engine"].to_csv(out / f"per_engine_{name}.csv", index=False)
            res["segments"].to_csv(out / f"segments_{name}.csv", index=False)
        (out / "summary.json").write_text(json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk not in ("per_engine", "segments")}
             for k, v in results.items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
