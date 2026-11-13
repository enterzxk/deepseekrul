#!/usr/bin/env python
"""Aggregation caliber of the 20 per-engine test columns.

For every (test row, column) pair, decide whether the per-unit equal-weight
value and the pooled value are *algebraically* identical on that row, by
counting how many flights each engine contributes to the aggregation set.

Why counting and not comparing values: if every engine contributes k flights,

    pooled = (sum_e S_e) / (E*k) = (1/E) * sum_e (S_e/k) = per-unit equal weight

so the two agree by construction. A value comparison cannot tell that apart
from a coincidental equality, and they have different meanings: the first
means the column carries no information about the aggregation on that row,
the second is a genuine pass that merely happens to match.

States:
    degenerate        every engine contributes the same number of flights
                      (including the single-engine case) -> not assertable
    pass              counts differ and the values differ
    pass_coincident   counts differ but the values happen to agree -> pass,
                      flagged so nobody later reads "equal" as "no difference"

Usage: python scripts/nc_agg_caliber.py [--csv out.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import os

SEG_REL = (1 / 3, 2 / 3)
SEG_ABS = (75.0, 25.0)
SUBS = ["DS02", "DS05", "DS06", "DS07"]
BASE_COLS = ["rmse", "mae", "bias", "nasa"]
SEG_COLS = [(n, t, c) for t in ("relative", "absolute")
            for n in ("early", "mid", "late") for c in ("capped", "raw")]


def rmse(v):
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")


def nasa(d):
    return math.exp(-d / 13.0) - 1.0 if d < 0 else math.exp(d / 10.0) - 1.0


def agg(kind, e, ec):
    if kind == "rmse":
        return rmse(e), rmse(ec)
    if kind == "mae":
        return sum(abs(x) for x in e) / len(e), sum(abs(x) for x in ec) / len(ec)
    if kind == "bias":
        return sum(e) / len(e), sum(ec) / len(ec)
    return sum(nasa(x) for x in e), sum(nasa(x) for x in ec)


def state_of(counts, per, pooled):
    if all(c == counts[0] for c in counts):
        return "degenerate"
    return "pass_coincident" if abs(per - pooled) < 1e-12 else "pass"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)

    cells = []
    for s in SUBS:
        rows = list(csv.DictReader(io.open("experiments/ncmapss/%s/runs_test.csv" % s, encoding="utf-8")))
        for r in rows:
            pr = list(csv.DictReader(io.open(r["preds_path"], encoding="utf-8")))
            life = json.load(io.open(os.path.join(os.path.dirname(r["preds_path"]),
                                                  "unit_lifetime.json"), encoding="utf-8"))
            by = {}
            for x in pr:
                by.setdefault(int(x["unit"]), []).append(
                    (float(x["y_true_raw"]), float(x["y_true_cap125"]), float(x["y_pred"])))
            for kind in BASE_COLS:
                per_r, per_c, counts, pe, pc = [], [], [], [], []
                for u, v in sorted(by.items()):
                    e = [p - t for t, c, p in v]
                    ec = [p - c for t, c, p in v]
                    a_r, a_c = agg(kind, e, ec)
                    per_r.append(a_r)
                    per_c.append(a_c)
                    counts.append(len(v))
                    pe += e
                    pc += ec
                pooled_r, pooled_c = agg(kind, pe, pc)
                for suffix, per, pooled in (("raw", per_r, pooled_r), ("cap125", per_c, pooled_c)):
                    mean = sum(per) / len(per)
                    cells.append(dict(subset=s, run_id=r["run_id"],
                                      column="%s_%s_unit" % (kind, suffix),
                                      n=len(counts), counts=counts,
                                      per_unit=mean, pooled=pooled,
                                      state=state_of(counts, mean, pooled)))
            for name, tag, crit in SEG_COLS:
                per, counts, pool = [], [], []
                for u, v in sorted(by.items()):
                    L = float(life[str(u)])
                    y = [t for t, c, p in v]
                    yc = [c for t, c, p in v]
                    e = [p - t for t, c, p in v]
                    ec = [p - c for t, c, p in v]
                    frac = [t / max(L, 1e-9) for t in y]
                    if tag == "relative":
                        mask = {"early": [f > SEG_REL[1] for f in frac],
                                "mid": [SEG_REL[0] < f <= SEG_REL[1] for f in frac],
                                "late": [f <= SEG_REL[0] for f in frac]}[name]
                    else:
                        mask = {"early": [t >= SEG_ABS[0] for t in y],
                                "mid": [SEG_ABS[1] <= t < SEG_ABS[0] for t in y],
                                "late": [t < SEG_ABS[1] for t in y]}[name]
                    idx = [i for i, m in enumerate(mask) if m]
                    if not idx:
                        continue
                    err = [ec[i] if crit == "capped" else e[i] for i in idx]
                    per.append(rmse(err))
                    counts.append(len(idx))
                    pool += err
                if not per:
                    continue
                pooled = rmse(pool)
                cells.append(dict(subset=s, run_id=r["run_id"],
                                  column="seg_%s_%s_%s" % (name, tag, crit),
                                  n=len(counts), counts=counts,
                                  per_unit=sum(per) / len(per), pooled=pooled,
                                  state=state_of(counts, sum(per) / len(per), pooled)))

    st = collections.Counter(c["state"] for c in cells)
    print("cells %d  pass %d  degenerate %d  pass_coincident %d"
          % (len(cells), st["pass"], st["degenerate"], st["pass_coincident"]))
    by = collections.Counter((c["column"], c["subset"], c["state"]) for c in cells
                             if c["state"] != "pass")
    for (col, sub, state), n in sorted(by.items()):
        print("  %-28s %-5s %-15s %2d rows" % (col, sub, state, n))
    if a.csv:
        with io.open(a.csv, "w", encoding="utf-8", newline="\n") as fh:
            w = csv.DictWriter(fh, fieldnames=["subset", "run_id", "column", "n", "counts",
                                               "per_unit", "pooled", "state"],
                               lineterminator="\n")
            w.writeheader()
            for c in cells:
                row = dict(c)
                row["counts"] = ";".join(str(x) for x in c["counts"])
                w.writerow(row)
        print("wrote %s" % a.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
