"""Aggregate the B-line run files into tables.

Reports, per (subset, variant, hold): mean RMSE over folds and seeds, its spread,
and the paired comparison against ``erm`` on the *same* test engines, with the
engine as the independent unit (a paired bootstrap over engines, not over runs:
the 259 test engines are the same everywhere, so resampling them is the only
honest way to put a standard error on a difference of RMSEs).
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "bline" / "runs"


def load() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(RUNS / "*.json"))):
        try:
            out.append(json.load(open(f)))
        except json.JSONDecodeError:
            pass
    return out


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def paired_boot(pred_a, pred_b, truth, n=2000, seed=0):
    """Bootstrap the difference of RMSEs over engines."""
    rng = np.random.default_rng(seed)
    ea, eb = pred_a - truth, pred_b - truth
    k = len(truth)
    idx = rng.integers(0, k, size=(n, k))
    d = np.sqrt((ea[idx] ** 2).mean(1)) - np.sqrt((eb[idx] ** 2).mean(1))
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def schedule_table(runs):
    """In-distribution runs carry an extra evaluation under replayed schedules."""
    rows = [r for r in runs if r.get("sched")]
    if not rows:
        return
    variants = sorted({r["variant"] for r in rows})
    print("\n=== schedule shift (all setpoints seen; test replayed under one schedule)")
    print("  RMSE with every cycle at one setpoint, and with the setpoint alternating")
    print(f"  {'variant':10s} {'n':>3s} {'original':>9s} " +
          " ".join(f"{'c'+str(c):>7s}" for c in range(6)) + f" {'alt':>7s} {'drift':>7s}")
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        orig = np.mean([r["rmse"] for r in rs])
        per = {c: np.mean([r["sched"][str(c)]["rmse"] for r in rs]) for c in range(6)}
        alt = np.mean([r["sched"]["alt"]["rmse"] for r in rs])
        drift = max(abs(per[c] - orig) for c in range(6))
        print(f"  {v:10s} {len(rs):>3d} {orig:>9.2f} " +
              " ".join(f"{per[c]:>7.2f}" for c in range(6)) +
              f" {alt:>7.2f} {drift:>7.2f}")


def across_holds(runs):
    """Mean over the six leave-one-setpoint-out cells: the headline number."""
    rows = [r for r in runs if r["hold"] >= 0]
    if not rows:
        return
    variants = sorted({r["variant"] for r in rows})
    holds = sorted({r["hold"] for r in rows})
    print("\n=== leave-one-setpoint-out, mean over the six held-out setpoints")
    print(f"  {'variant':10s} " + " ".join(f"{'h'+str(h):>7s}" for h in holds) +
          f" {'mean':>7s} {'sd':>6s}")
    for v in variants:
        per = []
        for h in holds:
            e = [r["rmse"] for r in rows if r["variant"] == v and r["hold"] == h]
            per.append(np.mean(e) if e else np.nan)
        per = np.array(per)
        print(f"  {v:10s} " + " ".join(f"{x:>7.2f}" for x in per) +
              f" {np.nanmean(per):>7.2f} {np.nanstd(per):>6.2f}")


def main():
    runs = load()
    if not runs:
        print("no runs")
        return
    groups = defaultdict(list)
    for r in runs:
        groups[(r["subset"], r["variant"], r["hold"])].append(r)

    subsets = sorted({k[0] for k in groups})
    for sub in subsets:
        holds = sorted({k[2] for k in groups if k[0] == sub})
        variants = sorted({k[1] for k in groups if k[0] == sub})
        for hold in holds:
            name = "in-distribution" if hold < 0 else f"hold out setpoint {hold}"
            print(f"\n=== {sub}  {name}")
            print(f"  {'variant':10s} {'n':>3s} {'RMSE':>7s} {'sd':>6s} {'MAE':>7s} "
                  f"{'NASA':>8s} {'bias':>7s} {'vs erm':>16s}")
            base = None
            for v in variants:
                rs = groups.get((sub, v, hold), [])
                if not rs:
                    continue
                e = np.array([r["rmse"] for r in rs])
                mae = np.mean([r["mae"] for r in rs])
                nas = np.mean([r["nasa"] for r in rs])
                bias = np.mean([r["bias"] for r in rs])
                line = (f"  {v:10s} {len(rs):>3d} {e.mean():>7.2f} {e.std():>6.2f} "
                        f"{mae:>7.2f} {nas:>8.0f} {bias:>+7.2f}")
                if v == "erm":
                    base = rs
                    line += f" {'—':>16s}"
                elif base is not None and len(base) == len(rs):
                    # pair the runs that share seed, then bootstrap over engines
                    ds = []
                    for rb, rv in zip(sorted(base, key=lambda x: x["seed"]),
                                      sorted(rs, key=lambda x: x["seed"])):
                        pa, pb = np.array(rb["pred"]), np.array(rv["pred"])
                        t = np.array(rb["truth"])
                        if len(pa) != len(pb):
                            continue
                        ds.append(paired_boot(pb, pa, t))
                    if ds:
                        m = float(np.mean([d[0] for d in ds]))
                        lo = float(np.mean([d[1] for d in ds]))
                        hi = float(np.mean([d[2] for d in ds]))
                        line += f" {m:>+7.2f} [{lo:+.1f},{hi:+.1f}]"
                print(line)
    across_holds(runs)
    schedule_table(runs)
    print("\nnegative 'vs erm' = lower RMSE than erm; bracket is a 95% engine bootstrap")
    print("'drift' is the largest change in RMSE caused only by replaying the schedule")


if __name__ == "__main__":
    main()
