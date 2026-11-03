"""Summarise PCDT runs: per-arm tables plus engine-cluster paired bootstrap.

Bootstrap unit = engine (windows of one engine move together), matching the
frozen protocol in paper/LSC_DESIGN.md section 4.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

import numpy as np


def variant(r):
    """Group key: arm + scale-head kind + detach flag (gru/plain-head runs
    recorded before these fields existed default to gru/no-detach)."""
    head = r.get("scale_head", "gru")
    det = "+det" if r.get("detach_lam") else ""
    return f"{r['arm']}{'' if head == 'gru' else ':' + head}{det}"


def load_runs(root="experiments/lsc/runs"):
    runs = []
    for f in sorted(glob.glob(f"{root}/*.json")):
        runs.append(json.load(open(f, encoding="utf-8")))
    return runs


def arm_table(runs):
    by = defaultdict(list)
    for r in runs:
        by[variant(r)].append(r)
    print(f"{'arm':8s} {'n':>3s} {'RMSE(all)':>10s} {'sd':>5s} {'lam_corr':>9s} "
          f"{'MAE@c100':>9s} {'bias@c100':>10s} params")
    for arm, rs in sorted(by.items()):
        rm = [r["rmse_all"] for r in rs]
        lc = [r["lam_corr_all"] for r in rs if r.get("lam_corr_all") is not None]
        m100 = [r["fixed_age"]["100"]["mae"] for r in rs if "100" in r.get("fixed_age", {})]
        b100 = [r["fixed_age"]["100"]["bias"] for r in rs if "100" in r.get("fixed_age", {})]
        print(f"{arm:8s} {len(rs):>3d} {np.mean(rm):>10.3f} {np.std(rm):>5.3f} "
              f"{np.mean(lc) if lc else float('nan'):>9.3f} "
              f"{np.mean(m100) if m100 else float('nan'):>9.3f} "
              f"{np.mean(b100) if b100 else float('nan'):>10.2f} {rs[0]['params']}")


def bucket_table(runs):
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for q, v in r.get("bias_buckets", {}).items():
            by[variant(r)][q].append(v)
    if not by:
        return
    print("\n=== lifetime-scale stratification (all windows, train-quartile buckets)")
    qs = sorted({q for rs in by.values() for q in rs})
    print(f"{'arm':8s} " + " ".join(f"{q:>14s}" for q in qs))
    for arm, qs_d in sorted(by.items()):
        cells = []
        for q in qs:
            v = qs_d[q]
            cells.append(f"{np.mean([x['rmse'] for x in v]):>6.2f}/{np.mean([x['bias'] for x in v]):>+6.2f}")
        print(f"{arm:8s} " + " ".join(f"{c:>14s}" for c in cells))
    print("  cell = RMSE/bias")


def paired_bootstrap(runs, a, b, reps=4000, seed=2026):
    """Positive gain = arm a worse than arm b (b better).  Engine clusters."""
    def eng_err(arm):
        d = defaultdict(list)
        for r in runs:
            if variant(r) != arm:
                continue
            for e in r["err_all"]:
                d[e["eng"]].append(e["err"] ** 2)
        return {k: float(np.mean(v)) for k, v in d.items()}
    ea, eb = eng_err(a), eng_err(b)
    units = sorted(set(ea) & set(eb))
    if len(units) < 5:
        return None
    v = np.array([ea[u] - eb[u] for u in units])       # a worse -> positive
    rmse_a = float(np.sqrt(np.mean([ea[u] for u in units])))
    rmse_b = float(np.sqrt(np.mean([eb[u] for u in units])))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(reps, len(v)))
    boots = np.sqrt(v[idx].mean(axis=1).clip(0) + 0)   # not exact; use mean diff instead
    # paired RMSE-difference bootstrap on MSE units (variance-stable)
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return dict(n_engines=len(units), rmse_a=rmse_a, rmse_b=rmse_b,
                dmse_mean=float(v.mean()), dmse_ci=[float(lo), float(hi)])


def pair_table(runs, focal="full"):
    arms = sorted({variant(r) for r in runs})
    print(f"\n=== paired engine-cluster bootstrap vs {focal} (MSE units, 95% CI)")
    print("  positive CI = focal worse than the comparator")
    for arm in arms:
        if arm == focal:
            continue
        r = paired_bootstrap(runs, focal, arm)
        if r:
            print(f"  {focal:8s} vs {arm:8s}  RMSE {r['rmse_a']:.3f} vs {r['rmse_b']:.3f}"
                  f"  dMSE {r['dmse_mean']:+.2f} CI [{r['dmse_ci'][0]:+.2f}, {r['dmse_ci'][1]:+.2f}]"
                  f"  n={r['n_engines']}")


def main():
    runs = load_runs(sys.argv[1] if len(sys.argv) > 1 else "experiments/lsc/runs")
    if not runs:
        print("no runs found")
        return
    arm_table(runs)
    bucket_table(runs)
    focal = sys.argv[2] if len(sys.argv) > 2 else "full"
    pair_table(runs, focal)


if __name__ == "__main__":
    main()
