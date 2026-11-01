"""Audit the test labels against the official RUL files before trusting any bias number.

A systematic +9 cycle over-prediction in one true-RUL band, identical across three
backbones, is exactly what a label artefact would look like. This checks the mundane
explanation first: that `y_true_raw` in the prediction CSVs equals the value NASA publishes,
that the unit ids are the ones the model actually scored, and that the sample was not
thinned in a way that correlates with RUL.

Also reports, per subset, which official units are missing and whether their RUL differs
from the retained ones -- if engines were dropped at random the band composition is
unaffected; if they were dropped because their trajectory was short, the band composition
is biased by construction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "CMAPSS"
PRED = ROOT / "results" / "dsh" / "e21" / "preds"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
ARMS = ["A", "BE", "PE"]
CAP = 125


def official(subset):
    vals = [float(x) for x in (DATA / f"RUL_{subset}.txt").read_text().split()]
    return {i + 1: v for i, v in enumerate(vals)}


def lens(subset, split):
    """Trajectory length per unit for split in {train, test}."""
    out = {}
    with open(DATA / f"{split}_{subset}.txt") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            out[u] = out.get(u, 0) + 1
    return out


def main():
    print(f"{'subset':7s} {'arm':4s} {'n_pred':>6s} {'n_off':>5s} {'max|dy|':>8s} "
          f"{'mismatch':>8s}")
    bad = []
    dropped = {}
    for s in SUBSETS:
        off = official(s)
        for arm in ARMS:
            rows = list(csv.DictReader(open(PRED / f"{s}_{arm}.csv")))
            n = len(rows)
            mx, mism = 0.0, 0
            seen = set()
            for r in rows:
                u = int(r["unit"])
                seen.add(u)
                raw = float(r["y_true_raw"])
                d = abs(raw - off.get(u, float("nan")))
                mx = max(mx, d)
                if d > 1e-6:
                    mism += 1
                    bad.append((s, arm, u, raw, off.get(u)))
            miss = sorted(set(off) - seen)
            dropped[(s, arm)] = miss
            print(f"{s:7s} {arm:4s} {n:6d} {len(off):5d} {mx:8.2f} {mism:8d}")

        # were the dropped units systematically different?
        ltest = lens(s, "test")
        allu = sorted(off)
        miss = dropped[(s, "A")]
        kept = [u for u in allu if u not in miss]
        if miss:
            rk = [off[u] for u in kept]
            rm = [off[u] for u in miss]
            print(f"  {s}: dropped {len(miss)}/{len(allu)} units "
                  f"{miss[:12]}{'...' if len(miss) > 12 else ''}")
            print(f"    RUL kept    mean={sum(rk)/len(rk):6.2f}  n={len(rk)}")
            print(f"    RUL dropped mean={sum(rm)/len(rm):6.2f}  n={len(rm)}")
            print(f"    obs len kept    mean={sum(ltest[u] for u in kept)/len(kept):6.1f}"
                  f"  min={min(ltest[u] for u in kept)}")
            print(f"    obs len dropped mean={sum(ltest[u] for u in miss)/len(miss):6.1f}"
                  f"  min={min(ltest[u] for u in miss)}")

    if bad:
        print("\nMISMATCHES (subset, arm, unit, csv_raw, official):")
        for b in bad[:20]:
            print("  ", b)
    else:
        print("\nall y_true_raw values match the official RUL files exactly")

    # does the cap actually bite where we think it does?
    print("\ncap structure per subset (official test labels):")
    for s in SUBSETS:
        v = list(official(s).values())
        print(f"  {s}: n={len(v)}  y>125: {sum(1 for x in v if x > 125)}  "
              f"y==125: {sum(1 for x in v if x == 125)}  max={max(v)}")

    out = ROOT / "experiments" / "cal" / "label_audit.json"
    out.write_text(json.dumps({"mismatches": bad,
                               "dropped": {f"{k[0]}_{k[1]}": v for k, v in dropped.items()}},
                              indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
