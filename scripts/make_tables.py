"""Generate the paper tables from the saved result JSONs.

Usage:
    python scripts/make_tables.py --root experiments/punet_e2e

Outputs Markdown tables to stdout:
    Table 1  main results per subset (mean over folds, with std)
    Table 2  paired comparison CQR vs constant-width baseline
    Table 3  ablation A4 (sigma permutation), paired
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def load(root: Path, subset: str, fold: int, name: str):
    p = root / subset / f"f{fold}_s42" / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/punet_e2e")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    root = Path(args.root)

    print("## Table 1 — main results (mean over folds)\n")
    print("| Subset | PICP | PINAW | Winkler | engine-min coverage |")
    print("|---|---|---|---|---|")
    for s in SUBSETS:
        rows = [load(root, s, f, "e2e_cqr_none") for f in range(args.folds)]
        rows = [r for r in rows if r]
        if not rows:
            continue
        f = lambda k: st.mean(r[k] for r in rows)
        sd = lambda k: (st.stdev([r[k] for r in rows]) if len(rows) > 1 else 0.0)
        print(f"| {s} (n={len(rows)}) | {f('picp'):.3f} ± {sd('picp'):.3f} | "
              f"{f('pinaw'):.3f} | {f('winkler'):.1f} ± {sd('winkler'):.1f} | {f('engine_min'):.3f} |")

    print("\n## Table 2 — paired comparison vs constant-width baseline\n")
    print("| Subset | Fold | CQR | Baseline | Delta | CQR PICP | Base PICP |")
    print("|---|---|---|---|---|---|---|")
    deltas = []
    for s in SUBSETS:
        for f in range(args.folds):
            a = load(root, s, f, "e2e_cqr_none")
            b = load(root, s, f, "e2e_point_none")
            if not (a and b):
                continue
            d = (a["winkler"] - b["winkler"]) / b["winkler"] * 100
            deltas.append(d)
            print(f"| {s} | f{f} | {a['winkler']:.1f} | {b['winkler']:.1f} | "
                  f"{d:+.1f}% | {a['picp']:.3f} | {b['picp']:.3f} |")
    if deltas:
        wins = sum(1 for d in deltas if d < 0)
        print(f"\nPaired folds: {len(deltas)} | CQR better: {wins} "
              f"({100*wins/len(deltas):.0f}%) | mean {st.mean(deltas):+.1f}% | "
              f"median {st.median(deltas):+.1f}%")

    print("\n## Table 3 — ablation A4 (sigma permutation), paired\n")
    print("| Subset | Fold | CQR | A4 | Delta |")
    print("|---|---|---|---|---|")
    ad = []
    for s in SUBSETS:
        for f in range(args.folds):
            a = load(root, s, f, "e2e_cqr_none")
            b = load(root, s, f, "e2e_cqr_A4")
            if not (a and b):
                continue
            d = (b["winkler"] - a["winkler"]) / a["winkler"] * 100
            ad.append(d)
            print(f"| {s} | f{f} | {a['winkler']:.1f} | {b['winkler']:.1f} | {d:+.1f}% |")
    if ad:
        print(f"\nPaired folds: {len(ad)} | A4 worse: {sum(1 for d in ad if d > 0)} "
              f"| mean {st.mean(ad):+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
