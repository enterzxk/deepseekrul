"""Paired engine-level bootstrap for sprint arms (corrected).

Fix after the independent review: the same physical engines are evaluated under
every seed, so a bootstrap replicate must draw ONE set of engine indices and
apply it to all seeds and both arms. Drawing independently per seed would break
the cross-seed correlation and misstate the interval.

Statistic: relative change in engine-macro RMSE, pooled over seeds
    rel = sqrt(mean_{seed,engine} MSE_cand) / sqrt(mean_{seed,engine} MSE_base) - 1
The fraction of replicates with rel < 0 is reported as "P(rel<0)"; it is NOT a
p-value and NOT a posterior probability of the method being correct.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_mse(subset: str, seed: int, arm: str, fold: int = 0) -> pd.Series | None:
    f = Path(f"experiments/tim_sprint/{subset}/f{fold}_s{seed}_{arm}/preds.csv")
    if not f.exists():
        return None
    d = pd.read_csv(f)
    e2 = (d["y_hat"] - d["y_true"]) ** 2
    return pd.DataFrame({"u": d["unit"], "e2": e2}).groupby("u")["e2"].mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--base", default="R")
    ap.add_argument("--cand", default="A2")
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--reps", type=int, default=5000)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = {}
    for sd in seeds:
        b, c = load_mse(args.subset, sd, args.base), load_mse(args.subset, sd, args.cand)
        if b is None or c is None:
            print(f"seed {sd}: missing preds ({args.base} or {args.cand})")
            continue
        idx = b.index.intersection(c.index)
        per_seed[sd] = (b.loc[idx].to_numpy(), c.loc[idx].to_numpy())
        print(f"seed {sd}: {args.base}={np.sqrt(b.loc[idx].mean()):.3f} "
              f"{args.cand}={np.sqrt(c.loc[idx].mean()):.3f} "
              f"rel={(np.sqrt(c.loc[idx].mean())/np.sqrt(b.loc[idx].mean())-1)*100:+.2f}% "
              f"n_engines={len(idx)}")

    if not per_seed:
        return 1

    rng = np.random.default_rng(0)
    n = len(next(iter(per_seed.values()))[0])
    rels = []
    for _ in range(args.reps):
        pick = rng.integers(0, n, n)          # ONE draw, reused for every seed and arm
        mb = np.mean([mb_[pick].mean() for mb_, _ in per_seed.values()])
        mc = np.mean([mc_[pick].mean() for _, mc_ in per_seed.values()])
        rels.append((np.sqrt(mc) / np.sqrt(mb) - 1) * 100)
    rels = np.array(rels)
    print(f"\npaired engine bootstrap ({args.reps} reps, one engine draw shared across seeds):")
    print(f"  mean={rels.mean():+.2f}%  95%CI=[{np.percentile(rels,2.5):+.2f}%, {np.percentile(rels,97.5):+.2f}%]")
    print(f"  P(rel<0)={float((rels < 0).mean()):.3f}   (fraction of replicates, not a p-value)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
