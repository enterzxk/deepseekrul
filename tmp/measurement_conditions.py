"""Measurement-condition evaluation on saved weights (evaluation-side corruption only).

Conditions
----------
clean  unchanged inputs
mc1    20% of sensor entries set missing (per window/cycle/sensor, fixed seed),
       forward-filled along the window, leading gaps back-filled
mc2    +1.0 training-fold standard deviation added to the three sensors whose
       early-life mean has the largest |Spearman| with total life in the training fold

Usage: python measurement_conditions.py SUBSET FOLD ARM SEED CONDITION [--window 50]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parent
sys.path[:0] = [str(PARENT), str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
import run_fold as base
import run_ida as ida
from scipy.stats import spearmanr

MISSING_RATE = 0.20
OFFSET_SIGMA = 1.0


def top_lifetime_sensors(df, train_units, n=3):
    """Sensors whose early-life mean is most associated with total life (training only)."""
    sub = df[df.unit.isin(train_units)]
    lives = sub.groupby("unit").cycle.max()
    early = sub[sub.cycle <= 15].groupby("unit")[list(ida.COLS)].mean()
    scores = {}
    for col in ida.COLS:
        v = early[col].reindex(lives.index).to_numpy(float)
        scores[col] = abs(spearmanr(v, lives.to_numpy(float)).statistic)
    return [c for c, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:n]]


def corrupt(x, condition, seed, sensors_idx=None):
    x = x.copy()
    if condition == "clean":
        return x
    rng = np.random.default_rng(seed)
    if condition == "mc1":
        mask = rng.random(x.shape) < MISSING_RATE
        for i in range(x.shape[0]):
            frame = pd.DataFrame(x[i])
            frame = frame.mask(mask[i])
            frame = frame.ffill().bfill().fillna(0.0)
            x[i] = frame.to_numpy(np.float32)
        return x
    if condition == "mc2":
        x[:, :, sensors_idx] += OFFSET_SIGMA
        return x
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset"); ap.add_argument("fold", type=int)
    ap.add_argument("arm"); ap.add_argument("seed", type=int); ap.add_argument("condition")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--rundir", default="runs3_w50")
    args = ap.parse_args()

    dest = HERE / "measurement" / args.subset / str(args.fold)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{args.arm}_{args.seed}_{args.condition}.csv"
    if out.exists():
        print("cached", out); return

    src = HERE / args.rundir / args.subset / str(args.fold)
    weights = src / f"{args.arm}_{args.seed}.pt"
    if not weights.is_file():
        raise SystemExit(f"missing weights {weights}")
    folds, df = ida.outer_folds(args.subset)
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    train, valid = folds[args.fold]
    tables = base.prior_tables(df, args.subset, train, valid, args.fold)
    tr, va = ida.prep_fold(df, args.subset, args.fold, train, valid, window=args.window)
    xva, yva, eva, mva = va[:4]
    sensors = top_lifetime_sensors(df, train) if args.condition == "mc2" else None
    idx = [list(ida.COLS).index(c) for c in sensors] if sensors else None
    xva = corrupt(xva, args.condition, ida.OUTER_SEED + args.fold, idx)

    from run_full_fold3 import build_net
    net = build_net(args.arm, args.window)
    state = torch.load(weights, map_location="cuda")
    net.load_state_dict(state)
    net = net.cuda().eval()
    vx = [base.device_array(a) for a in (xva, yva, eva, base.prior_values(mva, tables["FULL"]))]
    with torch.no_grad():
        raw = torch.cat([net(vx[0][i], None, vx[2][i], vx[3][i])["prediction"].reshape(-1)
                         for i in torch.arange(len(vx[0]), device="cuda").split(256)])
    raw = raw.cpu().numpy()
    mu = vx[3][:, 0].cpu().numpy().astype(np.float64) * 250.0
    pred = raw * mu if args.arm == "PHASE" else raw
    frame = pd.DataFrame(mva, columns=["unit", "end_cycle"]).assign(
        y_true=yva, y_pred=pred, raw_output=raw, mu_T=mu)
    frame.to_csv(out, index=False)
    (dest / f"{args.arm}_{args.seed}_{args.condition}.json").write_text(json.dumps(dict(
        subset=args.subset, fold=args.fold, arm=args.arm, seed=args.seed,
        condition=args.condition, sensors=sensors, missing_rate=MISSING_RATE,
        offset_sigma=OFFSET_SIGMA, n_windows=int(len(frame)))))
    print("done", out)


if __name__ == "__main__":
    main()
