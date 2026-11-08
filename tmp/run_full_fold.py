"""Full-budget outer-fold runner: 80 epochs, ALL training windows, fixed last epoch.

Arms
----
M0        plain PE backbone (no extra lifetime input)               -- reference candidate
OPS       PE + age/condition lifetime prior injected (zero-init)    -- reference candidate
FULL      PE + sensor-history lifetime prior injected (zero-init)   -- reference candidate
NORM      PE with the cycle coordinate replaced by c / mu_T          -- new mechanism
NORM_SHUF same, mu_T permuted across engines (control: wrong identity)
NORM_CONST same, mu_T replaced by the training median (control: constant scale)
NORM_OPS  same as NORM but mu_T from the age/condition prior (control: sensor vs age)

Nothing is selected on these data: one invocation owns one outer fold and writes
per-window predictions.  Aggregation is a separate read-only script.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent                      # tim_20261105
ROOT = PARENT.parent                      # /home/user/rulb
sys.path[:0] = [str(PARENT), str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
import run_fold as base                   # noqa: E402  (prior_tables, prior_values, ida)
import run_ida as ida                     # noqa: E402

EPOCHS = 80
BATCH = 256


def shuffled_table(table, seed):
    units = sorted({u for u, _ in table})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(units))
    mapping = {u: units[perm[i]] for i, u in enumerate(units)}
    return {(mapping[u], c): v for (u, c), v in table.items()}


def constant_table(table):
    value = float(np.median(list(table.values())))
    return {key: value for key in table}


def build_arm(arm, tables, median_life):
    """Return (network, prior-table, uses-norm) for an arm name."""
    if arm in ("M0", "OPS", "FULL"):
        body = ida.CandidateModel(num_sensors=14, window_size=50, d_model=64, num_layers=1,
                                  dropout=0.15, use_baseline=True, use_ema_features=True,
                                  use_cycle=True, use_pe=True, use_graph=False,
                                  use_dual_head=False)
        return ida.PriorWrapper(body, "M0" if arm == "M0" else "M1"), tables[arm], False
    if arm in ("NORM", "NORM_SHUF", "NORM_CONST", "NORM_OPS"):
        body = ida.CandidateModel(num_sensors=14, window_size=50, d_model=64, num_layers=1,
                                  dropout=0.15, use_baseline=True, use_ema_features=True,
                                  use_cycle=True, use_pe=True, use_graph=False,
                                  use_dual_head=False)
        return ida.PriorWrapper(body, "M0", norm=True), tables[arm], True
    raise ValueError(arm)


def train_arm(arm, net, train_data, valid_data, target, metadata, seed):
    if target.exists():
        return False
    ida.set_seed(seed)
    g = torch.Generator(device="cuda").manual_seed(seed)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    tr = [base.device_array(a) for a in train_data]

    def forward(data, idx):
        x, y, extras, prior = (v[idx] for v in data)
        return net(x, None, extras, prior)["prediction"].reshape(-1)

    for _ in range(EPOCHS):
        net.train()
        for idx in torch.randperm(len(tr[0]), generator=g, device="cuda").split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            error = forward(tr, idx) - tr[1][idx]
            loss = error.square().mean() + .5 * torch.where(error > 0, 1.8, 1.).mul(error.square()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
        schedule.step()
    va = [base.device_array(a) for a in valid_data]
    net.eval()
    with torch.no_grad():
        pred = torch.cat([forward(va, idx) for idx in torch.arange(len(va[0]), device="cuda").split(BATCH)])
    pred = pred.cpu().numpy()
    assert np.isfinite(pred).all()
    frame = metadata.assign(y_true=valid_data[1], y_pred=pred)
    tmp = target.with_suffix(".partial")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, target)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset")
    ap.add_argument("fold", type=int)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--arms", default="M0,OPS,FULL,NORM,NORM_SHUF,NORM_CONST,NORM_OPS")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "0"))

    dest = HERE / "runs" / args.subset / str(args.fold)
    dest.mkdir(parents=True, exist_ok=True)
    folds, df = ida.outer_folds(args.subset)
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    train, valid = folds[args.fold]
    tables = base.prior_tables(df, args.subset, train, valid, args.fold)
    median_life = float(df.groupby("unit").cycle.max().loc[sorted(train)].median())
    arms = [a for a in args.arms.split(",") if a]
    if "NORM_SHUF" in arms:
        tables["NORM_SHUF"] = shuffled_table(tables["FULL"], ida.OUTER_SEED + 991 * (args.fold + 1) + args.seed)
    if "NORM_CONST" in arms:
        tables["NORM_CONST"] = constant_table(tables["FULL"])
    if "NORM_OPS" in arms:
        tables["NORM_OPS"] = tables["OPS"]
    tables["M0"] = tables["FULL"]          # only used for shape; M0 ignores priors

    tr, va = ida.prep_fold(df, args.subset, args.fold, train, valid)
    xtr, ytr, etr, mtr = tr[:4]
    xva, yva, eva, mva = va[:4]
    meta = pd.DataFrame(mva, columns=["unit", "end_cycle"])
    lives = df.groupby("unit").cycle.max()
    meta["short"] = meta.unit.map(lives) <= np.quantile(lives.loc[sorted(train)], .25)

    done = []
    for arm in arms:
        net, table, _ = build_arm(arm, tables, median_life)
        ida.set_seed(args.seed)
        net = net.cuda()
        tx = (xtr, ytr, etr, base.prior_values(mtr, table))
        vx = (xva, yva, eva, base.prior_values(mva, table))
        target = dest / (arm + "_" + str(args.seed) + ".csv")
        wrote = train_arm(arm, net, tx, vx, target, meta, args.seed)
        done.append(arm)
        print("completed", args.subset, args.fold, arm, args.seed, "trained" if wrote else "cached", flush=True)

    (dest / ("done_" + str(args.seed) + ".json")).write_text(json.dumps(dict(
        subset=args.subset, fold=args.fold, seed=args.seed, arms=done,
        n_train_units=len(train), n_valid_units=len(valid), n_train_windows=len(xtr),
        n_valid_windows=len(xva), epochs=EPOCHS, batch=BATCH,
        train_sha256=hashlib.sha256((ROOT / "data/CMAPSS" / f"train_{args.subset}.txt").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
