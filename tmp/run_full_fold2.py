"""Full-budget outer-fold runner v2: adds PHASE arms, prior logging and weight saving.

Arms
----
M0 OPS FULL NORM            -- identical to run_full_fold.py
NORM_SHUF NORM_CONST NORM_OPS -- NORM controls (mu_T permuted / constant / from age prior)
PHASE                       -- predict the remaining life fraction phi = (T-c)/T and
                               output RUL = phi * mu_T; loss is still the asymmetric RUL
                               loss in cycles (identical to the other arms)
PHASE_SHUF PHASE_CONST      -- inference-time controls for PHASE: same trained network,
                               mu_T replaced by a permuted / constant value

Every prediction CSV carries mu_T and the raw network output so that the PHASE controls
can be recomputed without retraining.  The state dict is saved for measurement-condition
experiments (corrupted inputs at evaluation only).
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
PARENT = HERE.parent
ROOT = PARENT.parent
sys.path[:0] = [str(PARENT), str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
import run_fold as base
import run_ida as ida

EPOCHS = 80
BATCH = 256
PE_ARMS = ("M0", "OPS", "FULL", "NORM", "NORM_SHUF", "NORM_CONST", "NORM_OPS")
PHASE_ARMS = ("PHASE", "PHASE_SHUF", "PHASE_CONST")


def shuffled_table(table, seed):
    units = sorted({u for u, _ in table})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(units))
    mapping = {u: units[perm[i]] for i, u in enumerate(units)}
    return {(mapping[u], c): v for (u, c), v in table.items()}


def constant_table(table):
    value = float(np.median(list(table.values())))
    return {key: value for key in table}


def build_body():
    return ida.CandidateModel(num_sensors=14, window_size=50, d_model=64, num_layers=1,
                              dropout=0.15, use_baseline=True, use_ema_features=True,
                              use_cycle=True, use_pe=True, use_graph=False,
                              use_dual_head=False)


def build_arm(arm):
    if arm in ("M0", "OPS", "FULL"):
        return ida.PriorWrapper(build_body(), "M0" if arm == "M0" else "M1")
    if arm in ("NORM", "NORM_SHUF", "NORM_CONST", "NORM_OPS"):
        return ida.PriorWrapper(build_body(), "M0", norm=True)
    if arm in PHASE_ARMS:
        return ida.PriorWrapper(build_body(), "M0")
    raise ValueError(arm)


def train_arm(arm, net, train_data, valid_data, target, metadata, seed, save_path=None):
    if target.exists():
        return False
    ida.set_seed(seed)
    g = torch.Generator(device="cuda").manual_seed(seed)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    tr = [base.device_array(a) for a in train_data]
    phase = arm in PHASE_ARMS
    life = tr[4] if phase else None                     # total life per training window

    def forward(data, idx):
        x, y, extras, prior = (v[idx] for v in data[:4])
        out = net(x, None, extras, prior)["prediction"].reshape(-1)
        if phase:
            mu = prior[:, 0].clamp_min(1.0) * 250.0
            return out * mu
        return out

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
    va = [base.device_array(a) for a in valid_data[:4]]
    net.eval()
    with torch.no_grad():
        raw = torch.cat([net(va[0][i], None, va[2][i], va[3][i])["prediction"].reshape(-1)
                         for i in torch.arange(len(va[0]), device="cuda").split(BATCH)])
    raw = raw.cpu().numpy()
    assert np.isfinite(raw).all()
    mu = valid_data[3][:, 0].astype(np.float64) * 250.0
    pred = raw * mu if phase else raw
    frame = metadata.assign(y_true=valid_data[1], y_pred=pred, raw_output=raw, mu_T=mu)
    tmp = target.with_suffix(".partial")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, target)
    if save_path is not None:
        torch.save(net.state_dict(), save_path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset")
    ap.add_argument("fold", type=int)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--arms", default="M0,OPS,FULL,NORM,PHASE")
    args = ap.parse_args()

    dest = HERE / "runs2" / args.subset / str(args.fold)
    dest.mkdir(parents=True, exist_ok=True)
    folds, df = ida.outer_folds(args.subset)
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    train, valid = folds[args.fold]
    tables = base.prior_tables(df, args.subset, train, valid, args.fold)
    tables["M0"] = tables["FULL"]
    tables["NORM_SHUF"] = shuffled_table(tables["FULL"], ida.OUTER_SEED + 991 * (args.fold + 1) + args.seed)
    tables["NORM_CONST"] = constant_table(tables["FULL"])
    tables["NORM_OPS"] = tables["OPS"]
    tables["PHASE"] = tables["FULL"]
    tables["PHASE_SHUF"] = tables["NORM_SHUF"]
    tables["PHASE_CONST"] = tables["NORM_CONST"]

    tr, va = ida.prep_fold(df, args.subset, args.fold, train, valid)
    xtr, ytr, etr, mtr = tr[:4]
    xva, yva, eva, mva = va[:4]
    lives = df.groupby("unit").cycle.max()
    meta = pd.DataFrame(mva, columns=["unit", "end_cycle"])
    meta["short"] = meta.unit.map(lives) <= np.quantile(lives.loc[sorted(train)], .25)
    meta["true_life"] = meta.unit.map(lives)
    life_tr = mtr[:, 0].astype(np.float32)  # placeholder, replaced below

    arms = [a for a in args.arms.split(",") if a]
    for arm in arms:
        net = build_arm(arm)
        ida.set_seed(args.seed)
        net = net.cuda()
        table = tables[arm]
        tx = [xtr, ytr, etr, base.prior_values(mtr, table)]
        vx = [xva, yva, eva, base.prior_values(mva, table)]
        if arm in PHASE_ARMS:
            tx.append(meta.unit.map(lives).to_numpy(dtype=np.float32))  # unused placeholder
            tx[4] = np.asarray([lives[u] for u in mtr[:, 0]], dtype=np.float32)
        else:
            tx.append(np.zeros(len(xtr), dtype=np.float32))
        target = dest / (arm + "_" + str(args.seed) + ".csv")
        weights = dest / (arm + "_" + str(args.seed) + ".pt")
        wrote = train_arm(arm, net, tx, vx, target, meta, args.seed, save_path=weights)
        print("completed", args.subset, args.fold, arm, args.seed, "trained" if wrote else "cached", flush=True)

    (dest / ("done_" + str(args.seed) + ".json")).write_text(json.dumps(dict(
        subset=args.subset, fold=args.fold, seed=args.seed, arms=arms,
        n_train_units=len(train), n_valid_units=len(valid), n_train_windows=len(xtr),
        n_valid_windows=len(xva), epochs=EPOCHS, batch=BATCH,
        train_sha256=hashlib.sha256((ROOT / "data/CMAPSS" / f"train_{args.subset}.txt").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
