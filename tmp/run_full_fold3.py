"""Representation screen (v3): same backbone/budget, different observation encodings.

Arms
----
M0      condition-normalised absolute window (reference)
DIFF    window minus the engine's own early-life condition-matched reference
        (extras[:, :14] is exactly that reference), i.e. a differential measurement
PHASE   remaining-fraction target with the lifetime scale applied at the output
EVID_AGE  prior injected through an age-gated scalar weight (sigmoid(a*c+b))
EVID_DEV  prior injected through a condition-matched-deviation-gated weight

--window changes the observation length (default 50) and therefore the PE table size.
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


def build_net(arm, window):
    body = ida.CandidateModel(num_sensors=14, window_size=window, d_model=64, num_layers=1,
                              dropout=0.15, use_baseline=True, use_ema_features=True,
                              use_cycle=True, use_pe=True, use_graph=False,
                              use_dual_head=False)
    if arm == "M0" or arm == "DIFF" or arm == "PHASE":
        return ida.PriorWrapper(body, "M0")
    if arm in ("EVID_AGE", "EVID_DEV"):
        return GatedPrior(body, arm)
    raise ValueError(arm)


class GatedPrior(torch.nn.Module):
    """Prior projection multiplied by a learned scalar gate on one observable."""

    def __init__(self, body, arm):
        super().__init__()
        self.body = body
        self.arm = arm
        self.proj = torch.nn.Linear(2, 64)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)
        self.gate = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.constant_(self.gate.bias, 0.0)   # gate starts at 0.5

    def forward(self, x, cond, extras, prior):
        pooled = self.body.encode(x, cond=cond, extras=extras)
        context = pooled.mean(dim=1)
        z = extras[:, 14:15] if self.arm == "EVID_AGE" else extras[:, 18:19]
        w = torch.sigmoid(self.gate(z))
        return {"prediction": self.body.reg_head(context + w * self.proj(prior))}


def train_arm(arm, net, train_data, valid_data, target, metadata, seed, save_path=None):
    if target.exists():
        return False
    ida.set_seed(seed)
    g = torch.Generator(device="cuda").manual_seed(seed)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    tr = [base.device_array(a) for a in train_data[:4]]
    phase = arm == "PHASE"
    diff = arm == "DIFF"

    def forward(data, idx):
        x, y, extras, prior = (v[idx] for v in data[:4])
        if diff:
            x = x - extras[:, None, :14]
        out = net(x, None, extras, prior)["prediction"].reshape(-1)
        if phase:
            return out * prior[:, 0].clamp_min(1.0) * 250.0
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
        raw = torch.cat([net(va[0][i] - va[2][i][:, None, :14] if diff else va[0][i], None, va[2][i], va[3][i])
                         ["prediction"].reshape(-1)
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
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--arms", default="M0,DIFF,PHASE")
    args = ap.parse_args()

    dest = HERE / f"runs3_w{args.window}" / args.subset / str(args.fold)
    dest.mkdir(parents=True, exist_ok=True)
    folds, df = ida.outer_folds(args.subset)
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    train, valid = folds[args.fold]
    tables = base.prior_tables(df, args.subset, train, valid, args.fold)
    tr, va = ida.prep_fold(df, args.subset, args.fold, train, valid, window=args.window)
    xtr, ytr, etr, mtr = tr[:4]
    xva, yva, eva, mva = va[:4]
    lives = df.groupby("unit").cycle.max()
    meta = pd.DataFrame(mva, columns=["unit", "end_cycle"])
    meta["short"] = meta.unit.map(lives) <= np.quantile(lives.loc[sorted(train)], .25)
    meta["true_life"] = meta.unit.map(lives)

    for arm in [a for a in args.arms.split(",") if a]:
        net = build_net(arm, args.window)
        ida.set_seed(args.seed)
        net = net.cuda()
        tx = [xtr, ytr, etr, base.prior_values(mtr, tables["FULL"])]
        vx = [xva, yva, eva, base.prior_values(mva, tables["FULL"])]
        target = dest / f"{arm}_{args.seed}.csv"
        wrote = train_arm(arm, net, tx, vx, target, meta, args.seed, dest / f"{arm}_{args.seed}.pt")
        print("completed", args.subset, args.fold, arm, args.window, "trained" if wrote else "cached", flush=True)

    (dest / f"done_{args.seed}.json").write_text(json.dumps(dict(
        subset=args.subset, fold=args.fold, seed=args.seed, window=args.window,
        arms=[a for a in args.arms.split(",") if a], n_train_windows=len(xtr),
        n_valid_windows=len(xva), epochs=EPOCHS, batch=BATCH,
        train_sha256=hashlib.sha256((ROOT / "data/CMAPSS" / f"train_{args.subset}.txt").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
