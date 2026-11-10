"""A2 readout intervention: does the RUL head actually use the degradation state d_T?

Loads a trained A2 checkpoint and recomputes TEST predictions under three
readouts of the same model:
  real      d_T, r_T as computed
  mean_d    d_T replaced by the mean d_T over the test windows
  shuffle_d d_T randomly permuted across test windows

If the head ignores d_T, all three give identical predictions. A large
real-vs-mean_d gap is the cheapest evidence that d_T carries the prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ida as R  # noqa: E402
from run_a2 import A2Model, split_es_test  # noqa: E402

CAP = 125.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--prior-dir", default="experiments/ida/prior_boot")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, df = R.outer_folds(args.subset)
    tr_u, va_u = folds[args.fold]
    prior_dir = ROOT / args.prior_dir
    meta_p = json.loads((prior_dir / f"{args.subset}_prior_meta.json").read_text())
    table = pd.read_csv(prior_dir / f"{args.subset}_prior_cutoffs.csv")
    median_T = float(meta_p["folds"][str(args.fold)]["median_T_train"])
    sp = R.StaircasePrior(table, {int(k): v for k, v in ((fk, fm["median_T_train"]) for fk, fm in meta_p["folds"].items())}, args.subset)
    (_, _, _, mtr, _), (xva, yva, eva, mva, cva) = R.prep_fold(df, args.subset, args.fold, tr_u, va_u, window=args.window)
    pva, _ = R.prior_vectors(mva, sp, args.fold, "M1", median_T)
    T = df.groupby("unit")["cycle"].max().to_dict()
    es_u, te_u = split_es_test(va_u, T)
    te_idx = np.isin(mva[:, 0], te_u)

    loader = DataLoader(R.WindowSet(xva[te_idx], yva[te_idx], cva[te_idx], eva[te_idx], pva[te_idx]),
                        batch_size=512, shuffle=False)

    body = R.CandidateModel(num_sensors=14, window_size=args.window, d_model=64, num_layers=1, dropout=0.15,
                            use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True,
                            use_graph=False, use_dual_head=False)
    model = A2Model(body).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ys, d_real, r_real, ctxs = [], [], [], []
    with torch.no_grad():
        for x, y, cond, extras, prior in loader:
            cond = cond.to(device) if cond is not None else None
            pooled = model.body.encode(x.to(device), cond=cond, extras=extras.to(device))
            ctx = pooled.mean(dim=1) + model.proj(prior.to(device))
            w = torch.nn.functional.softplus(model.w_delta(pooled).squeeze(-1))
            rate = torch.nn.functional.softplus(model.b_rate + model.gamma(cond).squeeze(-1)) if cond is not None else torch.ones_like(w)
            d = torch.cumsum(w * rate, dim=1)[:, -1]
            doc = torch.zeros_like(cond) if cond is not None else torch.zeros(x.shape[0], x.shape[1], 3, device=device)
            if cond is not None:
                doc[:, 1:] = cond[:, 1:] - cond[:, :-1]
            eta = model.eta(doc).squeeze(-1)
            rho = torch.sigmoid(model.rho_raw)
            W = doc.shape[1]
            idx = torch.arange(W, device=device)
            diff = (idx[:, None] - idx[None, :]).clamp(min=0).to(eta.dtype)
            wmat = torch.where(idx[:, None] >= idx[None, :], rho.pow(diff), torch.zeros_like(diff))
            r = torch.einsum("tw,bw->bt", wmat, eta)[:, -1]
            ys.append(y.numpy()); d_real.append(d.cpu().numpy()); r_real.append(r.cpu().numpy())
            ctxs.append(ctx.cpu().numpy())

    y_all = np.concatenate(ys); d_all = np.concatenate(d_real); r_all = np.concatenate(r_real)
    ctx_all = np.concatenate(ctxs)

    def predict(d_in, r_in):
        with torch.no_grad():
            feat = torch.cat([torch.from_numpy(ctx_all), torch.from_numpy(d_in[:, None]), torch.from_numpy(r_in[:, None])], dim=-1).to(device)
            return model.head(feat).squeeze(-1).cpu().numpy()

    rng = np.random.default_rng(0)
    variants = {
        "real": predict(d_all, r_all),
        "mean_d": predict(np.full_like(d_all, d_all.mean()), r_all),
        "shuffle_d": predict(d_all[rng.permutation(len(d_all))], r_all),
    }
    units = mva[te_idx][:, 0]
    out = {}
    for k, yh in variants.items():
        e = yh - y_all
        g = pd.DataFrame({"u": units, "e2": e ** 2}).groupby("u")["e2"].mean()
        out[k] = dict(rmse_macro=float(np.sqrt(g.mean())),
                      delta_vs_real_pct=float((np.sqrt(g.mean()) / np.sqrt(pd.DataFrame({"u": units, "e2": (variants["real"] - y_all) ** 2}).groupby("u")["e2"].mean().mean()) - 1) * 100))
    out["d_std"] = float(d_all.std())
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
