"""PCDT-RUL training driver.  One process = one (subset, arm, fold, seed).

Usage (from repo root, PYTHONPATH=src):
  python lsc/run.py --subset FD002 --arm full --fold 0 --seed 2027 --epochs 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lsc import data as D
from lsc import nets as N


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate(model, va, edges, device, lam_ext=None, bs=4096):
    """All-window metrics plus fixed-age diagnostics.

    Dev engines run to failure, so end-of-life windows are degenerate (RUL=0);
    the deployable diagnostics are (a) RMSE over all held-out windows and
    (b) RUL errors at fixed absolute ages c, mirroring the stage-1 gate.
    """
    model.eval()
    X = torch.from_numpy(va["X"]).to(device)
    Xs = torch.from_numpy(va["Xs"]).to(device)
    life = torch.from_numpy(va["life"]).to(device)
    cyc = torch.from_numpy(va["cyc"]).to(device)
    lam_e = (torch.from_numpy(lam_ext).to(device) if lam_ext is not None else None)
    mus, lam = [], []
    ext = lam_ext is not None
    for i in range(0, len(X), bs):
        mu, sg, ev = model(X[i:i + bs], life[i:i + bs], cyc[i:i + bs],
                           xs=Xs[i:i + bs], lam_ext=lam_e[i:i + bs] if lam_e is not None else None)
        mus.append(mu.cpu().numpy())
        # the hgb arm's bias consumes the external lambda, so that is the
        # scale whose quality must be reported -- not the idle scale head
        lam.append(lam_e[i:i + bs].cpu().numpy() if ext else ev["mu"].cpu().numpy())
    mus = np.concatenate(mus)
    lam = np.concatenate(lam)
    if not ext:
        lam = np.exp(lam)
    err = mus - va["rul"]
    out = {
        "rmse_all": float(np.sqrt(np.mean(err ** 2))),
        "nasa_all": D.nasa_score(err),
        "bias_all": float(err.mean()),
        "lam_corr_all": _corr(lam, va["life"]),
        "lam_mae_all": float(np.mean(np.abs(lam - va["life"]))),
        # per-engine aggregation happens at the summary stage
        "err_all": [{"eng": int(va["eng"][i]), "err": float(err[i]),
                     "rul": float(va["rul"][i]), "life": float(va["life"][i]),
                     "cyc": int(va["cyc"][i])} for i in range(len(err))],
    }
    # fixed-age diagnostics: windows ending exactly at c, uncapped RUL = T - c
    ages = {}
    for c in (50, 100, 125):
        m = va["cyc"] == c
        if m.sum() >= 20:
            rul_true = va["life"][m] - c
            e = mus[m] - rul_true
            ages[str(c)] = {"n": int(m.sum()),
                            "rmse": float(np.sqrt(np.mean(e ** 2))),
                            "mae": float(np.mean(np.abs(e))),
                            "bias": float(e.mean()),
                            "lam_corr": _corr(lam[m], va["life"][m]),
                            "lam_mae": float(np.mean(np.abs(lam[m] - va["life"][m])))}
    out["fixed_age"] = ages
    # lifetime-scale stratification over all windows
    T = va["life"]
    buckets, lo = {}, -np.inf
    for k, hi in enumerate(list(edges) + [np.inf]):
        m = (T >= lo) & (T < hi)
        if m.any():
            buckets[f"q{k}"] = {"n": int(m.sum()), "lo": float(lo), "hi": float(hi),
                                "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                                "bias": float(err[m].mean())}
        lo = hi
    out["bias_buckets"] = buckets
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--arm", required=True,
                    choices=("plain", "rel", "tau", "rand", "oracle", "full", "both", "hgb"))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--scale_head", default="gru", choices=("gru", "attn"))
    ap.add_argument("--detach_lam", action="store_true")
    ap.add_argument("--hgb_file", default="results/lsc_probe/fd002_predictions.json",
                    help="OOF gate predictions for the hgb arm")
    ap.add_argument("--out", default="experiments/lsc/runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 31))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va, scaler, edges = D.make_fold_data(args.subset, args.fold)
    model = N.PCDT(D.NF, ffn=args.ffn, arm=args.arm, L=D.WINDOW,
                   scale_head=args.scale_head, detach_lam=args.detach_lam).to(device)
    n_params = N.count_params(model)

    X = torch.from_numpy(tr["X"]).to(device)
    Xs = torch.from_numpy(tr["Xs"]).to(device)
    y = torch.from_numpy(tr["rul"]).to(device)
    life = torch.from_numpy(tr["life"]).to(device)
    cyc = torch.from_numpy(tr["cyc"]).to(device)
    # a window ending exactly at failure cannot condition on T > c
    lam_mask = (life > cyc)
    # hgb arm: external leak-free lifetime estimate per window
    lam_ext_tr = lam_ext_va = None
    if args.arm == "hgb":
        from lsc import hgb_lam as H
        lam_ext_tr = H.lam_windows(args.hgb_file, args.subset, tr["eng"], tr["cyc"])
        lam_ext_va = H.lam_windows(args.hgb_file, args.subset, va["eng"], va["cyc"])
        lam_ext_tr_t = torch.from_numpy(lam_ext_tr).to(device)
        print(f"[hgb] train lam corr with life: "
              f"{np.corrcoef(lam_ext_tr, tr['life'])[0, 1]:.3f}  "
              f"val: {np.corrcoef(lam_ext_va, va['life'])[0, 1]:.3f}", flush=True)
    else:
        lam_ext_tr_t = None

    # ---- warm-up: fit the scale head on the lifetime loss alone -----------
    # identical for every arm; the stage-1 gate showed lifetime information
    # grows with prefix length, and joint training from scratch left the head
    # lazy (pooled corr 0.2).  The long strided history is its input.
    warm = 25
    opt_s = torch.optim.AdamW(model.scale.parameters(), lr=2e-3)
    rng0 = np.random.default_rng(args.seed + 1)
    for _ in range(warm):
        perm = rng0.permutation(len(Xs))
        for i in range(0, len(Xs), args.batch):
            b = torch.from_numpy(perm[i:i + args.batch]).to(device)
            bl = b[lam_mask[b]]
            lm, ls = model.scale(Xs[bl])
            loss = N.trunc_lognorm_nll(lm, ls, life[bl], cyc[bl])
            opt_s.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.scale.parameters(), 5.0)
            opt_s.step()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    rng = np.random.default_rng(args.seed)
    best, bad, t0, ep = np.inf, 0, time.time(), 0
    best_val = None
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(X))
        for i in range(0, len(X), args.batch):
            b = torch.from_numpy(perm[i:i + args.batch]).to(device)
            le = lam_ext_tr_t[b] if lam_ext_tr_t is not None else None
            mu, sg, _ = model(X[b], life[b], cyc[b], xs=Xs[b], lam_ext=le)
            loss = N.gauss_nll(mu, sg, y[b])
            bl = b[lam_mask[b]]
            if len(bl):
                lm, ls = model.scale(Xs[bl])
                loss = loss + N.trunc_lognorm_nll(lm, ls, life[bl], cyc[bl])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        val = evaluate(model, va, edges, device, lam_ext=lam_ext_va)
        if val["rmse_all"] < best - 1e-4:
            best, bad, best_val = val["rmse_all"], 0, val
        else:
            bad += 1
            if bad >= args.patience:
                break
    best_val["epochs"] = ep + 1
    best_val["secs"] = round(time.time() - t0, 1)
    result = dict(subset=args.subset, arm=args.arm, fold=args.fold, seed=args.seed,
                  params=n_params, val_rmse=best, **best_val)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = ("_attn" if args.scale_head == "attn" else "") + ("_det" if args.detach_lam else "")
    name = out / f"{args.subset}_{args.arm}{tag}_f{args.fold}_s{args.seed}.json"
    name.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("subset", "arm", "fold", "seed", "params", "val_rmse",
                       "rmse_all", "lam_corr_all", "epochs", "secs")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
