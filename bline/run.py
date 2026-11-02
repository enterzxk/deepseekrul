"""One run of the B-line matrix.

    python bline/run.py --subset FD002 --variant cai --hold 2 --fold 0 --seed 2027

``--hold -1`` is the in-distribution setting (all six setpoints in training).
``--hold c`` is leave-one-setpoint-out: training rows at setpoint c are re-rendered
onto a seen setpoint with the interpolation map, and the whole test set is
re-rendered onto c, so the model never sees c in training and is scored on nothing
but c.  Probe B4 verified that re-rendering preserves the health state and
therefore the label (0.024-0.031 standardised units).

Representations are built **row by row, before windowing**.  The setpoint is a
per-cycle attribute (82% of consecutive rows differ), so applying one setpoint's
map to a whole 40-cycle window would mix channels from six observation models;
that was the first version of this file and it silently destroyed the very
structure CAI is meant to expose.

Writes one JSON per run so a crash costs a single cell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

import data as D          # noqa: E402
import nets as N          # noqa: E402
import torchdata as T     # noqa: E402

OUT = ROOT / "experiments" / "bline" / "runs"
FOLDS = 5
TARGET_PARAMS = int(os.environ.get("BL_PARAMS", 120_000))
WINDOW = int(os.environ.get("BL_WINDOW", 40))
PATIENCE = int(os.environ.get("BL_PATIENCE", 30))
LR = float(os.environ.get("BL_LR", 1e-3))
BATCH = 512


def nasa(e: np.ndarray) -> float:
    return float(np.sum(np.where(e < 0, np.exp(-e / 13.0) - 1.0, np.exp(e / 10.0) - 1.0)))


def setup(sub: str, hold: int, seed: int) -> dict:
    tr = D.load(sub, "train")
    te = D.load(sub, "test")
    sc = D.setting_scale([tr])
    cen = D.fit_conditions([tr], k=6)
    lab_tr = D.assign_conditions(tr, cen, sc)
    lab_te = D.assign_conditions(te, cen, sc)
    mu, sd = tr.x.mean(0), np.where(tr.x.std(0) > 1e-9, tr.x.std(0), 1.0)
    z_tr, z_te = (tr.x - mu) / sd, (te.x - mu) / sd

    conds = list(range(6)) if hold < 0 else [c for c in range(6) if c != hold]
    maps, _ = T.fit_maps(z_tr, tr, lab_tr, conds)
    maps_true = dict(maps)
    if hold >= 0:
        # the map the method must use is interpolated in the settings; the map
        # the world actually applies is fitted on the held-out rows themselves.
        # Keeping both separates "can it extrapolate" from "is it self-consistent".
        maps_true[hold] = T.fit_one_map(z_tr, tr, lab_tr, hold, conds)
        maps[hold] = T.interpolate_map(maps, cen, sc, cen[hold])

    # Rows are replayed under a different setpoint, so their setpoint label must
    # move with them: inverting a re-rendered row with its ORIGINAL label applies
    # the wrong observation model and produces a representation the model never
    # saw.  ``lab_fit_eff`` / ``lab_test_eff`` carry the post-replay labels.
    z_fit, lab_fit_eff = z_tr.copy(), lab_tr.copy()
    if hold >= 0:
        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(lab_tr == hold)
        drawn = rng.choice(conds, size=len(idx))
        lab_fit_eff[idx] = drawn
        for c in conds:
            sel = idx[drawn == c]
            if len(sel):
                z_fit[sel] = T.render(z_tr[sel], hold, maps_true[c][0], maps_true[c][1],
                                      maps_true)
    # The test set is left exactly as recorded.  Rendering it with the same map
    # the method inverts with would make the round trip an identity and prove
    # nothing; the interpolation error has to show up at test time.
    z_test, lab_test_eff = z_te, lab_te

    cstat, sstat = {}, {}
    for c in conds:
        m = lab_tr == c
        cstat[c] = tr.x[m].mean(0)
        sstat[c] = np.where(tr.x[m].std(0) > 1e-9, tr.x[m].std(0), 1.0)
    if hold >= 0:
        seen = sorted(cstat)
        Xd = np.column_stack([np.ones(len(seen)), cen[seen] / sc])
        xt = np.column_stack([np.ones(1), (cen[hold] / sc)[None, :]])
        cc, *_ = np.linalg.lstsq(Xd, np.stack([cstat[c] for c in seen]), rcond=None)
        ss, *_ = np.linalg.lstsq(Xd, np.stack([sstat[c] for c in seen]), rcond=None)
        cstat[hold], sstat[hold] = (xt @ cc)[0], (xt @ ss)[0]

    return dict(tr=tr, te=te, lab_tr=lab_tr, lab_te=lab_te, mu=mu, sd=sd,
                maps=maps, maps_true=maps_true, z_fit=z_fit, z_test=z_test, lab_fit_eff=lab_fit_eff,
                lab_test_eff=lab_test_eff, cen=cen, sc=sc,
                cstat=cstat, sstat=sstat)


def transform_rows(variant: str, z: np.ndarray, lab: np.ndarray, P: dict) -> np.ndarray:
    """Per-row map into the variant's input space.  ``z`` is (N, 21)."""
    if variant in ("erm", "erm_u", "dann", "coral"):
        return z
    if variant in ("cai", "cai_w"):
        s = T.invert(z, lab, P["maps"])
        if variant == "cai":
            return s
        w = np.empty_like(s)
        for c, (a, b) in P["maps"].items():
            m = lab == c
            if m.any():
                w[m] = np.abs(T.safe_gain(b))
        return np.concatenate([s, s * w], axis=1)
    if variant == "condnorm":
        raw = z * P["sd"] + P["mu"]
        out = np.empty_like(raw)
        for c in range(6):
            m = lab == c
            if m.any():
                out[m] = (raw[m] - P["cstat"][c]) / P["sstat"][c]
        return out
    raise ValueError(variant)


def windows(f, z, stride=1, last_only=False):
    """Sliding windows over an already transformed per-row array."""
    L = WINDOW
    X, U, Y, E = [], [], [], []
    for e in f.engines():
        idx = f.index_of(e)
        idx = idx[np.argsort(f.cycle[idx])]
        n = len(idx)
        if n < L:
            idx = np.concatenate([np.repeat(idx[:1], L - n), idx])
            n = L
        starts = [n - L] if last_only else list(range(0, n - L + 1, stride))
        for s in starts:
            sl = idx[s: s + L]
            X.append(z[sl])
            U.append(f.u[sl])
            Y.append(f.rul[sl[-1]])
            E.append(e)
    return (np.asarray(X, np.float32), np.asarray(U, np.float32),
            np.asarray(Y, np.float32), np.asarray(E, np.int32))


def add_settings(X, U):
    return np.concatenate([X, np.broadcast_to(U, X.shape[:2] + (3,))], axis=2)


def domain_labels(f, lab, stride=1):
    """Majority setpoint inside each training window, for the adversarial arms."""
    L = WINDOW
    out = []
    for e in f.engines():
        idx = f.index_of(e)
        idx = idx[np.argsort(f.cycle[idx])]
        n = len(idx)
        if n < L:
            idx = np.concatenate([np.repeat(idx[:1], L - n), idx])
            n = L
        sub = lab[idx]
        for s in range(0, n - L + 1, stride):
            out.append(np.bincount(sub[s: s + L], minlength=6).argmax())
    return np.asarray(out, np.int64)


def run(args) -> dict:
    import torch
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    P = setup(args.subset, args.hold, args.seed)
    tr, te = P["tr"], P["te"]

    eng = tr.engines()
    rng = np.random.default_rng(1000 + args.fold)
    perm = rng.permutation(len(eng))
    val_eng = eng[perm[: len(eng) // FOLDS]]

    zr_tr = transform_rows(args.variant, P["z_fit"], P["lab_fit_eff"], P)
    zr_te = transform_rows(args.variant, P["z_test"], P["lab_test_eff"], P)
    Xtr, Utr, ytr, Etr = windows(tr, zr_tr, stride=2)
    Xte, Ute, yte, Ete = windows(te, zr_te, last_only=True)
    if args.variant == "erm_u":
        Xtr, Xte = add_settings(Xtr, Utr), add_settings(Xte, Ute)
    is_val = np.isin(Etr, val_eng)

    x_tr, x_va, x_te = Xtr[~is_val], Xtr[is_val], Xte
    m_in = x_tr.mean((0, 1), keepdims=True)
    s_in = x_tr.std((0, 1), keepdims=True) + 1e-6
    x_tr, x_va, x_te = (x_tr - m_in) / s_in, (x_va - m_in) / s_in, (x_te - m_in) / s_in

    d_in = x_tr.shape[2]
    n_cond = 6 if args.variant in ("dann", "coral") else 0
    width, n_par = N.match_width(d_in, TARGET_PARAMS, n_cond=n_cond)
    model = N.TCN(d_in, width, n_cond=n_cond)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    def tt(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)

    Xt, yt = tt(x_tr), tt(np.minimum(ytr[~is_val], T.CAP))
    Xv, yv = tt(x_va), tt(np.minimum(ytr[is_val], T.CAP))
    Xe = tt(x_te)
    gdom = domain_labels(tr, P["lab_tr"], stride=2)
    gtr = torch.from_numpy(gdom[~is_val]).long().to(dev)

    best, best_state, bad = 1e9, None, 0
    n = len(Xt)
    ep = 0
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            b = order[i: i + BATCH]
            if len(b) < 32:
                continue
            yhat, dlogit, f = model(Xt[b], lam=1.0 if args.variant == "dann" else 0.0)
            loss = torch.nn.functional.mse_loss(yhat, yt[b])
            if args.variant == "dann":
                loss = loss + 0.1 * torch.nn.functional.cross_entropy(dlogit, gtr[b])
            elif args.variant == "coral":
                loss = loss + 1.0 * N.coral_penalty(f, gtr[b], 6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv)[0]
        v = float(torch.sqrt(torch.nn.functional.mse_loss(pv, yv)))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xe)[0].cpu().numpy()
    pred = np.clip(pred, 0, T.CAP)
    truth = np.minimum(yte, T.CAP)
    err = pred - truth

    sched_res = {}
    if args.hold < 0:
        for target in range(6):
            zt = np.empty_like(P["z_test"])
            for c in range(6):
                k = P["lab_te"] == c
                if k.any():
                    zt[k] = T.render(P["z_test"][k], c, P["maps"][target][0],
                                     P["maps"][target][1], P["maps"])
            sched_res[target] = _score(model, tt, args.variant, te, zt,
                                       np.full(len(zt), target), P, m_in, s_in, yte)
        cyc = te.cycle
        tgt = cyc % 6
        zt = np.empty_like(P["z_test"])
        for src in range(6):
            for t in range(6):
                k = (P["lab_te"] == src) & (tgt == t)
                if k.any():
                    zt[k] = T.render(P["z_test"][k], src, P["maps"][t][0],
                                     P["maps"][t][1], P["maps"])
        sched_res["alt"] = _score(model, tt, args.variant, te, zt, tgt, P, m_in, s_in, yte)

    return {"subset": args.subset, "variant": args.variant, "hold": args.hold,
            "fold": args.fold, "seed": args.seed, "width": width, "params": n_par,
            "d_in": int(d_in), "n_train": int(len(Xt)), "n_test": int(len(pred)),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "nasa": nasa(err), "bias": float(err.mean()),
            "val_rmse": best, "epochs": ep + 1, "secs": round(time.time() - t0, 1),
            "eng": te.engines().tolist(), "pred": pred.round(3).tolist(),
            "truth": truth.round(3).tolist(), "sched": sched_res}


def _score(model, tt, variant, te, z_rows, lab_rows, P, m_in, s_in, yte):
    zr = transform_rows(variant, z_rows, lab_rows, P)
    X, U, _, _ = windows(te, zr, last_only=True)
    if variant == "erm_u":
        X = add_settings(X, U)
    X = (X - m_in) / s_in
    import torch
    model.eval()
    with torch.no_grad():
        pr = model(tt(X))[0].cpu().numpy()
    pr = np.clip(pr, 0, T.CAP)
    e = pr - np.minimum(yte, T.CAP)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()),
            "nasa": nasa(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--variant", default="cai", choices=N.VARIANTS)
    ap.add_argument("--hold", type=int, default=-1)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    r = run(args)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{r['subset']}_{r['variant']}_h{r['hold']}_f{r['fold']}_s{r['seed']}"
    (OUT / f"{tag}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ("subset", "variant", "hold", "fold", "seed",
                                       "rmse", "mae", "nasa", "bias", "val_rmse",
                                       "epochs", "params")}))


if __name__ == "__main__":
    main()
