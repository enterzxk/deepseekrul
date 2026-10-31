"""ECS evaluation: fused RUL distribution on the official test sets (H1, H2).

Fusion: p(R | w, a) ~ p0(R | a) * p1(R | w)^alpha, with
  alpha = clip((beta0 + beta_kappa*kappa + beta_rho*rho) / alpha_scale, 0, 1)
where the beta come from the Stage-B cross-fitted regression (frozen before this
script is applied to test data) and alpha_scale is fixed at 2.0 nats (a log-density
gain of 2 corresponds to an e^2 ~ 7.4x likelihood advantage, treated as "full trust").

Comparators: alpha = 0 (prior only), alpha = 1 (likelihood only), alpha = 0.5.

Usage: python ecs_eval.py --subset FD002 --seed 2027
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

ROOT = "/home/yndx/zxk/bearing-rul-pi"
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/scripts")

import dsh_e21_test_all as e21
import dsh_e22a_full_test as e22a
import run_candidate as rc
from ecs.prior import AgeConditionalPrior
from ecs.likelihood import BinLikelihood, N_BINS
from dataset import CMAPSS_COLUMNS, SELECTED_SENSOR_COLUMNS

ALPHA_SCALE = 2.0


def load_models(subset, seed, folds=(0, 1, 2, 3, 4)):
    ms, priors = [], []
    for f in folds:
        d = f"{ROOT}/experiments/ecs/{subset}/f{f}_s{seed}"
        ck = torch.load(f"{d}/like.pt", map_location="cpu")
        m = BinLikelihood(); m.load_state_dict(ck["model"]); m.eval()
        ms.append(m)
        meta = json.load(open(f"{d}/metrics.json"))
        priors.append((meta["prior"]["k"], meta["prior"]["lam"]))
    return ms, priors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()
    device = "cpu"
    stB = json.load(open(f"{ROOT}/results/ecs/stageB_{a.subset}_s{a.seed}.json"))
    b0, bk, br = stB["beta"]["intercept"], stB["beta"]["kappa"], stB["beta"]["rho"]

    ms, priors = load_models(a.subset, a.seed)
    ks = np.array([p[0] for p in priors]); lams = np.array([p[1] for p in priors])

    traj = rc.load_cmapss_train(os.path.join(ROOT, "data", "CMAPSS"), a.subset)
    del traj
    import pandas as pd
    t = pd.read_csv(f"{ROOT}/data/CMAPSS/test_{a.subset}.txt", sep=r"\s+", header=None)
    t.columns = list(CMAPSS_COLUMNS)
    rul = pd.read_csv(f"{ROOT}/data/CMAPSS/RUL_{a.subset}.txt", sep=r"\s+",
                      header=None)[0].to_numpy(float)
    lengths = t.groupby("unit").cycle.max()
    cols = list(SELECTED_SENSOR_COLUMNS)

    # one prior per fold: average the per-fold parameters on the log scale
    prior = AgeConditionalPrior.__new__(AgeConditionalPrior)
    prior.cap, prior.n_bins = 125.0, N_BINS
    prior.edges = np.linspace(0.0, 125.0, N_BINS + 1)
    prior.k, prior.lam = float(np.exp(np.log(ks).mean())), float(np.exp(np.log(lams).mean()))

    recs = []
    for fold, m in enumerate(ms):
        _, tr, _, _ = e21.split_frames(a.subset, fold)
        scaler = e21.fit_scaler(a.subset, tr)
        ts = e21.scale_frame(a.subset, scaler, t)
        x, extras, units, _ = e22a.build_full_features(ts, t)
        with torch.no_grad():
            lg = m(torch.from_numpy(x), cond=None, extras=torch.from_numpy(extras))
            lp = torch.log_softmax(lg, -1).numpy()
        for i, u in enumerate(units):
            L = int(lengths[u])
            kappa = float(min(1.0, L / 50.0))
            g = t[t.unit == u]
            win = g.iloc[-50:][cols].to_numpy(float)
            tt = np.arange(win.shape[0], dtype=float)
            tt = (tt - tt.mean()) / (tt.std() + 1e-9)
            sl = (win * tt[:, None]).sum(0) / win.shape[0]
            mag = np.abs(sl)
            rho = float(np.clip(1 - mag.std() / (mag.mean() + 1e-9), 0, 1)) if mag.mean() > 1e-6 else 0.0
            recs.append(dict(fold=fold, unit=int(u), L=L, kappa=kappa, rho=rho,
                             y=float(min(rul[u - 1], 125.0)), y_raw=float(rul[u - 1]),
                             lp=lp[i].tolist()))

    import pandas as pd
    df = pd.DataFrame(recs)
    centres = np.linspace(125.0 / N_BINS / 2, 125.0 - 125.0 / N_BINS / 2, N_BINS)
    out = {}
    for label, alpha_mode in [("alpha=0", "zero"), ("alpha=1", "one"),
                              ("alpha=0.5", "half"), ("ctsf", "fit")]:
        preds, q05, q95 = [], [], []
        for _, r in df.iterrows():
            p0 = prior.prob(r["L"])
            l1 = np.array(r["lp"]); p1 = np.exp(l1)
            if alpha_mode == "zero":
                al = 0.0
            elif alpha_mode == "one":
                al = 1.0
            elif alpha_mode == "half":
                al = 0.5
            else:
                dhat = b0 + bk * r["kappa"] + br * r["rho"]
                al = float(np.clip(dhat / ALPHA_SCALE, 0.0, 1.0))
            p = p0 * np.exp(al * l1); p /= p.sum()
            preds.append(float((p * centres).sum()))
            c = np.cumsum(p)
            q05.append(float(centres[min(int(np.searchsorted(c, 0.05)), N_BINS - 1)]))
            q95.append(float(centres[min(int(np.searchsorted(c, 0.95)), N_BINS - 1)]))
        deltas = []
        df[f"pred_{label}"] = preds
        err = np.array(preds) - df["y"].to_numpy()
        # short-trajectory group: engines that needed padding
        short = df["L"].to_numpy() < 50
        out[label] = dict(
            rmse=float(np.sqrt((err ** 2).mean())),
            rmse_short=float(np.sqrt((err[short] ** 2).mean())) if short.any() else None,
            n_short=int(short.sum()),
            coverage=float(np.mean((np.array(q05) <= df["y"]) & (df["y"] <= np.array(q95)))),
        )
        print(f"{label:8s} rmse={out[label]['rmse']:.3f} "
              f"short={out[label]['rmse_short']} (n={out[label]['n_short']}) "
              f"cov90={out[label]['coverage']:.2f}", flush=True)

    os.makedirs(f"{ROOT}/results/ecs", exist_ok=True)
    df.to_csv(f"{ROOT}/results/ecs/eval_{a.subset}_s{a.seed}.csv", index=False)
    json.dump(out, open(f"{ROOT}/results/ecs/eval_{a.subset}_s{a.seed}.json", "w"), indent=2)
    print("wrote results/ecs/eval_%s_s%d.json" % (a.subset, a.seed))


if __name__ == "__main__":
    main()
