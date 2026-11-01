"""Train the frozen N-CMAPSS backbone and write out-of-fold dev / test predictions.

Architecture and training schedule are copied from paper/NCMAPSS_PROTOCOL.md without
adjustment: 54-64-32-1 MLP, tanh, dropout 0.1, MSE, Adam lr 1e-3, batch 64, exactly 300
epochs, no early stopping. Early stopping would need a validation split, and the only
honest validation split here is the calibration sample itself.

Dev predictions are leave-one-unit-out: every dev row is scored by models that saw none
of its unit. Test predictions come from an ensemble over all dev units of the same fleet.

Standardisation statistics are estimated on the training fold only; using the whole dev
set would leak the held-out unit's feature scale into its own prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

SEEDS = (0, 1, 2)
EPOCHS = 300
BATCH = 64
LR = 1e-3
HID = (64, 32)
DROPOUT = 0.1
YMAX = 100.0


class Backbone(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, HID[0]), nn.Tanh(), nn.Dropout(DROPOUT),
            nn.Linear(HID[0], HID[1]), nn.Tanh(),
            nn.Linear(HID[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_one(X, y, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    net = Backbone(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = len(Xs)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xs[idx]), yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    return net, mu, sd


def predict(net, mu, sd, X):
    with torch.no_grad():
        z = torch.tensor((X - mu) / sd, dtype=torch.float32)
        return net(z).numpy().astype(np.float64)


def ensemble_predict(Xtr, ytr, Xte):
    ps = []
    for s in SEEDS:
        net, mu, sd = fit_one(Xtr, ytr, s)
        ps.append(predict(net, mu, sd, Xte))
    return np.clip(np.mean(ps, axis=0), 0, YMAX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/ncmapss/data")
    ap.add_argument("--out", default="experiments/ncmapss/preds")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, quality = [], {}
    t0 = time.time()
    for dev_f in sorted(data.glob("*_dev.npz")):
        fleet = dev_f.name.replace("_dev.npz", "")
        te_f = data / f"{fleet}_test.npz"
        dev = np.load(dev_f)
        te = np.load(te_f)
        Xd, yd, ud, cd = dev["X"], dev["y"], dev["unit"], dev["cyc"]

        # leave-one-unit-out over dev
        yhat_dev = np.zeros(len(yd))
        for u in np.unique(ud):
            m = ud == u
            yhat_dev[m] = ensemble_predict(Xd[~m], yd[~m], Xd[m])
        rmse_dev = float(np.sqrt(np.mean((yhat_dev - yd) ** 2)))

        # all dev -> test
        yhat_te = ensemble_predict(Xd, yd, te["X"])
        rmse_te = float(np.sqrt(np.mean((yhat_te - te["y"]) ** 2)))

        for split, uu, cc, yy, pp in [
                ("dev", ud, cd, yd, yhat_dev),
                ("test", te["unit"], te["cyc"], te["y"], yhat_te)]:
            for a_, b_, c_, d_ in zip(uu, cc, yy, pp):
                rows.append(dict(fleet=fleet, split=split, unit=int(a_), cyc=int(b_),
                                 y=float(c_), yhat=float(d_)))
        quality[fleet] = dict(
            n_dev=int(len(yd)), n_test=int(len(te["y"])),
            units_dev=int(len(np.unique(ud))), units_test=int(len(np.unique(te["unit"]))),
            rmse_dev_oof=round(rmse_dev, 3), rmse_test=round(rmse_te, 3),
            sd_y_dev=round(float(yd.std()), 3), sd_y_test=round(float(te["y"].std()), 3),
            r2_dev=round(1 - rmse_dev ** 2 / float(yd.var()), 4),
            r2_test=round(1 - rmse_te ** 2 / float(te["y"].var()), 4),
            mean_bias_test=round(float((yhat_te - te["y"]).mean()), 3),
        )
        print(f"{fleet:6s} dev n={len(yd):4d} RMSE={rmse_dev:6.2f} R2={quality[fleet]['r2_dev']:6.3f} | "
              f"test n={len(te['y']):4d} RMSE={rmse_te:6.2f} R2={quality[fleet]['r2_test']:6.3f} "
              f"bias={quality[fleet]['mean_bias_test']:+6.2f}")

    with open(out / "preds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fleet", "split", "unit", "cyc", "y", "yhat"])
        w.writeheader()
        w.writerows(rows)
    (out / "quality.json").write_text(json.dumps(quality, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} rows -> {out/'preds.csv'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
