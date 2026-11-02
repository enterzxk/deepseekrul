"""How much does the 66-parameter layer's reported coverage move when nothing changes?

Same dev rows, same objective, same architecture, same hyperparameters; only the torch
initialisation seed differs. Every fit is then scored on (a) the dev rows it was fitted on
and (b) the official test rows, both by randomized PIT at 0.90 with the same scale.

If the spread across seeds is comparable to the differences reported between calibration
methods, then a PICP table computed from one fit is not evidence about a method.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts" / "cal"))
from layer import fit_layer, interval, pmf_np  # noqa: E402
from calibrate import load_dev, load_test  # noqa: E402

K, MAX_RUL, SCALE = 126, 125.0, 0.9
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SEEDS = list(range(20))


def rand_cov(p, y, level=0.90, nrep=60, seed=7):
    cdf = np.cumsum(p, axis=1)
    prev = np.concatenate([np.zeros((len(p), 1)), cdf[:, :-1]], axis=1)
    y = np.asarray(y, int)
    idx = np.arange(len(y))
    a = (1 - level) / 2
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(nrep):
        u = prev[idx, y] + rng.random(len(y)) * p[idx, y]
        acc += float(((u >= a) & (u <= 1 - a)).mean())
    return acc / nrep


def main():
    stage = Path(os.environ.get("STAGE", str(ROOT)))
    every = {s: load_dev(stage, s, "A") for s in SUBSETS}
    dev = {k: np.concatenate([every[s][k] for s in SUBSETS]) for k in ("yhat", "y")}
    test = {s: load_test(stage, s, "A") for s in SUBSETS}

    rows = []
    for seed in SEEDS:
        net = fit_layer(dev, seed=seed)
        p_dev = pmf_np(*net.params_np(dev["yhat"]), mode="censored") if hasattr(net, "params_np") else None
        import torch
        with torch.no_grad():
            mu, sd = net.params(torch.tensor(dev["yhat"], dtype=torch.float64))
        p_dev = pmf_np(mu.numpy(), sd.numpy() * SCALE, "censored")
        rec = {"seed": seed, "dev90": round(rand_cov(p_dev, dev["y"]), 4)}
        for s in SUBSETS:
            with torch.no_grad():
                mu, sd = net.params(torch.tensor(test[s]["yhat"], dtype=torch.float64))
            p = pmf_np(mu.numpy(), sd.numpy() * SCALE, "censored")
            rec[s] = round(rand_cov(p, test[s]["y"]), 4)
        rows.append(rec)
        print(json.dumps(rec), flush=True)

    keys = ["dev90"] + list(SUBSETS)
    summary = {k: dict(mean=round(float(np.mean([r[k] for r in rows])), 4),
                       sd=round(float(np.std([r[k] for r in rows], ddof=1)), 4),
                       lo=round(float(np.min([r[k] for r in rows])), 4),
                       hi=round(float(np.max([r[k] for r in rows])), 4)) for k in keys}
    out = {"scale": SCALE, "n_seeds": len(SEEDS), "rows": rows, "summary": summary}
    (ROOT / "experiments" / "calx" / "layer_stability.json").write_text(json.dumps(out, indent=1))
    print("SUMMARY", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
