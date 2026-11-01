"""Why does one arm need an extreme scale? Look at the development residual tails."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import ROOT, choose_scale, coverage_of, fit, load_dev, pmf_scaled  # noqa: E402

stage = Path(ROOT)
for arm in ("A", "BE", "PE"):
    devs = [load_dev(stage, s, arm) for s in ("FD001", "FD002", "FD003", "FD004")]
    pooled = {k: np.concatenate([d[k] for d in devs]) for k in ("yhat", "y")}
    r = pooled["y"] - pooled["yhat"]
    net = fit(pooled)
    p1 = pmf_scaled(net, pooled["yhat"], scale=1.0)
    print("%-3s n=%d  resid sd=%.2f  q01=%.1f q50=%.1f q99=%.1f max|r|=%.1f  cover@1.0=%.3f"
          % (arm, len(r), r.std(), np.quantile(r, .01), np.quantile(r, .5),
             np.quantile(r, .99), np.abs(r).max(), coverage_of(p1, pooled["y"])))
    print("     yhat range [%.1f, %.1f]  y range [%.1f, %.1f]  frac(y=125)=%.2f"
          % (pooled["yhat"].min(), pooled["yhat"].max(), pooled["y"].min(),
             pooled["y"].max(), np.mean(pooled["y"] >= 125)))
