"""In-sample vs cross-fitted coverage of the calibration layer, per arm."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import K, ROOT, coverage_of, fit, load_dev, pmf_scaled  # noqa: E402

stage = Path(ROOT)
for arm in ("A", "BE", "PE"):
    devs = [load_dev(stage, s, arm) for s in ("FD001", "FD002", "FD003", "FD004")]
    pooled = {k: np.concatenate([d[k] for d in devs]) for k in ("yhat", "y")}
    n = len(pooled["y"])
    net = fit(pooled)
    ins = coverage_of(pmf_scaled(net, pooled["yhat"]), pooled["y"])

    order = np.random.default_rng(0).permutation(n)
    folds = np.array_split(order, 5)
    ps = np.zeros((n, K), dtype=np.float32)
    for f in folds:
        mask = np.ones(n, bool)
        mask[f] = False
        net_f = fit({k: pooled[k][mask] for k in ("yhat", "y")})
        ps[f] = pmf_scaled(net_f, pooled["yhat"][f])
    cf = coverage_of(ps, pooled["y"])
    print("%-3s in-sample %.3f   cross-fit %.3f   gap %+.3f" % (arm, ins, cf, cf - ins))
