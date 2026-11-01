"""Coverage-vs-scale curve from the same cross-fitted nets choose_scale uses."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import K, ROOT, choose_scale, coverage_of, fit, load_dev, pmf_scaled  # noqa: E402

stage = Path(ROOT)
scales = [0.9, 1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.5]
for arm in ("A", "BE", "PE"):
    devs = [load_dev(stage, s, arm) for s in ("FD001", "FD002", "FD003", "FD004")]
    pooled = {k: np.concatenate([d[k] for d in devs]) for k in ("yhat", "y")}
    n = len(pooled["y"])
    order = np.random.default_rng(0).permutation(n)
    folds = np.array_split(order, 5)
    nets = []
    for f in folds:
        mask = np.ones(n, bool)
        mask[f] = False
        nets.append((fit({k: pooled[k][mask] for k in ("yhat", "y")}), f))
    curve = []
    for s in scales:
        ps = np.zeros((n, K), dtype=np.float32)
        for net, f in nets:
            ps[f] = pmf_scaled(net, pooled["yhat"][f], scale=s)
        curve.append("%.1f:%.3f" % (s, coverage_of(ps, pooled["y"])))
    print("%-3s %s   -> choose_scale=%s" % (arm, "  ".join(curve), choose_scale(pooled)))
