"""Consistency gate: does E[delta_hat] match the analytic/population target?"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/yndx/zxk/bearing-rul-pi/scripts/prc")
from simulator_v2 import make_scenario
from scenarios_v2 import SCENARIOS

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
MODE = sys.argv[3] if len(sys.argv) > 3 else "conditional"

out = {}
for name, spec in SCENARIOS.items():
    d_hat, d_true, theta_hat = [], None, []
    for r in range(REPS):
        rng = np.random.default_rng(10_000 + r)
        sc = make_scenario(rng, spec, n_engine=N, inference=MODE, pop_reps=200_000)
        d_true = sc["delta_true"]
        ra = np.sqrt(np.nanmean(sc["err_a"] ** 2, axis=0))
        rb = np.sqrt(np.nanmean(sc["err_b"] ** 2, axis=0))
        d_hat.append(rb - ra)
        theta_hat.append(float((rb - ra).min()))
    d_hat = np.array(d_hat)
    mean = d_hat.mean(axis=0)
    se = d_hat.std(axis=0, ddof=1) / np.sqrt(REPS)
    bias = mean - d_true
    z = bias / np.maximum(se, 1e-12)
    out[name] = dict(delta_true=list(np.round(d_true, 3)),
                     mean_delta_hat=list(np.round(mean, 3)),
                     bias=list(np.round(bias, 3)),
                     z=list(np.round(z, 2)),
                     consistent=bool(np.all(np.abs(z) < 2.5)))
    print(f"{name:11s} true={np.round(d_true,2)} mean_hat={np.round(mean,2)} "
          f"bias={np.round(bias,2)} |z|max={np.abs(z).max():.1f} "
          f"{'OK' if out[name]['consistent'] else 'INCONSISTENT'}", flush=True)

json.dump(out, open(f"/home/yndx/zxk/bearing-rul-pi/results/prc/consistency_{MODE}.json", "w"), indent=2)
print("wrote consistency_%s.json" % MODE)
