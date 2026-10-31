"""First coverage gate: PRC vs per-protocol Bonferroni on known truth (§5.3)."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/yndx/zxk/bearing-rul-pi/scripts/prc")
from simulator import make_scenario
from prc import prc_inference, naive_per_protocol

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_ENGINE = int(sys.argv[2]) if len(sys.argv) > 2 else 100

scenarios = {
    "H+_clear": (1.0, 2.0, 0.5, 1.5),
    "H+_small": (0.5, 1.0, 0.3, 0.7),
    "H0_flip": (0.3, -0.3, 0.2, -0.2),
    "H0_zero": (0.0, 0.0, 0.0, 0.0),
}
out = {}
for name, pat in scenarios.items():
    cov_theta = cov_band = 0
    dec_prc, dec_naive = [], []
    t_lo_gap = []
    for r in range(REPS):
        rng = np.random.default_rng(1000 + r)
        sc = make_scenario(rng, n_engine=N_ENGINE, delta_pattern=pat, sigma_e=1.0,
                           pop_reps=30000)
        res = prc_inference(sc["err_a"], sc["err_b"], reps=399, seed=5000 + r)
        # band-based theta bounds: min(L) <= min(delta) <= min(U)
        lo = res["band"][:, 0].min()
        hi = res["band"][:, 1].min()
        cov_theta += (lo <= sc["theta_true"] <= hi)
        band = res["band"]
        d = res["delta_hat"]
        cov_band += bool(np.all((band[:, 0] <= sc["delta_true"]) & (sc["delta_true"] <= band[:, 1])))
        dec_prc.append(res["decision"])
        r2 = naive_per_protocol(sc["err_a"], sc["err_b"], reps=399, seed=5000 + r)
        dec_naive.append(r2["decision"])
        t_lo_gap.append(hi - lo)
    k = len(pat)
    out[name] = dict(
        theta_true=float(min(  # recomputed population truth for the record
            0 for _ in [])) if False else None,
        coverage_theta=cov_theta / REPS, coverage_band=cov_band / REPS,
        prc_decisions={d: dec_prc.count(d) / REPS for d in set(dec_prc)},
        naive_decisions={d: dec_naive.count(d) / REPS for d in set(dec_naive)},
        mean_theta_width=float(np.mean(t_lo_gap)))
    print(name, "theta_cov=%.3f band_cov=%.3f" % (cov_theta / REPS, cov_band / REPS),
          "PRC:", out[name]["prc_decisions"], "naive:", out[name]["naive_decisions"],
          flush=True)
json.dump(out, open("/home/yndx/zxk/bearing-rul-pi/results/prc/coverage_v1.json", "w"), indent=2)
