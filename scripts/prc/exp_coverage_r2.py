"""PRC-R2 coverage gate: four scenarios, three methods, MC CIs, timing."""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/yndx/zxk/bearing-rul-pi/scripts/prc")
from simulator_v2 import make_scenario
from scenarios_v2 import SCENARIOS
from prc_v2 import prc_v2, baseline_iut, baseline_bonferroni, rmse

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
MODE = sys.argv[3] if len(sys.argv) > 3 else "conditional"
EXPECT = {"A_dominant": "A robustly better", "B_dominant": "B robustly better",
          "sign_flip": "protocol dependence confirmed", "near_zero": "inconclusive"}

def mc_ci(p, n):
    se = np.sqrt(p * (1 - p) / n)
    return [round(max(0, p - 1.96 * se), 3), round(min(1, p + 1.96 * se), 3)]

out = {}
for name, spec in SCENARIOS.items():
    cov_b = cov_t = 0
    dec_prc, dec_iut, dec_bonf = [], [], []
    widths, times = [], []
    for r in range(REPS):
        rng = np.random.default_rng(20_000 + r)
        sc = make_scenario(rng, spec, n_engine=N, inference=MODE, pop_reps=150_000)
        t0 = time.time()
        res = prc_v2(sc["err_a"], sc["err_b"], reps=499, seed=30_000 + r)
        times.append(time.time() - t0)
        lo, hi = res["band"]
        cov_b += bool(np.all((lo <= sc["delta_true"]) & (sc["delta_true"] <= hi)))
        cov_t += bool(res["theta_ci"][0] <= sc["theta_true"] <= res["theta_ci"][1])
        dec_prc.append(res["decision"])
        d = rmse(sc["err_b"]) - rmse(sc["err_a"])
        se = res["se"]
        dec_iut.append(baseline_iut(d, se))
        dec_bonf.append(baseline_bonferroni(sc["err_a"], sc["err_b"], reps=499,
                                            seed=30_000 + r)["decision"])
        widths.append(float(np.mean(hi - lo)))
    k = len(spec["protocols"])
    exp = EXPECT[name]
    out[name] = dict(
        band_cov=round(cov_b / REPS, 3), band_cov_mc_ci=mc_ci(cov_b / REPS, REPS),
        theta_cov=round(cov_t / REPS, 3), theta_cov_mc_ci=mc_ci(cov_t / REPS, REPS),
        prc={d: dec_prc.count(d) / REPS for d in set(dec_prc)},
        iut={d: dec_iut.count(d) / REPS for d in set(dec_iut)},
        bonferroni={d: dec_bonf.count(d) / REPS for d in set(dec_bonf)},
        prc_correct=dec_prc.count(exp) / REPS,
        iut_correct=dec_iut.count(exp) / REPS,
        mean_width=round(float(np.mean(widths)), 3),
        mean_seconds=round(float(np.mean(times)), 3),
        expected=exp)
    print(f"{name:11s} band_cov={out[name]['band_cov']:.2f}"
          f"{out[name]['band_cov_mc_ci']} theta_cov={out[name]['theta_cov']:.2f}"
          f" PRC_correct={out[name]['prc_correct']:.2f}"
          f" IUT_correct={out[name]['iut_correct']:.2f}"
          f" width={out[name]['mean_width']:.2f}", flush=True)

json.dump(dict(reps=REPS, n=N, mode=MODE, results=out),
          open(f"/home/yndx/zxk/bearing-rul-pi/results/prc/coverage_r2_{MODE}_{N}.json", "w"),
          indent=2)
print("wrote coverage_r2_%s_%d.json" % (MODE, N))
