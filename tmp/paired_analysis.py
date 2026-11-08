"""Paired per-fold comparison: e2e-CQR vs e2e point + constant width.

Same subset, same fold, same validation engines -> the only difference is the
conditional quantile heads. Reporting paired deltas avoids the confound that
different folds have different engine difficulty.
"""
import json, os, statistics as st

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

def load(sub, fold, name):
    p = "experiments/punet_e2e/%s/f%d_s42/%s.json" % (sub, fold, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))

print("subset fold | CQR_winkler  base_winkler  delta_pct | CQR_picp base_picp")
deltas = []
for sub in SUBSETS:
    for f in range(5):
        a = load(sub, f, "e2e_cqr_none")
        b = load(sub, f, "e2e_point_none")
        if not a or not b:
            continue
        d = (a["winkler"] - b["winkler"]) / b["winkler"] * 100.0
        deltas.append((sub, f, d))
        print("%s f%d | %8.1f %8.1f %+8.1f%% | %.3f %.3f" % (
            sub, f, a["winkler"], b["winkler"], d, a["picp"], b["picp"]))

if deltas:
    wins = sum(1 for _, _, d in deltas if d < 0)
    print("\npaired folds: %d | CQR better: %d (%.0f%%) | mean delta %+.1f%% | median %+.1f%%" % (
        len(deltas), wins, 100.0 * wins / len(deltas),
        st.mean(d for _, _, d in deltas), st.median(d for _, _, d in deltas)))
