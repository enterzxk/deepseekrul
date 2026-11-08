import json, statistics as st
from pathlib import Path
E = Path("experiments/punet_e2e")
subs = ["FD001","FD002","FD003","FD004"]
print("subset  PICP  eng_min  frac<0.80  bins1..5")
for s in subs:
    js = []
    for f in range(5):
        p = E / s / ("f%d_s42" % f) / "e2e_cqr_none.json"
        if p.exists():
            js.append(json.loads(p.read_text()))
    picp = st.mean(j["picp"] for j in js)
    emin = st.mean(j["engine_min"] for j in js)
    frac = sum(1 for j in js if j["engine_min"] < 0.80) / len(js)
    bins = []
    for b in range(5):
        vals = [j["diagnostics"]["stage_bins"][b]["coverage"] for j in js
                if len(j["diagnostics"]["stage_bins"]) == 5]
        bins.append(st.mean(vals) if vals else float("nan"))
    print("%s  %.3f  %.3f  %.2f  %s" % (s, picp, emin, frac, " ".join("%.3f" % x for x in bins)))
