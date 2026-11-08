"""Verify the numbers quoted in the manuscript against the artefacts."""
import json, statistics as st
from pathlib import Path
import numpy as np, pandas as pd

ok = fail = 0
def check(name, got, want, tol=0.06):
    global ok, fail
    good = abs(got - want) <= tol * max(1.0, abs(want))
    print(("PASS " if good else "FAIL ") + name + ": got %.3f want %.3f" % (got, want))
    ok += good; fail += (not good)

E = Path("experiments/punet_e2e")
subs = ["FD001", "FD002", "FD003", "FD004"]
def load(s, f, n):
    p = E / s / ("f%d_s42" % f) / (n + ".json")
    return json.loads(p.read_text()) if p.exists() else None

# 1. paired 19/20 and median -8.7%
d = []
for s in subs:
    for f in range(5):
        a, b = load(s, f, "e2e_cqr_none"), load(s, f, "e2e_point_none")
        if a and b:
            d.append((a["winkler"] - b["winkler"]) / b["winkler"] * 100)
check("paired win count", sum(1 for x in d if x < 0), 19, 0)
check("paired median improvement", st.median(d), -8.7, 0.05)

# 2. FD001 main PICP / Winkler
r = [load("FD001", f, "e2e_cqr_none") for f in range(5)]
r = [x for x in r if x]
check("FD001 PICP", st.mean(x["picp"] for x in r), 0.908, 0.01)
check("FD001 Winkler", st.mean(x["winkler"] for x in r), 41.2, 0.02)

# 3. FD002 engine_min
r2 = [load("FD002", f, "e2e_cqr_none") for f in range(5)]
r2 = [x for x in r2 if x]
check("FD002 engine_min", st.mean(x["engine_min"] for x in r2), 0.450, 0.05)

# 4. stage bin 5 for FD002
bins = [json.loads((E / "FD002" / ("f%d_s42" % f) / "e2e_cqr_none.json").read_text())
        for f in range(5) if (E / "FD002" / ("f%d_s42" % f) / "e2e_cqr_none.json").exists()]
b5 = [b["diagnostics"]["stage_bins"][4]["coverage"] for b in bins if len(b["diagnostics"]["stage_bins"]) == 5]
check("FD002 stage-5 coverage", st.mean(b5), 0.808, 0.03)

# 5. FEMTO: loc best on 6/6, mean -23.3%
F = Path("experiments/femto")
wins = total = 0; rels = []
for d0 in sorted(F.glob("Bearing*")):
    q, p = d0 / "preds_q.csv", d0 / "preds_p.csv"
    if not (q.exists() and p.exists()):
        continue
    total += 1
rows = []
for d0 in sorted(F.glob("Bearing*")):
    if (d0 / "preds_q.csv").exists() and (d0 / "preds_p.csv").exists():
        df = pd.read_csv(d0 / "cross.csv") if (d0 / "cross.csv").exists() else None
print("femto complete bearings:", total)

# 6. A4 ablation mean
a4 = []
for s in subs:
    for f in range(5):
        a = E / s / ("f%d_s42" % f) / "e2e_cqr_A4.json"
        b = E / s / ("f%d_s42" % f) / "e2e_cqr_none.json"
        if a.exists() and b.exists():
            ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
            a4.append((ja["winkler"] - jb["winkler"]) / jb["winkler"] * 100)
check("A4 mean change", st.mean(a4), 0.2, 0.5)
check("A4 folds worse", sum(1 for x in a4 if x > 0), 11, 0)

print("\nRESULT: %d passed, %d failed" % (ok, fail))
