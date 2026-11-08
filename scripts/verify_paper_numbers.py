"""Verify every quantitative claim in the manuscript against the artefacts.

Reads docs/manuscripts/v1.0-draft.md and the experiment outputs, recomputes each
number from source, and reports PASS/FAIL. Exit code 1 if anything mismatches.

Run on the machine that holds experiments/ (the a40 server).
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
MD = ROOT / "docs/manuscripts/v1.0-draft.md"
MATRIX = ROOT / "experiments/punet_matrix"
E2E = ROOT / "experiments/punet_e2e"
FEMTO = ROOT / "experiments/femto"

text = MD.read_text(encoding="utf-8") if MD.exists() else ""
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


def cells() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "docs/tables/cells_all.csv")
    return df


# ---- claim 1: 80 training cells, four seeds ----
n_cells = len(list(MATRIX.glob("*/f*_s*/preds_point.csv")))
check("80 training cells present", n_cells == 80, f"found {n_cells}")
seeds = sorted({int(p.parent.name.split("_s")[1]) for p in MATRIX.glob("*/f*_s*/preds_point.csv")})
check("four seeds", len(seeds) == 4, str(seeds))

# ---- claim 2: CQR vs constant width at matched coverage ----
df = cells()
rel = (df["cqr"] - df["const"]) / df["const"] * 100
wins, n = int((rel < 0).sum()), len(rel)
check("CQR wins 75/80", (wins, n) == (75, 80), f"{wins}/{n}")
check("mean -9.1%", abs(rel.mean() + 9.1) < 0.3, f"{rel.mean():+.2f}%")
check("median -10.8%", abs(rel.median() + 10.8) < 0.4, f"{rel.median():+.2f}%")
if text:
    check("manuscript states 75 of 80", "75 of 80" in text)

# ---- claim 3: per-subset numbers ----
want = {"FD001": -7.6, "FD002": -14.1, "FD003": -6.5, "FD004": -8.2}
for sub, val in want.items():
    d = df[df.subset == sub]
    r = (d["cqr"] - d["const"]) / d["const"] * 100
    check(f"{sub} mean {val:+.1f}%", abs(r.mean() - val) < 0.6, f"{r.mean():+.2f}%")

# ---- claim 4: nominal level 19/20, median -8.7% ----
def load(s, f, name):
    p = E2E / s / f"f{f}_s42" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None

d = []
for s in ["FD001", "FD002", "FD003", "FD004"]:
    for f in range(5):
        a, b = load(s, f, "e2e_cqr_none"), load(s, f, "e2e_point_none")
        if a and b:
            d.append((a["winkler"] - b["winkler"]) / b["winkler"] * 100)
check("nominal 19/20", sum(1 for x in d if x < 0) == 19, f"{sum(1 for x in d if x < 0)}/{len(d)}")
check("nominal median -8.7%", abs(st.median(d) + 8.7) < 0.3, f"{st.median(d):+.2f}%")

# ---- claim 5: A4 permutation +0.2% over 20 folds ----
a4 = []
for s in ["FD001", "FD002", "FD003", "FD004"]:
    for f in range(5):
        p1 = E2E / s / f"f{f}_s42" / "e2e_cqr_A4.json"
        p2 = E2E / s / f"f{f}_s42" / "e2e_cqr_none.json"
        if p1.exists() and p2.exists():
            a, b = json.loads(p1.read_text()), json.loads(p2.read_text())
            a4.append((a["winkler"] - b["winkler"]) / b["winkler"] * 100)
check("A4 mean +0.2%", abs(st.mean(a4) - 0.2) < 0.5, f"{st.mean(a4):+.2f}%")
check("A4 11/20 folds worse", sum(1 for x in a4 if x > 0) == 11, f"{sum(1 for x in a4 if x > 0)}/20")

# ---- claim 6: per-engine minimum coverage 80-100% below 0.80 ----
fracs = {}
for s in ["FD001", "FD002", "FD003", "FD004"]:
    mins = []
    for f in range(5):
        p = E2E / s / f"f{f}_s42" / "e2e_cqr_none.json"
        if p.exists():
            j = json.loads(p.read_text())
            if j.get("engine_min") is not None:
                mins.append(j["engine_min"])
    fracs[s] = float(np.mean(np.array(mins) < 0.80))
lo, hi = min(fracs.values()), max(fracs.values())
check("engine_min<0.80 in 80-100%", abs(lo - 0.80) < 0.05 and abs(hi - 1.0) < 0.01, f"{lo:.2f}-{hi:.2f}")

# ---- claim 7: FEMTO 5 of 6, mean 12.2% ----
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_femto import at, pts_cqr, pts_scaled  # noqa: E402

rels = []
for d0 in sorted(FEMTO.glob("Bearing*")):
    q, p = d0 / "preds_q.csv", d0 / "preds_p.csv"
    if not (q.exists() and p.exists()):
        continue
    qq, pp = pd.read_csv(q), pd.read_csv(p)
    tq, tp = qq.split == "test", pp.split == "test"
    y, lo, hi = qq.y.values, qq.lo.values, qq.hi.values
    yp, mu = pp.y.values, pp.mu.values
    c = at(pts_cqr(y[tq], lo[tq], hi[tq]))
    k = at(pts_scaled(yp[tp], mu[tp], np.ones(tp.sum())))
    rels.append((c - k) / k * 100)
check("FEMTO 5/6 better", sum(1 for x in rels if x < 0) == 5, f"{sum(1 for x in rels if x < 0)}/{len(rels)}")
check("FEMTO mean -12.2%", abs(np.mean(rels) + 12.2) < 1.0, f"{np.mean(rels):+.1f}%")

# ---- report ----
ok = sum(1 for _, v, _ in results if v)
print(f"{'CHECK':<42} RESULT   DETAIL")
for name, v, detail in results:
    print(f"{name:<42} {'PASS' if v else 'FAIL':<8} {detail}")
print(f"\n{ok}/{len(results)} checks passed")
raise SystemExit(0 if ok == len(results) else 1)
